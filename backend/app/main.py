import csv
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.context_graph import (
    RELATIONSHIP_TYPES,
    add_node,
    add_relationship,
    graph_snapshot,
    neighbors,
    node_key,
    seed_context_graph,
    similar_situations,
)
from app.database import SessionLocal, engine, get_db
from app.init_db import create_tables
from app.ml.features import FeatureDataError
from app.ml.model_registry import DEFAULT_MODEL_NAME, ModelName
from app.ml.predict import (
    DecisionMakerNotFoundError,
    ModelNotTrainedError,
    NoAvailableAlternativesError,
    PreferenceNotFoundError,
    ScenarioNotFoundError,
    predict_choice,
)
from app.models import (
    Alternative,
    Assessment,
    ContextNode,
    ContextRelationship,
    Decision,
    DecisionEvent,
    DecisionMaker,
    ExtractedRule,
    ModelExplanation,
    Outcome,
    Prediction,
    Preference,
    RuleApplication,
    Situation,
)
from app.rule_api import router as rule_router

CONFIG_PATH = Path(__file__).parent / "config" / "decision_pipeline.json"
TRAINING_DATA_PATH = Path(__file__).parent.parent / "data" / "training"


def normalize_weights(weights: Dict[str, float], criterion_keys: set[str]) -> Dict[str, float]:
    if set(weights) != criterion_keys:
        raise ValueError(f"Weights must contain exactly: {', '.join(sorted(criterion_keys))}")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Weights must be finite and cannot be negative.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return {key: value / total for key, value in weights.items()}


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    criteria = config.get("criteria", [])
    options = config.get("options", [])
    criterion_keys = {criterion["key"] for criterion in criteria}
    if not criteria or not options or len(criterion_keys) != len(criteria):
        raise ValueError("Configuration must define unique criteria and at least one option.")

    normalize_weights(config["operational_weights"], criterion_keys)
    for option in options:
        if set(option.get("metrics", {})) != criterion_keys:
            raise ValueError(f"Option {option.get('id', '<unknown>')} must define every criterion.")
        if any(
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 100
            for score in option["metrics"].values()
        ):
            raise ValueError("Option metric scores must be finite numbers from 0 to 100.")
    for profile in config.get("decision_makers", []):
        normalize_weights(profile["preferences"], criterion_keys)
    return config


CONFIG = load_config()
CRITERIA = CONFIG["criteria"]
CRITERION_ORDER = [criterion["key"] for criterion in CRITERIA]
CRITERION_KEYS = {criterion["key"] for criterion in CRITERIA}
TWIN_STATE = deepcopy(CONFIG["digital_twin"]["state"])
TWIN_EVENTS: list[dict] = []


def read_csv_rows(filename: str) -> list[dict]:
    with (TRAINING_DATA_PATH / filename).open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_training_data() -> dict:
    scenarios = {}
    for row in read_csv_rows("scenarios.csv"):
        scenario_id = row["scenario_id"]
        if scenario_id in scenarios:
            raise ValueError(f"Duplicate training scenario ID: {scenario_id}")
        additional_parameters = {}
        for key in (
            "demand_increase_percent",
            "current_inventory_days",
            "required_inventory_days",
            "available_budget_lakh",
            "threat_level",
        ):
            value = row.get(key, "").strip()
            if value:
                additional_parameters[key] = (
                    value if key == "threat_level"
                    else float(value) if key == "available_budget_lakh"
                    else int(value)
                )
        scenarios[scenario_id] = {
            "id": scenario_id,
            "title": row["title"],
            "situation": row["situation"],
            "decision_conflict": row["decision_conflict"],
            "required_units": int(row["required_units"]),
            "required_within_days": int(row["required_within_days"]),
            "stockout_risk": row["stockout_risk"],
            "location": row["location"],
            "product": row["product"],
            "operational_weights": json.loads(row["operational_weights_json"]),
            "additional_parameters": additional_parameters,
            "context": {
                "Product": row["product"],
                "Required quantity": f"{int(row['required_units']):,} units",
                "Required within": f"{row['required_within_days']} days",
                "Stockout risk": row["stockout_risk"],
                "Location": row["location"],
                **additional_parameters,
            },
            "data_source": "synthetic",
            "options": [],
        }

    for row in read_csv_rows("options.csv"):
        scenario = scenarios.get(row["scenario_id"])
        if scenario is None:
            raise ValueError(f"Option references unknown scenario: {row['scenario_id']}")
        scenario["options"].append({
            "id": row["option_id"],
            "name": row["name"],
            "metrics": json.loads(row["metrics_json"]),
            "attributes": json.loads(row["attributes_json"]),
            "notes": row["notes"],
            "available": row["available"].lower() == "true",
        })

    profile_rows = read_csv_rows("decision_maker_profiles.csv")
    profiles = []
    for row in profile_rows:
        preferences = {
            key: float(row[key])
            for key in CRITERION_ORDER
        }
        normalize_weights(preferences, CRITERION_KEYS)
        profiles.append({
            "id": row["decision_maker_id"],
            "name": row["name"],
            "role": row["role"],
            "preferences": preferences,
            "description": row["description"],
        })

    decisions = read_csv_rows("decisions.csv")
    outcomes = read_csv_rows("outcomes.csv")
    outcome_by_decision = {row["decision_id"]: row for row in outcomes}
    if len(outcome_by_decision) != len(outcomes) or set(outcome_by_decision) != {row["decision_id"] for row in decisions}:
        raise ValueError("Every training decision must have exactly one outcome record.")
    profile_ids = {profile["id"] for profile in profiles}
    for scenario_id, scenario in scenarios.items():
        option_ids = [option["id"] for option in scenario["options"]]
        if len(scenario["options"]) != 3 or len(set(option_ids)) != 3:
            raise ValueError(f"Training scenario {scenario_id} must have exactly three options.")
        if any(
            set(option["metrics"]) != CRITERION_KEYS
            or any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100 for score in option["metrics"].values())
            for option in scenario["options"]
        ):
            raise ValueError(f"Training scenario {scenario_id} options must define every criterion.")
    for decision in decisions:
        if decision["scenario_id"] not in scenarios or decision["decision_maker_id"] not in profile_ids:
            raise ValueError(f"Training decision {decision['decision_id']} has an unknown scenario or decision-maker.")
        option_ids = {option["id"] for option in scenarios[decision["scenario_id"]]["options"]}
        if decision["recorded_choice_id"] not in option_ids:
            raise ValueError(f"Training decision {decision['decision_id']} references an unknown option.")
        raw_probabilities = decision.get("predicted_probabilities_json", "")
        if raw_probabilities:
            probabilities = json.loads(raw_probabilities)
            if set(probabilities) != option_ids or any(
                not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
                for probability in probabilities.values()
            ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
                raise ValueError(
                    f"Training decision {decision['decision_id']} must define a valid probability distribution for every option."
                )
            decision["predicted_probabilities"] = probabilities
        decision["outcome"] = outcome_by_decision[decision["decision_id"]]

    decision_pairs = {(row["scenario_id"], row["decision_maker_id"]) for row in decisions}
    expected_pairs = {
        (scenario_id, profile["id"])
        for scenario_id in scenarios
        for profile in profiles
    }
    expected_decision_count = len(scenarios) * len(profiles)
    if (
        len(scenarios) < 8
        or len(profiles) != 4
        or len(decisions) != expected_decision_count
        or decision_pairs != expected_pairs
        or len(decision_pairs) != len(decisions)
    ):
        raise ValueError(
            "Training data must contain at least 8 scenarios, exactly 4 decision-maker profiles, "
            "and one decision per scenario/profile pair."
        )
    return {"scenarios": scenarios, "profiles": profiles, "decisions": decisions}


TRAINING_DATA = load_training_data()

app = FastAPI(
    title="Decision Intelligence Pipeline API",
    description="Configurable three-stage suitability, preference, and choice-probability prototype",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rule_router)


@app.on_event("startup")
def startup_event():
    create_tables()
    with SessionLocal() as db:
        seed_context_graph(db, TRAINING_DATA, CRITERIA)


class PreferenceProfile(BaseModel):
    decision_maker_id: str
    weights: Dict[str, float] = Field(description="Preference weight for each configured criterion.")


class AssessRequest(BaseModel):
    scenario_id: str | None = None
    operational_weights: Dict[str, float] | None = None
    preference_profiles: List[PreferenceProfile] | None = None
    temperature: float | None = Field(default=None, gt=0, le=100)


class MLPredictionRequest(BaseModel):
    scenario_id: str = Field(min_length=1)
    decision_maker_id: str = Field(min_length=1)
    model_name: ModelName = DEFAULT_MODEL_NAME


class MLDecisionRequest(BaseModel):
    prediction_event_id: str = Field(min_length=1, max_length=60)
    chosen_option_id: str = Field(min_length=1, max_length=30)
    rationale: str = Field(min_length=1, max_length=1000)
    operational_objective: str = Field(min_length=1, max_length=500)
    constraints: List[str] = Field(default_factory=list, max_length=20)
    influencing_factors: List[str] = Field(default_factory=list, max_length=20)
    reasoning_steps: List[str] = Field(default_factory=list, max_length=20)
    considered_alternative_codes: List[str] = Field(default_factory=list, max_length=20)
    confidence: float | None = Field(default=None, ge=0, le=1)


class HumanDecisionRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=50)
    decision_maker_id: str = Field(min_length=1, max_length=50)
    chosen_option_id: str = Field(min_length=1, max_length=30)
    prediction_event_id: str | None = Field(default=None, max_length=60)
    operational_objective: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)
    constraints: List[str] = Field(default_factory=list, max_length=20)
    influencing_factors: List[str] = Field(default_factory=list, max_length=20)
    reasoning_steps: List[str] = Field(default_factory=list, max_length=20)
    considered_alternative_codes: List[str] = Field(default_factory=list, max_length=20)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TwinUpdateRequest(BaseModel):
    source: str = Field(min_length=1, max_length=100)
    entity_type: Literal["inventory", "supplier", "shipment", "order"]
    entity_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    value: Any


class DecisionFeedbackRequest(BaseModel):
    scenario_id: str | None = None
    decision_maker_id: str
    chosen_option_id: str
    rationale: str = Field(default="", max_length=1000)
    operational_objective: str | None = Field(default=None, max_length=500)
    constraints: List[str] = Field(default_factory=list, max_length=20)
    influencing_factors: List[str] = Field(default_factory=list, max_length=20)
    reasoning_steps: List[str] = Field(default_factory=list, max_length=20)
    considered_alternative_codes: List[str] = Field(default_factory=list, max_length=20)
    confidence: float | None = Field(default=None, ge=0, le=1)


class OutcomeFeedbackRequest(BaseModel):
    status: Literal["fulfilled", "delayed", "failed", "partial"]
    notes: str = Field(default="", max_length=1000)
    twin_updates: List[TwinUpdateRequest] = Field(default_factory=list)


class ContextNodeRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=250)
    context_type: Literal["constraint", "event", "policy", "external_condition", "evidence", "source"]
    situation_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: str = Field(min_length=1, max_length=500)


class ContextRelationshipRequest(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: str = Field(min_length=1, max_length=500)


TWIN_UPDATE_FIELDS = {
    "inventory": {"on_hand_units", "reserved_units", "daily_demand_units"},
    "supplier": {"capacity_units", "is_available", *CRITERION_KEYS},
    "shipment": {"status", "location", "estimated_delivery_days"},
    "order": {"status"},
}
TWIN_COLLECTIONS = {
    "inventory": "inventory",
    "supplier": "suppliers",
    "shipment": "shipments",
    "order": "orders",
}


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_twin_update(update: TwinUpdateRequest) -> dict:
    if update.field not in TWIN_UPDATE_FIELDS[update.entity_type]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{update.field}' cannot be updated for {update.entity_type}.",
        )

    record = next(
        (item for item in TWIN_STATE[TWIN_COLLECTIONS[update.entity_type]] if item["id"] == update.entity_id),
        None,
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown {update.entity_type} ID: {update.entity_id}",
        )

    target = record.get("metrics", {}) if update.entity_type == "supplier" and update.field in CRITERION_KEYS else record
    current_value = target.get(update.field)
    if isinstance(current_value, bool):
        valid_value = isinstance(update.value, bool)
    elif isinstance(current_value, (int, float)):
        valid_value = isinstance(update.value, (int, float)) and not isinstance(update.value, bool)
    else:
        valid_value = isinstance(update.value, str) and bool(update.value.strip())
    if not valid_value:
        raise HTTPException(
            status_code=422,
            detail=f"Value type does not match the current value for '{update.field}'.",
        )
    if isinstance(update.value, (int, float)) and (
        not math.isfinite(update.value)
        or update.value < 0
        or (update.field in CRITERION_KEYS and update.value > 100)
    ):
        raise HTTPException(status_code=422, detail="Numeric twin values must be finite and non-negative; scores cannot exceed 100.")

    target[update.field] = update.value
    TWIN_STATE["revision"] += 1
    TWIN_STATE["last_updated"] = current_timestamp()
    TWIN_STATE["last_source"] = update.source
    event = {
        "revision": TWIN_STATE["revision"],
        "timestamp": TWIN_STATE["last_updated"],
        "source": update.source,
        "entity_type": update.entity_type,
        "entity_id": update.entity_id,
        "field": update.field,
        "value": update.value,
    }
    TWIN_EVENTS.insert(0, event)
    return event


def utility(option: dict, weights: Dict[str, float]) -> float:
    return sum(option["metrics"][key] * weights[key] for key in CRITERION_ORDER)


def softmax_probabilities(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    scaled = {key: value / temperature for key, value in scores.items()}
    maximum = max(scaled.values())
    exponentials = {key: math.exp(value - maximum) for key, value in scaled.items()}
    total = sum(exponentials.values())
    probabilities = {key: round(value / total * 100, 1) for key, value in exponentials.items()}
    highest_probability_key = max(exponentials, key=exponentials.get)
    probabilities[highest_probability_key] = round(
        probabilities[highest_probability_key] + 100 - sum(probabilities.values()),
        1,
    )
    return probabilities


def ranked_scores(options: list, weights: Dict[str, float]) -> list:
    rows = []
    for option in options:
        rows.append({
            "option_id": option["id"],
            "option_name": option["name"],
            "score": round(utility(option, weights), 2),
            "attributes": option.get("attributes", {}),
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def scenario_data(scenario_id: str | None) -> tuple[dict, list, bool]:
    if scenario_id is None and CONFIG["domain_name"] != "Supply Chain":
        return CONFIG["scenario"], CONFIG["options"], False
    selected_id = scenario_id or next(iter(TRAINING_DATA["scenarios"]))
    scenario = TRAINING_DATA["scenarios"].get(selected_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario ID: {selected_id}")
    return scenario, scenario["options"], True


def latest_twin_value(entity_id: str, field: str, default: Any) -> Any:
    event = next(
        (
            event for event in TWIN_EVENTS
            if event["entity_type"] == "supplier"
            and event["entity_id"] == entity_id
            and event["field"] == field
        ),
        None,
    )
    return event["value"] if event else default


def calculate_assessment(request: AssessRequest) -> dict:
    scenario, configured_options, is_training_scenario = scenario_data(request.scenario_id)
    required_units = scenario.get("required_units", 0)
    twin_suppliers = {supplier["id"]: supplier for supplier in TWIN_STATE["suppliers"]}
    options = []
    for configured_option in configured_options:
        twin_supplier = twin_suppliers.get(configured_option["id"])
        if twin_supplier is None:
            continue
        capacity = latest_twin_value(
            configured_option["id"],
            "capacity_units",
            configured_option.get("attributes", {}).get("capacity_units", twin_supplier["capacity_units"])
            if is_training_scenario
            else twin_supplier["capacity_units"],
        )
        is_available = latest_twin_value(
            configured_option["id"],
            "is_available",
            twin_supplier["is_available"],
        )
        if (
            not configured_option.get("available", True)
            or not is_available
            or capacity < required_units
        ):
            continue
        option = deepcopy(configured_option)
        if not is_training_scenario:
            option["metrics"].update(twin_supplier["metrics"])
        else:
            for event in reversed(TWIN_EVENTS):
                if (
                    event["entity_type"] == "supplier"
                    and event["entity_id"] == option["id"]
                    and event["field"] in CRITERION_KEYS
                ):
                    option["metrics"][event["field"]] = event["value"]
        option.setdefault("attributes", {})["capacity_units"] = capacity
        options.append(option)
    if not options:
        raise HTTPException(status_code=409, detail="No available supplier has enough capacity for this order.")
    temperature = request.temperature or CONFIG.get("probability_temperature", 8.0)
    try:
        operational_weights = normalize_weights(
            request.operational_weights or scenario.get("operational_weights", CONFIG["operational_weights"]),
            CRITERION_KEYS,
        )
        profiles = request.preference_profiles
        if profiles is None:
            configured_profiles = TRAINING_DATA["profiles"] if is_training_scenario else CONFIG.get("decision_makers", [])
            profiles = [
                PreferenceProfile(decision_maker_id=person["id"], weights=person["preferences"])
                for person in configured_profiles
            ]

        configured_profiles = TRAINING_DATA["profiles"] if is_training_scenario else CONFIG.get("decision_makers", [])
        people_by_id = {person["id"]: person for person in configured_profiles}
        stage1_rows = ranked_scores(options, operational_weights)
        stage2 = []
        stage3 = []
        for profile in profiles:
            person = people_by_id.get(profile.decision_maker_id)
            if person is None:
                raise ValueError(f"Unknown decision-maker: {profile.decision_maker_id}")
            weights = normalize_weights(profile.weights, CRITERION_KEYS)
            rows = ranked_scores(options, weights)
            scores = {row["option_id"]: row["score"] for row in rows}
            probabilities = softmax_probabilities(scores, temperature)
            top_row = rows[0]
            stage2.append({
                "decision_maker_id": person["id"],
                "decision_maker_name": person["name"],
                "role": person.get("role", ""),
                "description": person.get("description", ""),
                "weights": weights,
                "rankings": rows,
                "top_choice": top_row["option_name"],
                "top_choice_id": top_row["option_id"],
            })
            stage3.append({
                "decision_maker_id": person["id"],
                "decision_maker_name": person["name"],
                "probabilities": [
                    {
                        "option_id": option["id"],
                        "option_name": option["name"],
                        "probability_percent": probabilities[option["id"]],
                        "utility_score": scores[option["id"]],
                    }
                    for option in options
                ],
                "most_likely_choice": top_row["option_name"],
                "most_likely_choice_probability_percent": probabilities[top_row["option_id"]],
            })
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    stage1_winner = stage1_rows[0]["option_name"]
    comparisons = [
        {
            "decision_maker": person["decision_maker_name"],
            "stage1_top_option": stage1_winner,
            "personal_top_option": person["top_choice"],
            "matches_stage1": person["top_choice"] == stage1_winner,
            "explanation": (
                f"{person['decision_maker_name']} selects {person['top_choice']} under their own weights. "
                + ("This matches" if person["top_choice"] == stage1_winner else "This differs from")
                + f" the Stage 1 top-ranked option ({stage1_winner})."
            ),
        }
        for person in stage2
    ]

    inventory_twin = (
        TWIN_STATE["inventory"][0]
        if TWIN_STATE["inventory"] and scenario["id"] == next(iter(TRAINING_DATA["scenarios"]))
        else None
    )
    available_inventory = (
        max(0, inventory_twin["on_hand_units"] - inventory_twin["reserved_units"])
        if inventory_twin
        else None
    )
    days_of_inventory = (
        round(available_inventory / inventory_twin["daily_demand_units"], 1)
        if inventory_twin and inventory_twin["daily_demand_units"] > 0
        else None
    )
    return {
        "domain_name": CONFIG["domain_name"],
        "scenario": scenario,
        "scenario_id": scenario["id"],
        "operational_snapshot": {
            "twin_revision": TWIN_STATE["revision"],
            "available_inventory_units": available_inventory,
            "days_of_inventory": days_of_inventory,
            "required_within_days": CONFIG["scenario"].get("required_within_days"),
            "available_supplier_count": len(options),
        },
        "criteria": CRITERIA,
        "attribute_labels": CONFIG.get("attribute_labels", {}),
        "stage1": {
            "weights": operational_weights,
            "rankings": stage1_rows,
            "top_option": stage1_winner,
            "interpretation": "Suitability is a weighted fit score, not a probability.",
        },
        "stage2": stage2,
        "stage3": {
            "temperature": temperature,
            "decision_maker_probabilities": stage3,
            "comparisons": comparisons,
            "interpretation": "Probabilities are illustrative softmax estimates, not calibrated real-world likelihoods.",
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "decision-intelligence-pipeline"}


@app.get("/api/config")
@app.get("/api/mock-data")
def get_config(scenario_id: str | None = None):
    scenario, options, _ = scenario_data(scenario_id)
    return {
        **CONFIG,
        "scenario": scenario,
        "operational_weights": scenario.get("operational_weights", CONFIG["operational_weights"]),
        "options": deepcopy(options),
        "decision_makers": deepcopy(TRAINING_DATA["profiles"]),
        "training_scenarios": [
            {
                "id": item["id"],
                "title": item["title"],
                "decision_conflict": item["decision_conflict"],
            }
            for item in TRAINING_DATA["scenarios"].values()
        ],
    }

@app.post("/api/assess")
def assess(request: AssessRequest):
    return calculate_assessment(request)


@app.get("/api/training/decisions")
def get_training_decisions(scenario_id: str | None = None):
    decisions = [
        {
            **decision,
            "scenario_title": TRAINING_DATA["scenarios"][decision["scenario_id"]]["title"],
        }
        for decision in TRAINING_DATA["decisions"]
        if scenario_id is None or decision["scenario_id"] == scenario_id
    ]
    return {
        "synthetic": True,
        "count": len(decisions),
        "decisions": decisions,
    }


@app.get("/api/training/evaluation")
def evaluate_training_data():
    evaluations = []
    brier_totals = []
    correct = 0
    scenario_choice_counts: dict[str, dict[str, int]] = {}
    for decision in TRAINING_DATA["decisions"]:
        assessment = calculate_assessment(AssessRequest(scenario_id=decision["scenario_id"]))
        person = next(
            item for item in assessment["stage3"]["decision_maker_probabilities"]
            if item["decision_maker_id"] == decision["decision_maker_id"]
        )
        probabilities = {
            option["option_id"]: option["probability_percent"] / 100
            for option in person["probabilities"]
        }
        recorded_choice = decision["recorded_choice_id"]
        scenario_counts = scenario_choice_counts.setdefault(decision["scenario_id"], {})
        scenario_counts[recorded_choice] = scenario_counts.get(recorded_choice, 0) + 1
        prediction_matches = person["most_likely_choice"] == next(
            option["name"]
            for option in TRAINING_DATA["scenarios"][decision["scenario_id"]]["options"]
            if option["id"] == recorded_choice
        )
        brier_score = sum(
            (probabilities.get(option["id"], 0) - (1 if option["id"] == recorded_choice else 0)) ** 2
            for option in TRAINING_DATA["scenarios"][decision["scenario_id"]]["options"]
        )
        correct += int(prediction_matches)
        brier_totals.append(brier_score)
        option_probability = round(probabilities.get(recorded_choice, 0) * 100, 1)
        evaluations.append({
            "decision_id": decision["decision_id"],
            "scenario_id": decision["scenario_id"],
            "scenario_title": TRAINING_DATA["scenarios"][decision["scenario_id"]]["title"],
            "decision_maker_id": decision["decision_maker_id"],
            "decision_maker_name": next(
                profile["name"] for profile in TRAINING_DATA["profiles"]
                if profile["id"] == decision["decision_maker_id"]
            ),
            "predicted_choice": person["most_likely_choice"],
            "recorded_choice": next(
                option["name"]
                for option in TRAINING_DATA["scenarios"][decision["scenario_id"]]["options"]
                if option["id"] == recorded_choice
            ),
            "recorded_choice_probability_percent": option_probability,
            "prediction_matches_recorded": prediction_matches,
            "brier_score": round(brier_score, 4),
            "rationale": decision["rationale"],
            "outcome": decision["outcome"],
        })

    sample_count = len(evaluations)
    scenario_disagreement = []
    for scenario_id, choice_counts in scenario_choice_counts.items():
        decision_count = sum(choice_counts.values())
        same_choice_pairs = sum(count * (count - 1) // 2 for count in choice_counts.values())
        total_pairs = decision_count * (decision_count - 1) // 2
        scenario_disagreement.append({
            "scenario_id": scenario_id,
            "scenario_title": TRAINING_DATA["scenarios"][scenario_id]["title"],
            "distinct_recorded_choices": len(choice_counts),
            "pairwise_disagreement_count": total_pairs - same_choice_pairs,
            "decision_count": decision_count,
            "choice_counts": choice_counts,
        })
    return {
        "synthetic": True,
        "sample_count": sample_count,
        "top_choice_accuracy_percent": round(correct / sample_count * 100, 1) if sample_count else 0,
        "mean_multiclass_brier_score": round(sum(brier_totals) / sample_count, 4) if sample_count else 0,
        "interpretation": "Descriptive fit against synthetic recorded choices only; not evidence of real-world predictive accuracy.",
        "scenario_disagreement": scenario_disagreement,
        "records": evaluations,
    }


@app.post("/api/ml/predict")
def predict_ml_choice(request: MLPredictionRequest, db: Session = Depends(get_db)):
    try:
        prediction = predict_choice(
            db,
            request.scenario_id,
            request.decision_maker_id,
            model_name=request.model_name,
        )
    except ModelNotTrainedError as error:
        raise HTTPException(
            status_code=503,
            detail="The ML choice model is not trained. Run `python -m app.ml.train` from backend.",
        ) from error
    except (ScenarioNotFoundError, DecisionMakerNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (PreferenceNotFoundError, NoAvailableAlternativesError, FeatureDataError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return persist_ml_prediction(db, prediction)


def persist_ml_prediction(db: Session, prediction: dict) -> dict:
    scenario_id = prediction["scenario_id"]
    decision_maker_id = prediction["decision_maker"]["id"]
    preference = db.scalar(
        select(Preference)
        .where(
            Preference.decision_maker_id == decision_maker_id,
            Preference.is_current.is_(True),
        )
        .order_by(Preference.version.desc())
    )
    if preference is None:
        raise HTTPException(
            status_code=422,
            detail=f"No current preference profile exists for decision-maker {decision_maker_id}.",
        )

    event_number = (db.scalar(
        select(func.max(DecisionEvent.event_number)).where(
            DecisionEvent.situation_id == scenario_id,
            DecisionEvent.decision_maker_id == decision_maker_id,
        )
    ) or 0) + 1
    prediction_run_id = f"PRD-{uuid4().hex[:12].upper()}"
    event_id = f"{scenario_id}:{decision_maker_id}:ml:{uuid4().hex}"
    model_version = f"ml-{prediction['model']['version']}"
    model_name = prediction["model"]["name"]
    event_context = {
        "source": "ml_prediction",
        "data_source": "model",
        "scenario_id": scenario_id,
        "prediction_run_id": prediction_run_id,
        "model_name": model_name,
        "model_version": model_version,
        "training_data_source": prediction["model"]["training_data_source"],
        "reasoning_snapshot": prediction["reasoning_snapshot"],
    }
    event = DecisionEvent(
        id=event_id,
        situation_id=scenario_id,
        decision_maker_id=decision_maker_id,
        preference_id=preference.id,
        event_number=event_number,
        status="predicted",
        event_context=event_context,
        data_source="model",
    )
    db.add(event)
    db.flush()

    provenance = f"model:{model_name}"
    event_key = node_key("DecisionEvent", event_id)
    situation_key = node_key("Situation", scenario_id)
    decision_maker_key = node_key("DecisionMaker", decision_maker_id)
    preference_key = node_key("Preference", decision_maker_id)
    add_node(
        db,
        event_key,
        "DecisionEvent",
        f"{prediction['decision_maker']['name']} model prediction for {scenario_id}",
        event_context,
        provenance,
    )
    add_relationship(db, situation_key, event_key, "HAS_DECISION_EVENT", provenance=provenance)
    add_relationship(db, event_key, situation_key, "OCCURRED_IN", provenance=provenance)
    add_relationship(db, decision_maker_key, event_key, "PARTICIPATED_IN", provenance=provenance)
    add_relationship(
        db,
        event_key,
        preference_key,
        "INFLUENCED_BY",
        {"factor": "decision-maker preference weights"},
        provenance,
    )

    for item in prediction["probabilities"]:
        alternative_id = item["alternative_id"]
        prediction_key = node_key("Prediction", f"{event_id}:{item['alternative_code']}")
        db.add(Prediction(
            decision_event_id=event_id,
            alternative_id=alternative_id,
            probability=item["probability"],
            model_name=model_name,
            model_version=model_version,
            explanation=prediction["disclaimer"],
            data_source="model",
        ))
        add_node(
            db,
            prediction_key,
            "Prediction",
            f"Predicted probability for {item['alternative_name']}",
            {
                "probability": item["probability"],
                "model_name": model_name,
                "model_version": model_version,
                "prediction_run_id": prediction_run_id,
            },
            provenance,
        )
        add_relationship(
            db,
            event_key,
            prediction_key,
            "HAS_PREDICTION",
            {"probability": item["probability"]},
            provenance,
        )
        alternative_key = node_key("Alternative", alternative_id)
        add_relationship(
            db,
            prediction_key,
            alternative_key,
            "EVALUATES",
            {"probability": item["probability"]},
            provenance,
        )
        if item["alternative_code"] == prediction["predicted_choice"]:
            add_relationship(
                db,
                event_key,
                prediction_key,
                "PREDICTED_AS",
                {"probability": item["probability"]},
                provenance,
            )
            add_relationship(
                db,
                prediction_key,
                alternative_key,
                "PREDICTED_AS",
                {"probability": item["probability"]},
                provenance,
            )

    model_explanation = prediction.get("explanation")
    if model_explanation:
        explanation_row = ModelExplanation(
            prediction_event_id=event_id,
            model_version=model_version,
            top_features=model_explanation["top_features"],
            tree_vote_summary=model_explanation["tree_vote_summary"],
            rule_paths=model_explanation["rule_paths"],
            recurring_rules=model_explanation["recurring_rules"],
            data_source="model",
        )
        db.add(explanation_row)
        db.flush()
        explanation_key = node_key("ModelExplanation", event_id)
        predicted_alternative = next(
            item for item in prediction["probabilities"]
            if item["alternative_code"] == prediction["predicted_choice"]
        )
        predicted_alternative_key = node_key("Alternative", predicted_alternative["alternative_id"])
        add_node(
            db,
            explanation_key,
            "ModelExplanation",
            f"Experimental Random Forest patterns for {scenario_id}",
            {
                "model_version": model_version,
                "top_features": model_explanation["top_features"],
                "tree_vote_summary": model_explanation["tree_vote_summary"],
                "rule_path_note": model_explanation["rule_path_note"],
            },
            provenance,
        )
        add_relationship(
            db,
            event_key,
            explanation_key,
            "HAS_EXPLANATION",
            provenance=provenance,
        )
        for path in model_explanation["rule_paths"]:
            for condition in path["conditions"]:
                db.add(ExtractedRule(
                    id=str(uuid4()),
                    model_explanation_id=explanation_row.id,
                    prediction_event_id=event_id,
                    model_version=model_version,
                    tree_id=path["tree_id"],
                    feature_name=condition["feature_name"],
                    condition_operator=condition["condition_operator"],
                    threshold=condition["threshold"],
                    branch_taken=condition["branch_taken"],
                    predicted_choice_id=predicted_alternative["alternative_id"],
                    scenario_id=scenario_id,
                    tree_vote=path["tree_vote"],
                    support_count=condition["support_count"],
                    support_confidence=condition["support_confidence"],
                    data_source="model",
                ))
        for index, rule in enumerate(model_explanation["recurring_rules"][:5]):
            rule_key = node_key("ExtractedRule", f"{event_id}:{index}")
            factor_key = node_key("DecisionFactor", f"{event_id}:model:{index}")
            add_node(
                db,
                rule_key,
                "ExtractedRule",
                f"{rule['feature_name']} {rule['condition_operator']} {rule['threshold']:g}",
                {
                    **rule,
                    "interpretation": "Experimental pattern from historical human choices.",
                    "model_version": model_version,
                },
                provenance,
            )
            add_node(
                db,
                factor_key,
                "DecisionFactor",
                rule["feature_name"],
                {
                    "feature_name": rule["feature_name"],
                    "source": "model_derived",
                    "model_version": model_version,
                },
                provenance,
            )
            add_relationship(db, explanation_key, rule_key, "HAS_EXTRACTED_RULE", provenance=provenance)
            add_relationship(db, rule_key, factor_key, "USES_FEATURE", provenance=provenance)
            add_relationship(
                db,
                rule_key,
                predicted_alternative_key,
                "SUPPORTS",
                {"support_count": rule["support_count"], "support_confidence": rule["support_confidence"]},
                provenance,
            )

    db.commit()
    return {
        **prediction,
        "prediction_event_id": event_id,
        "prediction_run_id": prediction_run_id,
        "persisted": True,
    }


def _add_human_reasoning_graph(
    db: Session,
    decision_key: str,
    decision_id: str,
    event: DecisionEvent,
    reasoning: dict,
    rationale: str,
) -> None:
    provenance = "observed_human:dashboard"
    if reasoning.get("operational_objective"):
        objective_key = node_key("Objective", decision_id)
        add_node(
            db,
            objective_key,
            "Objective",
            reasoning["operational_objective"],
            {"decision_id": decision_id, "source": "human_stated"},
            provenance,
        )
        add_relationship(db, decision_key, objective_key, "PURSUED", provenance=provenance)

    if rationale:
        rationale_key = node_key("HumanStatedReason", decision_id)
        add_node(
            db,
            rationale_key,
            "HumanStatedReason",
            f"Human rationale for {decision_id}",
            {
                "text": rationale,
                "source": reasoning.get("rationale_source", "human_stated"),
                "decision_id": decision_id,
            },
            provenance,
        )
        add_relationship(
            db,
            decision_key,
            rationale_key,
            "HUMAN_STATED_REASON",
            provenance=provenance,
        )

    for index, constraint in enumerate(reasoning["constraints"]):
        constraint_key = node_key("Constraint", f"{decision_id}:{index}")
        add_node(
            db,
            constraint_key,
            "Constraint",
            constraint,
            {"decision_id": decision_id, "source": "human_stated"},
            provenance,
        )
        add_relationship(db, decision_key, constraint_key, "CONSTRAINED_BY", provenance=provenance)

    for index, factor in enumerate(reasoning["influencing_factors"]):
        factor_key = node_key("DecisionFactor", f"{decision_id}:human:{index}")
        add_node(
            db,
            factor_key,
            "DecisionFactor",
            factor,
            {"decision_id": decision_id, "source": "human_stated"},
            provenance,
        )
        add_relationship(db, decision_key, factor_key, "INFLUENCED_BY", provenance=provenance)

    for index, step in enumerate(reasoning["reasoning_steps"]):
        step_key = node_key("ReasoningStep", f"{decision_id}:{index}")
        add_node(
            db,
            step_key,
            "ReasoningStep",
            step,
            {"decision_id": decision_id, "step_index": index + 1, "source": "human_stated"},
            provenance,
        )
        add_relationship(
            db,
            decision_key,
            step_key,
            "HAS_REASONING_STEP",
            {"step_index": index + 1},
            provenance,
        )

    snapshot_alternatives = {
        item["code"]: item
        for item in reasoning["available_alternatives"]
    }
    for code in reasoning["considered_alternative_codes"]:
        alternative = snapshot_alternatives[code]
        add_relationship(
            db,
            decision_key,
            node_key("Alternative", alternative["id"]),
            "CONSIDERED",
            {"alternative_code": code},
            provenance,
        )

    preference_key = node_key("Preference", event.decision_maker_id)
    if db.get(ContextNode, preference_key) is not None:
        add_relationship(
            db,
            decision_key,
            preference_key,
            "INFLUENCED_BY",
            {"weights_at_decision": reasoning["preference_weights"]},
            provenance,
        )


@app.post("/api/ml/decisions")
def record_ml_decision(request: MLDecisionRequest, db: Session = Depends(get_db)):
    event = db.get(DecisionEvent, request.prediction_event_id)
    if event is None or (event.event_context or {}).get("source") != "ml_prediction":
        raise HTTPException(status_code=404, detail="ML prediction event was not found.")
    if event.decision is not None:
        raise HTTPException(status_code=409, detail="A human decision is already recorded for this prediction.")

    chosen_prediction = next(
        (
            item for item in event.predictions
            if item.alternative.code == request.chosen_option_id
        ),
        None,
    )
    if chosen_prediction is None:
        raise HTTPException(
            status_code=422,
            detail="The selected alternative was not included in this prediction.",
        )
    objective = request.operational_objective.strip()
    if not objective:
        raise HTTPException(status_code=422, detail="An operational objective is required.")
    rationale = request.rationale.strip()
    if not rationale:
        raise HTTPException(status_code=422, detail="A human-stated rationale is required.")
    reasoning_snapshot = (event.event_context or {}).get("reasoning_snapshot", {})
    available_alternatives = reasoning_snapshot.get("alternatives", [])
    if not available_alternatives:
        available_alternatives = [
            {
                "id": item.alternative_id,
                "code": item.alternative.code,
                "name": item.alternative.name,
                "available": True,
                "eligible": True,
                "exclusion_reason": None,
                "attributes": item.alternative.attributes,
                "criterion_scores": {},
                "overall_suitability": None,
            }
            for item in event.predictions
        ]
    eligible_codes = {
        item["code"] for item in available_alternatives
        if item.get("eligible")
    }
    considered_codes = request.considered_alternative_codes or sorted(eligible_codes)
    if not set(considered_codes).issubset(eligible_codes):
        raise HTTPException(
            status_code=422,
            detail="Considered alternatives must be available and eligible for this decision.",
        )
    if request.chosen_option_id not in considered_codes:
        raise HTTPException(
            status_code=422,
            detail="The selected alternative must be included among those considered.",
        )

    def clean_items(values: List[str], label: str) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 250 for value in cleaned):
            raise HTTPException(
                status_code=422,
                detail=f"{label} entries must contain 1 to 250 non-whitespace characters.",
            )
        return cleaned

    human_reasoning = {
        "operational_objective": objective,
        "constraints": clean_items(request.constraints, "Constraint"),
        "influencing_factors": clean_items(request.influencing_factors, "Influencing factor"),
        "reasoning_steps": clean_items(request.reasoning_steps, "Reasoning step"),
        "considered_alternative_codes": considered_codes,
        "available_alternatives": available_alternatives,
        "situation_at_decision": reasoning_snapshot.get("situation", {}),
        "preference_weights": reasoning_snapshot.get("preference_weights", {}),
        "confidence": request.confidence,
        "rationale_source": "human_stated" if request.rationale else "not_provided",
        "rationale": rationale,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    public_id = f"DEC-{uuid4().hex[:12].upper()}"
    event.status = "decided"
    event.event_context = {
        **(event.event_context or {}),
        "public_decision_id": public_id,
        "recorded_choice_id": chosen_prediction.alternative.code,
        "human_reasoning": human_reasoning,
    }
    decision = Decision(
        decision_event=event,
        chosen_alternative_id=chosen_prediction.alternative_id,
        rationale=rationale,
        label_source="observed_human",
        confidence_note=public_id,
        data_source="observed_human",
    )
    db.add(decision)
    db.flush()
    if event.model_explanation:
        for rule in event.model_explanation.extracted_rules:
            rule.decision_id = decision.id

    decision_key = node_key("Decision", public_id)
    event_key = node_key("DecisionEvent", event.id)
    alternative_key = node_key("Alternative", chosen_prediction.alternative_id)
    add_node(
        db,
        decision_key,
        "Decision",
        f"{event.decision_maker.name} selected {chosen_prediction.alternative.name}",
        {
            "scenario_id": event.situation_id,
            "decision_maker_id": event.decision_maker_id,
            "recorded_choice_id": chosen_prediction.alternative.code,
            "rationale": rationale,
            "rationale_source": "human_stated",
            "human_reasoning": human_reasoning,
            "label_source": "observed_human",
            "prediction_run_id": (event.event_context or {}).get("prediction_run_id"),
        },
        "observed_human:dashboard",
    )
    add_relationship(
        db,
        event_key,
        decision_key,
        "HAS_DECISION",
        provenance="observed_human:dashboard",
    )
    add_relationship(
        db,
        decision_key,
        alternative_key,
        "SELECTED",
        {"rationale": request.rationale},
        "observed_human:dashboard",
    )
    add_relationship(
        db,
        event_key,
        alternative_key,
        "SELECTED",
        {"rationale": request.rationale},
        "observed_human:dashboard",
    )
    _add_human_reasoning_graph(
        db,
        decision_key,
        public_id,
        event,
        human_reasoning,
        rationale,
    )
    db.commit()
    return feedback_record(event, decision)


@app.post("/api/decisions", status_code=201)
def record_human_decision(
    request: HumanDecisionRequest,
    db: Session = Depends(get_db),
):
    scenario = db.get(Situation, request.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Unknown situation.")
    decision_maker = db.get(DecisionMaker, request.decision_maker_id)
    if decision_maker is None or not decision_maker.is_active:
        raise HTTPException(status_code=404, detail="Unknown active decision-maker.")
    preference = db.scalar(
        select(Preference)
        .where(
            Preference.decision_maker_id == decision_maker.id,
            Preference.is_current.is_(True),
        )
        .order_by(Preference.version.desc())
    )
    if preference is None:
        raise HTTPException(status_code=422, detail="Decision-maker has no current preference profile.")

    alternatives = db.scalars(
        select(Alternative)
        .where(Alternative.situation_id == scenario.id)
        .order_by(Alternative.code)
    ).all()
    assessments = db.scalars(
        select(Assessment)
        .where(Assessment.situation_id == scenario.id)
        .order_by(Assessment.assessment_version.desc(), Assessment.id.desc())
    ).all()
    assessment_by_alternative = {}
    for assessment in assessments:
        assessment_by_alternative.setdefault(assessment.alternative_id, assessment)
    required_units = scenario.context_data.get("required_units")
    available_alternatives = []
    eligible_codes = set()
    for alternative in alternatives:
        capacity = alternative.attributes.get("capacity_units")
        try:
            eligible = alternative.is_available and (
                required_units is None
                or capacity is not None and float(capacity) >= float(required_units)
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"Alternative {alternative.code} has invalid capacity data.",
            ) from error
        exclusion_reason = None
        if not alternative.is_available:
            exclusion_reason = "Alternative is unavailable."
        elif not eligible:
            exclusion_reason = "Alternative capacity is below the required quantity."
        if eligible:
            eligible_codes.add(alternative.code)
        assessment = assessment_by_alternative.get(alternative.id)
        available_alternatives.append({
            "id": alternative.id,
            "code": alternative.code,
            "name": alternative.name,
            "available": alternative.is_available,
            "eligible": eligible,
            "exclusion_reason": exclusion_reason,
            "attributes": alternative.attributes,
            "criterion_scores": assessment.criterion_scores if assessment else {},
            "overall_suitability": assessment.overall_score if assessment else None,
        })
    if request.chosen_option_id not in eligible_codes:
        raise HTTPException(
            status_code=422,
            detail="Chosen alternative is not currently eligible for this situation.",
        )
    considered_codes = list(dict.fromkeys(
        request.considered_alternative_codes or sorted(eligible_codes)
    ))
    if (
        not set(considered_codes).issubset(eligible_codes)
        or request.chosen_option_id not in considered_codes
    ):
        raise HTTPException(
            status_code=422,
            detail="Considered alternatives must be eligible and include the selected alternative.",
        )

    def clean_items(values: List[str], label: str) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 250 for value in cleaned):
            raise HTTPException(
                status_code=422,
                detail=f"{label} entries must contain 1 to 250 non-whitespace characters.",
            )
        return cleaned

    rationale = request.rationale.strip()
    objective = request.operational_objective.strip()
    if not rationale or not objective:
        raise HTTPException(
            status_code=422,
            detail="A non-empty human rationale and operational objective are required.",
        )

    event = None
    if request.prediction_event_id:
        event = db.get(DecisionEvent, request.prediction_event_id)
        if (
            event is None
            or event.situation_id != scenario.id
            or event.decision_maker_id != decision_maker.id
        ):
            raise HTTPException(
                status_code=404,
                detail="Prediction event does not match the situation and decision-maker.",
            )
        if event.decision is not None:
            raise HTTPException(status_code=409, detail="This event already has a recorded decision.")
    else:
        latest_event_number = db.scalar(
            select(func.max(DecisionEvent.event_number)).where(
                DecisionEvent.situation_id == scenario.id,
                DecisionEvent.decision_maker_id == decision_maker.id,
            )
        ) or 0
        event_id = f"{scenario.id}:{decision_maker.id}:human:{uuid4().hex}"
        event = DecisionEvent(
            id=event_id,
            situation_id=scenario.id,
            decision_maker_id=decision_maker.id,
            preference_id=preference.id,
            event_number=latest_event_number + 1,
            status="decided",
            event_context={
                "source": "human_decision",
                "scenario_id": scenario.id,
                "decision_maker_id": decision_maker.id,
                "event_number": latest_event_number + 1,
            },
            data_source="observed_human",
        )
        db.add(event)

    event_context = dict(event.event_context or {})
    reasoning_snapshot = event_context.get("reasoning_snapshot")
    if reasoning_snapshot is None:
        reasoning_snapshot = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "situation": {
                "id": scenario.id,
                "title": scenario.title,
                "description": scenario.description,
                "context_data": scenario.context_data,
            },
            "operational_objective": objective,
            "decision_maker": {
                "id": decision_maker.id,
                "name": decision_maker.name,
                "role": decision_maker.role,
            },
            "preference_weights": preference.weights,
            "alternatives": available_alternatives,
        }
    human_reasoning = {
        "operational_objective": objective,
        "constraints": clean_items(request.constraints, "Constraint"),
        "influencing_factors": clean_items(
            request.influencing_factors,
            "Influencing factor",
        ),
        "reasoning_steps": clean_items(request.reasoning_steps, "Reasoning step"),
        "considered_alternative_codes": considered_codes,
        "available_alternatives": reasoning_snapshot.get(
            "alternatives",
            available_alternatives,
        ),
        "situation_at_decision": reasoning_snapshot.get("situation", {}),
        "preference_weights": reasoning_snapshot.get(
            "preference_weights",
            preference.weights,
        ),
        "confidence": request.confidence,
        "rationale_source": "human_stated",
        "rationale": rationale,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    public_id = f"DEC-{uuid4().hex[:12].upper()}"
    event.status = "decided"
    event.event_context = {
        **event_context,
        "source": event_context.get("source", "human_decision"),
        "public_decision_id": public_id,
        "reasoning_snapshot": reasoning_snapshot,
        "human_reasoning": human_reasoning,
    }
    event.data_source = "observed_human"
    decision = Decision(
        decision_event_id=event.id,
        chosen_alternative_id=next(
            alternative.id
            for alternative in alternatives
            if alternative.code == request.chosen_option_id
        ),
        rationale=rationale,
        label_source="observed_human",
        confidence_note=public_id,
        data_source="observed_human",
    )
    db.add(decision)
    db.flush()
    if event.model_explanation:
        for rule in event.model_explanation.extracted_rules:
            rule.decision_id = decision.id

    event_key = node_key("DecisionEvent", event.id)
    situation_key = node_key("Situation", scenario.id)
    decision_key = node_key("Decision", public_id)
    alternative_key = node_key(
        "Alternative",
        decision.chosen_alternative_id,
    )
    add_node(
        db,
        event_key,
        "DecisionEvent",
        f"{decision_maker.name} decision for {scenario.id}",
        event.event_context,
        "observed_human:decision-recording",
    )
    add_relationship(
        db,
        situation_key,
        event_key,
        "HAS_DECISION_EVENT",
        provenance="observed_human:decision-recording",
    )
    add_relationship(
        db,
        node_key("DecisionMaker", decision_maker.id),
        event_key,
        "PARTICIPATED_IN",
        provenance="observed_human:decision-recording",
    )
    if db.get(ContextNode, node_key("Preference", decision_maker.id)):
        add_relationship(
            db,
            event_key,
            node_key("Preference", decision_maker.id),
            "INFLUENCED_BY",
            {
                "weights_at_decision": reasoning_snapshot.get(
                    "preference_weights",
                    preference.weights,
                ),
            },
            "observed_human:decision-recording",
        )
    add_node(
        db,
        decision_key,
        "Decision",
        f"{decision_maker.name} selected {decision.chosen_alternative.name}",
        {
            "scenario_id": scenario.id,
            "decision_maker_id": decision_maker.id,
            "recorded_choice_id": request.chosen_option_id,
            "rationale": rationale,
            "rationale_source": "human_stated",
            "human_reasoning": human_reasoning,
            "label_source": "observed_human",
            "source": event.data_source,
        },
        "observed_human:decision-recording",
    )
    add_relationship(
        db,
        event_key,
        decision_key,
        "HAS_DECISION",
        provenance="observed_human:decision-recording",
    )
    add_relationship(
        db,
        decision_key,
        alternative_key,
        "SELECTED",
        {"rationale": rationale},
        "observed_human:decision-recording",
    )
    _add_human_reasoning_graph(
        db,
        decision_key,
        public_id,
        event,
        human_reasoning,
        rationale,
    )
    db.commit()
    return feedback_record(event, decision)


@app.get("/api/physical")
def get_physical_network():
    return CONFIG["physical_network"]


@app.get("/api/context-graph")
def get_context_graph(
    scenario_id: str | None = None,
    depth: int = 3,
    db: Session = Depends(get_db),
):
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=422, detail="Graph depth must be between 1 and 5.")
    if scenario_id and scenario_id not in TRAINING_DATA["scenarios"]:
        raise HTTPException(status_code=404, detail=f"Unknown scenario ID: {scenario_id}")
    return graph_snapshot(db, scenario_id, depth)


@app.get("/api/context-graph/nodes/{node_id}/neighbors")
def get_context_neighbors(
    node_id: str,
    depth: int = 1,
    db: Session = Depends(get_db),
):
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=422, detail="Graph depth must be between 1 and 5.")
    result = neighbors(db, node_id, depth)
    if result["node"] is None:
        raise HTTPException(status_code=404, detail=f"Unknown context graph node: {node_id}")
    return result


@app.get("/api/context-graph/similar/{scenario_id}")
def get_similar_situations(
    scenario_id: str,
    limit: int = 5,
    db: Session = Depends(get_db),
):
    if scenario_id not in TRAINING_DATA["scenarios"]:
        raise HTTPException(status_code=404, detail=f"Unknown scenario ID: {scenario_id}")
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 20.")
    return similar_situations(db, scenario_id, limit)


@app.post("/api/context-graph/nodes", status_code=201)
def create_context_node(request: ContextNodeRequest, db: Session = Depends(get_db)):
    situation_key = node_key("Situation", request.situation_id)
    if db.get(ContextNode, situation_key) is None:
        raise HTTPException(status_code=404, detail=f"Unknown situation ID: {request.situation_id}")
    context_key = node_key("Context", request.node_id)
    if db.get(ContextNode, context_key) is not None:
        raise HTTPException(status_code=409, detail=f"Context node already exists: {request.node_id}")
    add_node(
        db,
        context_key,
        "Context",
        request.name,
        {**request.properties, "context_type": request.context_type, "situation_id": request.situation_id},
        request.provenance,
    )
    add_relationship(
        db,
        situation_key,
        context_key,
        "HAS_CONTEXT",
        {"context_type": request.context_type},
        request.provenance,
    )
    db.commit()
    return {
        "node": {
            "id": context_key,
            "type": "Context",
            "name": request.name,
            "properties": {**request.properties, "context_type": request.context_type, "situation_id": request.situation_id},
            "provenance": request.provenance,
        },
        "connected_to": situation_key,
        "relationship": "HAS_CONTEXT",
        "bidirectional_traversal": True,
    }


@app.post("/api/context-graph/relationships", status_code=201)
def create_context_relationship(
    request: ContextRelationshipRequest,
    db: Session = Depends(get_db),
):
    if request.relationship_type not in RELATIONSHIP_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported context graph relationship type.")
    if request.source_node_id == request.target_node_id:
        raise HTTPException(status_code=422, detail="A relationship must connect two different nodes.")
    if db.get(ContextNode, request.source_node_id) is None or db.get(ContextNode, request.target_node_id) is None:
        raise HTTPException(status_code=404, detail="Both context graph nodes must exist.")
    relationship = db.query(ContextRelationship).filter_by(
        source_node_key=request.source_node_id,
        target_node_key=request.target_node_id,
        relationship_type=request.relationship_type,
    ).first()
    if relationship:
        raise HTTPException(status_code=409, detail="This context graph relationship already exists.")
    add_relationship(
        db,
        request.source_node_id,
        request.target_node_id,
        request.relationship_type,
        request.properties,
        request.provenance,
    )
    db.commit()
    return {
        "source": request.source_node_id,
        "target": request.target_node_id,
        "type": request.relationship_type,
        "properties": request.properties,
        "provenance": request.provenance,
        "bidirectional_traversal": True,
    }


@app.get("/api/twin")
def get_digital_twin():
    return {
        **deepcopy(TWIN_STATE),
        "recent_events": deepcopy(TWIN_EVENTS[:20]),
    }


@app.post("/api/twin/updates")
def update_digital_twin(request: TwinUpdateRequest):
    event = apply_twin_update(request)
    return {"event": event, "twin": get_digital_twin()}


@app.get("/api/feedback")
def get_decision_feedback(db: Session = Depends(get_db)):
    return {"records": persisted_feedback_records(db)}


def create_persisted_decision(db: Session, request: DecisionFeedbackRequest) -> dict:
    assessment = calculate_assessment(AssessRequest(scenario_id=request.scenario_id))
    person = next(
        (item for item in assessment["stage3"]["decision_maker_probabilities"] if item["decision_maker_id"] == request.decision_maker_id),
        None,
    )
    preference_profile = next(
        (
            item
            for item in assessment["stage2"]
            if item["decision_maker_id"] == request.decision_maker_id
        ),
        {},
    )
    if person is None:
        raise HTTPException(status_code=422, detail=f"Unknown decision-maker: {request.decision_maker_id}")
    option = next(
        (item for item in person["probabilities"] if item["option_id"] == request.chosen_option_id),
        None,
    )
    if option is None:
        raise HTTPException(status_code=422, detail="Chosen option is not currently available in the assessment.")

    scenario_id = assessment["scenario_id"]
    public_id = f"DEC-{uuid4().hex[:12].upper()}"
    event_id = f"{scenario_id}:{request.decision_maker_id}:actual:{uuid4().hex}"
    rationale_text = request.rationale.strip()
    eligible_codes = {item["option_id"] for item in person["probabilities"]}
    considered_codes = request.considered_alternative_codes or sorted(eligible_codes)
    if not set(considered_codes).issubset(eligible_codes) or request.chosen_option_id not in considered_codes:
        raise HTTPException(
            status_code=422,
            detail="Considered alternatives must be eligible and include the selected alternative.",
        )

    def clean_items(values: List[str], label: str) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 250 for value in cleaned):
            raise HTTPException(
                status_code=422,
                detail=f"{label} entries must contain 1 to 250 non-whitespace characters.",
            )
        return cleaned

    scenario = db.get(Situation, scenario_id)
    scenario_alternatives = db.scalars(
        select(Alternative)
        .where(Alternative.situation_id == scenario_id)
        .order_by(Alternative.code)
    ).all()
    scenario_assessments = db.scalars(
        select(Assessment)
        .where(Assessment.situation_id == scenario_id)
        .order_by(Assessment.assessment_version.desc(), Assessment.id.desc())
    ).all()
    assessment_by_alternative = {}
    for item in scenario_assessments:
        assessment_by_alternative.setdefault(item.alternative_id, item)
    available_alternatives = [
        {
            "id": item.id,
            "code": item.code,
            "name": item.name,
            "available": item.is_available,
            "eligible": item.code in eligible_codes,
            "exclusion_reason": None if item.code in eligible_codes else "Excluded by current assessment.",
            "attributes": item.attributes,
            "criterion_scores": (
                assessment_by_alternative[item.id].criterion_scores
                if item.id in assessment_by_alternative
                else {}
            ),
            "overall_suitability": (
                assessment_by_alternative[item.id].overall_score
                if item.id in assessment_by_alternative
                else None
            ),
        }
        for item in scenario_alternatives
    ]
    preference = db.scalar(
        select(Preference).where(
            Preference.decision_maker_id == request.decision_maker_id,
            Preference.is_current.is_(True),
        )
    )
    reasoning_snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "situation": {
            "id": scenario.id if scenario else scenario_id,
            "title": scenario.title if scenario else "",
            "description": scenario.description if scenario else "",
            "context_data": scenario.context_data if scenario else {},
        },
        "decision_maker": {
            "id": request.decision_maker_id,
            "name": person["decision_maker_name"],
            "role": person.get("role", preference_profile.get("role", "")),
        },
        "preference_weights": preference.weights if preference else {},
        "alternatives": available_alternatives,
    }
    human_reasoning = {
        "operational_objective": request.operational_objective.strip() if request.operational_objective else None,
        "constraints": clean_items(request.constraints, "Constraint"),
        "influencing_factors": clean_items(request.influencing_factors, "Influencing factor"),
        "reasoning_steps": clean_items(request.reasoning_steps, "Reasoning step"),
        "considered_alternative_codes": considered_codes,
        "available_alternatives": available_alternatives,
        "situation_at_decision": reasoning_snapshot["situation"],
        "preference_weights": reasoning_snapshot["preference_weights"],
        "confidence": request.confidence,
        "rationale_source": "human_stated" if rationale_text else "not_provided",
        "rationale": rationale_text,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_event_number = db.scalar(
        select(func.max(DecisionEvent.event_number)).where(
            DecisionEvent.situation_id == scenario_id,
            DecisionEvent.decision_maker_id == request.decision_maker_id,
        )
    ) or 0
    event = DecisionEvent(
        id=event_id,
        situation_id=scenario_id,
        decision_maker_id=request.decision_maker_id,
        preference_id=preference.id if preference else None,
        event_number=latest_event_number + 1,
        status="decided",
        event_context={
            "source": "interactive",
            "public_decision_id": public_id,
            "scenario_id": scenario_id,
            "event_number": latest_event_number + 1,
            "twin_revision": assessment["operational_snapshot"]["twin_revision"],
            "reasoning_snapshot": reasoning_snapshot,
            "human_reasoning": human_reasoning,
        },
        data_source="observed_human",
    )
    db.add(event)
    db.flush()

    event_key = node_key("DecisionEvent", event_id)
    scenario_key = node_key("Situation", scenario_id)
    preference_key = node_key("Preference", request.decision_maker_id)
    add_node(
        db,
        event_key,
        "DecisionEvent",
        f"{person['decision_maker_name']} decision for {scenario_id}",
        event.event_context,
        "observed_human:dashboard",
    )
    add_relationship(db, scenario_key, event_key, "HAS_DECISION_EVENT", provenance="observed_human:dashboard")
    if db.get(ContextNode, preference_key):
        add_relationship(db, event_key, preference_key, "INFLUENCED_BY", {"factor": "decision-maker preference"}, "observed_human:dashboard")

    for prediction in person["probabilities"]:
        alternative_id = f"{scenario_id}:{prediction['option_id']}"
        db.add(Prediction(
            decision_event_id=event_id,
            alternative_id=alternative_id,
            utility_score=prediction["utility_score"],
            probability=prediction["probability_percent"] / 100,
            model_name="illustrative_softmax",
            model_version=f"interactive-{public_id}",
            explanation="Illustrative model estimate at decision time.",
        ))
        prediction_key = node_key("Prediction", f"{event_id}:{prediction['option_id']}")
        alternative_key = node_key("Alternative", alternative_id)
        add_node(
            db,
            prediction_key,
            "Prediction",
            f"Predicted probability for {prediction['option_name']}",
            {
                "probability": prediction["probability_percent"] / 100,
                "utility_score": prediction["utility_score"],
                "model_version": f"interactive-{public_id}",
            },
            "model:illustrative_softmax",
        )
        add_relationship(
            db,
            event_key,
            prediction_key,
            "HAS_PREDICTION",
            {"probability": prediction["probability_percent"] / 100},
            "model:illustrative_softmax",
        )
        add_relationship(
            db,
            prediction_key,
            alternative_key,
            "EVALUATES",
            {"probability": prediction["probability_percent"] / 100},
            "model:illustrative_softmax",
        )
        if prediction["option_name"] == person["most_likely_choice"]:
            add_relationship(
                db,
                event_key,
                prediction_key,
                "PREDICTED_AS",
                {"probability": prediction["probability_percent"] / 100},
                "model:illustrative_softmax",
            )
            add_relationship(
                db,
                prediction_key,
                alternative_key,
                "PREDICTED_AS",
                {"probability": prediction["probability_percent"] / 100},
                "model:illustrative_softmax",
            )

    alternative_id = f"{scenario_id}:{option['option_id']}"
    if db.get(Alternative, alternative_id) is None:
        raise HTTPException(status_code=422, detail="Chosen option is not stored for this situation.")
    decision = Decision(
        decision_event_id=event_id,
        chosen_alternative_id=alternative_id,
        rationale=rationale_text,
        label_source="observed_human",
        confidence_note=public_id,
        data_source="observed_human",
    )
    db.add(decision)
    db.flush()
    decision_key = node_key("Decision", public_id)
    add_node(
        db,
        decision_key,
        "Decision",
        f"{person['decision_maker_name']} selected {option['option_name']}",
        {
            "scenario_id": scenario_id,
            "decision_maker_id": request.decision_maker_id,
            "recorded_choice_id": option["option_id"],
            "rationale": rationale_text,
            "rationale_source": human_reasoning["rationale_source"],
            "human_reasoning": human_reasoning,
            "label_source": "observed_human",
        },
        "observed_human:dashboard",
    )
    add_relationship(db, event_key, decision_key, "HAS_DECISION", provenance="observed_human:dashboard")
    add_relationship(
        db,
        decision_key,
        node_key("Alternative", alternative_id),
        "SELECTED",
        {"rationale": request.rationale},
        "observed_human:dashboard",
    )
    _add_human_reasoning_graph(
        db,
        decision_key,
        public_id,
        event,
        human_reasoning,
        rationale_text,
    )
    db.commit()
    return feedback_record(event, decision)

def feedback_record(event: DecisionEvent, decision: Decision) -> dict:
    event_context = event.event_context or {}
    if event_context.get("source") == "ml_prediction":
        predictions = list(event.predictions)
    else:
        predictions = [
            prediction for prediction in event.predictions
            if prediction.model_version.startswith("interactive-")
        ]
    predicted = max(predictions, key=lambda prediction: prediction.probability) if predictions else None
    predicted_name = predicted.alternative.name if predicted else None
    chosen = decision.chosen_alternative
    chosen_prediction = next(
        (prediction for prediction in predictions if prediction.alternative_id == chosen.id),
        None,
    )
    outcome = decision.outcome
    return {
        "id": decision.confidence_note or f"DEC-{decision.id}",
        "timestamp": decision.decided_at.isoformat(),
        "scenario_id": event.situation_id,
        "scenario_title": event.situation.title,
        "decision_maker_id": event.decision_maker_id,
        "decision_maker_name": event.decision_maker.name,
        "chosen_option_id": chosen.code,
        "chosen_option_name": chosen.name,
        "prediction_event_id": event.id,
        "prediction_run_id": event_context.get("prediction_run_id"),
        "predicted_choice": predicted_name,
        "predicted_choice_probability_percent": round(predicted.probability * 100, 1) if predicted else 0,
        "chosen_option_probability_percent": round(chosen_prediction.probability * 100, 1) if chosen_prediction else 0,
        "choice_matches_prediction": chosen.id == predicted.alternative_id if predicted else False,
        "probabilities": [
            {
                "option_id": prediction.alternative.code,
                "option_name": prediction.alternative.name,
                "probability_percent": round(prediction.probability * 100, 1),
            }
            for prediction in sorted(
                predictions,
                key=lambda item: item.probability,
                reverse=True,
            )
        ],
        "preference_weights": event.preference.weights if event.preference else {},
        "model_name": predicted.model_name if predicted else None,
        "rationale": decision.rationale or "",
        "reasoning_context": event_context.get("human_reasoning", {}),
        "reasoning_snapshot": event_context.get("reasoning_snapshot", {}),
        "model_explanation": (
            {
                "model_version": event.model_explanation.model_version,
                "training_data_source": event_context.get("training_data_source"),
                "top_features": event.model_explanation.top_features,
                "tree_vote_summary": event.model_explanation.tree_vote_summary,
                "rule_paths": event.model_explanation.rule_paths,
                "recurring_rules": event.model_explanation.recurring_rules,
                "generated_at": event.model_explanation.generated_at.isoformat(),
            }
            if event.model_explanation
            else None
        ),
        "twin_revision": event_context.get("twin_revision", 0),
        "outcome": {
            "status": outcome.status,
            "notes": outcome.summary or "",
            "recorded_at": outcome.observed_at.isoformat() if outcome.observed_at else outcome.updated_at.isoformat(),
            "twin_updates": (outcome.metrics or {}).get("twin_updates", []),
        } if outcome else None,
    }


def persisted_feedback_records(db: Session) -> list[dict]:
    events = db.scalars(select(DecisionEvent).order_by(DecisionEvent.created_at.desc())).all()
    records = [
        feedback_record(event, event.decision)
        for event in events
        if (event.event_context or {}).get("source") in {
            "interactive",
            "ml_prediction",
            "human_decision",
        }
        and event.decision is not None
    ]
    return records


@app.get("/api/decisions/{decision_id}")
def get_decision_trace(decision_id: str, db: Session = Depends(get_db)):
    decision = db.scalar(
        select(Decision).where(Decision.confidence_note == decision_id)
    )
    if decision is None and decision_id.startswith("DEC-"):
        numeric_id = decision_id[4:]
        if numeric_id.isdigit():
            decision = db.get(Decision, int(numeric_id))
    if decision is None:
        raise HTTPException(status_code=404, detail=f"Unknown decision ID: {decision_id}")

    event = decision.decision_event
    event_context = event.event_context or {}
    human_reasoning = event_context.get("human_reasoning", {})
    snapshot = event_context.get("reasoning_snapshot", {})
    explanation = event.model_explanation
    extracted_rules = (
        [
            {
                "id": rule.id,
                "tree_id": rule.tree_id,
                "feature_name": rule.feature_name,
                "condition_operator": rule.condition_operator,
                "threshold": rule.threshold,
                "branch_taken": rule.branch_taken,
                "tree_vote": rule.tree_vote,
                "model_version": rule.model_version,
                "support_count": rule.support_count,
                "support_confidence": rule.support_confidence,
                "decision_id": rule.decision_id,
                "source": rule.data_source,
            }
            for rule in explanation.extracted_rules
        ]
        if explanation
        else []
    )
    outcome = decision.outcome
    graph_node_id = node_key(
        "Decision",
        decision.confidence_note or f"DEC-{decision.id}",
    )
    graph = neighbors(db, graph_node_id, depth=4)
    applications = db.scalars(
        select(RuleApplication)
        .where(RuleApplication.decision_id == decision.id)
        .order_by(RuleApplication.created_at)
    ).all()
    return {
        "decision_id": decision.confidence_note or f"DEC-{decision.id}",
        "decision_database_id": decision.id,
        "decision_event_id": event.id,
        "recorded_at": decision.decided_at.isoformat(),
        "source": {
            "decision": decision.data_source,
            "event": event.data_source,
            "prediction_models": sorted({
                prediction.model_name for prediction in event.predictions
            }),
            "model_versions": sorted({
                prediction.model_version for prediction in event.predictions
            }),
            "training_data_sources": sorted({
                event_context.get("training_data_source")
            } - {None}),
        },
        "operational_objective": human_reasoning.get(
            "operational_objective",
            snapshot.get("operational_objective"),
        ),
        "situation_snapshot": snapshot.get("situation"),
        "alternative_snapshot": snapshot.get("alternatives", []),
        "constraints": human_reasoning.get("constraints", []),
        "considered_alternatives": human_reasoning.get(
            "considered_alternative_codes",
            [],
        ),
        "preference_snapshot": human_reasoning.get(
            "preference_weights",
            snapshot.get("preference_weights", {}),
        ),
        "reasoning_steps": human_reasoning.get("reasoning_steps", []),
        "influencing_factors": human_reasoning.get("influencing_factors", []),
        "confidence": human_reasoning.get("confidence"),
        "recorded_choice": {
            "alternative_id": decision.chosen_alternative_id,
            "alternative_code": decision.chosen_alternative.code,
            "alternative_name": decision.chosen_alternative.name,
            "label_source": decision.label_source,
        },
        "human_rationale": {
            "text": decision.rationale or "",
            "source": human_reasoning.get("rationale_source", "human_stated"),
        },
        "predictions": [
            {
                "alternative_id": prediction.alternative_id,
                "alternative_code": prediction.alternative.code,
                "alternative_name": prediction.alternative.name,
                "probability": prediction.probability,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
            }
            for prediction in sorted(
                event.predictions,
                key=lambda item: item.probability,
                reverse=True,
            )
        ],
        "model_explanation": (
            {
                "model_version": explanation.model_version,
                "training_data_source": event_context.get("training_data_source"),
                "top_features": explanation.top_features,
                "tree_vote_summary": explanation.tree_vote_summary,
                "rule_paths": explanation.rule_paths,
                "recurring_rules": explanation.recurring_rules,
                "generated_at": explanation.generated_at.isoformat(),
                "source": explanation.data_source,
                "extracted_rules": extracted_rules,
            }
            if explanation
            else None
        ),
        "outcome": (
            {
                "status": outcome.status,
                "summary": outcome.summary,
                "metrics": outcome.metrics,
                "actual_observed": outcome.actual_observed,
                "observed_at": outcome.observed_at.isoformat() if outcome.observed_at else None,
                "source": outcome.data_source,
            }
            if outcome
            else None
        ),
        "rule_applications": [
            {
                "id": application.id,
                "rule_id": application.rule_id,
                "rule_version": application.rule_version,
                "relevance_status": application.relevance_status,
                "considered_by_human": application.considered_by_human,
                "feedback": application.feedback,
            }
            for application in applications
        ],
        "context_graph": {
            "node": graph["node"],
            "nodes": graph["nodes"],
            "relationships": graph["relationships"],
            "traversal": "bidirectional",
        },
    }


@app.post("/api/feedback/decisions")
def record_decision(request: DecisionFeedbackRequest, db: Session = Depends(get_db)):
    return create_persisted_decision(db, request)

@app.post("/api/feedback/decisions/{decision_id}/outcome")
def record_decision_outcome(
    decision_id: str,
    request: OutcomeFeedbackRequest,
    db: Session = Depends(get_db),
):
    return persist_decision_outcome(db, decision_id, request)


def persist_decision_outcome(
    db: Session,
    decision_id: str,
    request: OutcomeFeedbackRequest,
) -> dict:
    events = db.scalars(select(DecisionEvent)).all()
    event = next(
        (
            item for item in events
            if (item.event_context or {}).get("public_decision_id") == decision_id
            and item.decision is not None
        ),
        None,
    )
    if event is None:
        raise HTTPException(status_code=404, detail=f"Unknown decision ID: {decision_id}")

    applied_events = [apply_twin_update(update) for update in request.twin_updates]
    decision = event.decision
    outcome = decision.outcome
    if outcome is None:
        outcome = Outcome(decision_id=decision.id, data_source="observed_human")
        db.add(outcome)
    outcome.status = request.status
    outcome.summary = request.notes
    outcome.metrics = {"twin_updates": applied_events}
    outcome.actual_observed = True
    outcome.observed_at = datetime.now(timezone.utc)
    outcome.data_source = "observed_human"
    outcome_key = node_key("Outcome", decision_id)
    add_node(
        db,
        outcome_key,
        "Outcome",
        f"{request.status.title()} outcome for {decision_id}",
        {
            "status": request.status,
            "summary": request.notes,
            "actual_observed": True,
            "twin_updates": applied_events,
        },
        "observed_human:dashboard",
    )
    add_relationship(
        db,
        node_key("Decision", decision_id),
        outcome_key,
        "RESULTED_IN",
        {"status": request.status},
        "observed_human:dashboard",
    )
    for index, twin_update in enumerate(applied_events):
        context_key = node_key("Context", f"{decision_id}:outcome-update:{index}")
        add_node(
            db,
            context_key,
            "Context",
            f"Observed operational update: {twin_update['entity_id']} {twin_update['field']}",
            {"context_type": "event", **twin_update},
            "observed_human:outcome",
        )
        add_relationship(
            db,
            outcome_key,
            context_key,
            "INFLUENCED_BY",
            {"factor": "observed operational update"},
            "observed_human:outcome",
        )
    db.commit()
    return feedback_record(event, decision)

@app.get("/api/db-test")
def test_database():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return {
                "database": "Connected",
                "test_result": result.scalar()
            }
    except Exception as e:
        return {
            "database": "Connection failed",
            "error": str(e)
        }
