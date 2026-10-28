import csv
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
DECISION_FEEDBACK: list[dict] = []


def read_csv_rows(filename: str) -> list[dict]:
    with (TRAINING_DATA_PATH / filename).open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_training_data() -> dict:
    scenarios = {}
    for row in read_csv_rows("scenarios.csv"):
        scenario_id = row["scenario_id"]
        if scenario_id in scenarios:
            raise ValueError(f"Duplicate training scenario ID: {scenario_id}")
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
            "context": {
                "Product": row["product"],
                "Required quantity": f"{int(row['required_units']):,} units",
                "Required within": f"{row['required_within_days']} days",
                "Stockout risk": row["stockout_risk"],
                "Location": row["location"],
            },
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
        decision["outcome"] = outcome_by_decision[decision["decision_id"]]

    decision_pairs = {(row["scenario_id"], row["decision_maker_id"]) for row in decisions}
    expected_pairs = {
        (scenario_id, profile["id"])
        for scenario_id in scenarios
        for profile in profiles
    }
    if (
        len(scenarios) != 8
        or len(profiles) != 4
        or len(decisions) != 32
        or decision_pairs != expected_pairs
        or len(decision_pairs) != len(decisions)
    ):
        raise ValueError("Training data must contain 8 scenarios, 4 decision-maker profiles, and 32 decisions.")
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


class PreferenceProfile(BaseModel):
    decision_maker_id: str
    weights: Dict[str, float] = Field(description="Preference weight for each configured criterion.")


class AssessRequest(BaseModel):
    scenario_id: str | None = None
    operational_weights: Dict[str, float] | None = None
    preference_profiles: List[PreferenceProfile] | None = None
    temperature: float | None = Field(default=None, gt=0, le=100)


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


class OutcomeFeedbackRequest(BaseModel):
    status: Literal["fulfilled", "delayed", "failed", "partial"]
    notes: str = Field(default="", max_length=1000)
    twin_updates: List[TwinUpdateRequest] = Field(default_factory=list)


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


@app.get("/api/physical")
def get_physical_network():
    return CONFIG["physical_network"]


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
def get_decision_feedback():
    return {"records": deepcopy(DECISION_FEEDBACK)}


@app.post("/api/feedback/decisions")
def record_decision(request: DecisionFeedbackRequest):
    assessment = calculate_assessment(AssessRequest(scenario_id=request.scenario_id))
    person = next(
        (item for item in assessment["stage3"]["decision_maker_probabilities"] if item["decision_maker_id"] == request.decision_maker_id),
        None,
    )
    if person is None:
        raise HTTPException(status_code=422, detail=f"Unknown decision-maker: {request.decision_maker_id}")
    option = next(
        (item for item in person["probabilities"] if item["option_id"] == request.chosen_option_id),
        None,
    )
    if option is None:
        raise HTTPException(status_code=422, detail="Chosen option is not currently available in the digital twin.")
    record = {
        "id": f"DEC-{len(DECISION_FEEDBACK) + 1:04d}",
        "timestamp": current_timestamp(),
        "scenario_id": assessment["scenario_id"],
        "scenario_title": assessment["scenario"]["title"],
        "decision_maker_id": request.decision_maker_id,
        "decision_maker_name": person["decision_maker_name"],
        "chosen_option_id": option["option_id"],
        "chosen_option_name": option["option_name"],
        "predicted_choice": person["most_likely_choice"],
        "predicted_choice_probability_percent": person["most_likely_choice_probability_percent"],
        "chosen_option_probability_percent": option["probability_percent"],
        "choice_matches_prediction": option["option_name"] == person["most_likely_choice"],
        "rationale": request.rationale,
        "twin_revision": assessment["operational_snapshot"]["twin_revision"],
        "outcome": None,
    }
    DECISION_FEEDBACK.insert(0, record)
    return record


@app.post("/api/feedback/decisions/{decision_id}/outcome")
def record_decision_outcome(decision_id: str, request: OutcomeFeedbackRequest):
    record = next((item for item in DECISION_FEEDBACK if item["id"] == decision_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown decision ID: {decision_id}")

    applied_events = [apply_twin_update(update) for update in request.twin_updates]
    record["outcome"] = {
        "status": request.status,
        "notes": request.notes,
        "recorded_at": current_timestamp(),
        "twin_updates": applied_events,
    }
    return record
