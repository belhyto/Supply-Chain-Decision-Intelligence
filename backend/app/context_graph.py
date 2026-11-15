from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import networkx as nx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import (
    Alternative,
    Assessment,
    ContextNode,
    ContextRelationship,
    Decision,
    DecisionEvent,
    DecisionMaker,
    Outcome,
    Preference,
    Prediction,
    Situation,
)

PROVENANCE = "synthetic:backend/data/training"
RELATIONSHIP_TYPES = {
    "HAS_PARAMETER",
    "HAS_ALTERNATIVE",
    "EVALUATED_AGAINST",
    "HAS_PREFERENCE",
    "INFLUENCED_BY",
    "PREDICTED_AS",
    "HAS_PREDICTION",
    "SELECTED",
    "RESULTED_IN",
    "SIMILAR_TO",
    "HAS_CONTEXT",
    "HAS_ASSESSMENT",
    "HAS_DECISION_EVENT",
    "HAS_DECISION",
    "HAS_OUTCOME",
    "EVALUATES",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def node_key(node_type: str, entity_id: str | int) -> str:
    return f"{node_type}:{entity_id}"


def relation_id(source: str, target: str, relationship_type: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"context-graph:{source}:{relationship_type}:{target}"))


def add_node(
    session: Session,
    key: str,
    node_type: str,
    name: str,
    properties: dict,
    provenance: str = PROVENANCE,
) -> None:
    existing = session.get(ContextNode, key)
    if existing is None:
        session.add(ContextNode(
            node_key=key,
            node_type=node_type,
            name=name,
            properties=properties,
            provenance=provenance,
            is_synthetic=provenance.startswith("synthetic:"),
        ))
    else:
        existing.node_type = node_type
        existing.name = name
        existing.properties = properties
        existing.provenance = provenance
        existing.is_synthetic = provenance.startswith("synthetic:")


def add_relationship(
    session: Session,
    source: str,
    target: str,
    relationship_type: str,
    properties: dict | None = None,
    provenance: str = PROVENANCE,
    occurred_at: datetime | None = None,
) -> None:
    edge_id = relation_id(source, target, relationship_type)
    existing = session.get(ContextRelationship, edge_id)
    if existing is None:
        session.add(ContextRelationship(
            id=edge_id,
            source_node_key=source,
            target_node_key=target,
            relationship_type=relationship_type,
            properties=properties or {},
            provenance=provenance,
            occurred_at=occurred_at or utc_now(),
        ))
    else:
        existing.properties = properties or {}
        existing.provenance = provenance
        if occurred_at:
            existing.occurred_at = occurred_at


def _softmax(scores: dict[str, float], temperature: float = 8.0) -> dict[str, float]:
    maximum = max(score / temperature for score in scores.values())
    exponentials = {key: math.exp(score / temperature - maximum) for key, score in scores.items()}
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


def _node_payload(node: ContextNode) -> dict:
    return {
        "id": node.node_key,
        "type": node.node_type,
        "name": node.name,
        "properties": node.properties or {},
        "provenance": node.provenance,
        "is_synthetic": node.is_synthetic,
        "created_at": node.created_at.isoformat() if node.created_at else None,
        "updated_at": node.updated_at.isoformat() if node.updated_at else None,
    }


def _relationship_payload(relationship: ContextRelationship) -> dict:
    return {
        "id": relationship.id,
        "source": relationship.source_node_key,
        "target": relationship.target_node_key,
        "type": relationship.relationship_type,
        "properties": relationship.properties or {},
        "provenance": relationship.provenance,
        "timestamp": relationship.occurred_at.isoformat() if relationship.occurred_at else None,
        "bidirectional": True,
    }


def seed_context_graph(session: Session, training_data: dict, criteria: list[dict]) -> dict:
    session.execute(
        delete(ContextRelationship).where(ContextRelationship.provenance == PROVENANCE)
    )
    session.flush()
    scenarios = training_data["scenarios"]
    profiles = training_data["profiles"]
    decisions = training_data["decisions"]
    criterion_keys = [criterion["key"] for criterion in criteria]
    profile_by_id = {profile["id"]: profile for profile in profiles}
    decision_by_pair = {
        (decision["scenario_id"], decision["decision_maker_id"]): decision
        for decision in decisions
    }

    for profile in profiles:
        db_profile = session.get(DecisionMaker, profile["id"])
        if db_profile is None:
            db_profile = DecisionMaker(id=profile["id"], name=profile["name"])
            session.add(db_profile)
        db_profile.name = profile["name"]
        db_profile.role = profile["role"]
        db_profile.description = profile["description"]
        db_profile.is_active = True
        add_node(
            session,
            node_key("DecisionMaker", profile["id"]),
            "DecisionMaker",
            profile["name"],
            {"role": profile["role"], "description": profile["description"]},
        )
    session.flush()

    for criterion in criteria:
        add_node(
            session,
            node_key("Criterion", criterion["key"]),
            "Criterion",
            criterion["label"],
            {"key": criterion["key"], "scale": "0-100; higher is more suitable"},
        )

    profile_preference_ids = {}
    for profile in profiles:
        preference = session.scalar(
            select(Preference).where(
                Preference.decision_maker_id == profile["id"],
                Preference.version == 1,
            )
        )
        if preference is None:
            preference = Preference(
                decision_maker_id=profile["id"],
                version=1,
                weights=profile["preferences"],
                notes="Synthetic training profile",
                is_current=True,
            )
            session.add(preference)
            session.flush()
        else:
            preference.weights = profile["preferences"]
            preference.notes = "Synthetic training profile"
            preference.is_current = True
        profile_preference_ids[profile["id"]] = preference.id
        add_node(
            session,
            node_key("Preference", profile["id"]),
            "Preference",
            f"{profile['name']} preference profile",
            {"decision_maker_id": profile["id"], "weights": profile["preferences"], "version": preference.version},
        )
        add_relationship(
            session,
            node_key("DecisionMaker", profile["id"]),
            node_key("Preference", profile["id"]),
            "HAS_PREFERENCE",
            {"version": preference.version},
        )
        for criterion_key, weight in profile["preferences"].items():
            add_relationship(
                session,
                node_key("Preference", profile["id"]),
                node_key("Criterion", criterion_key),
                "EVALUATED_AGAINST",
                {"weight": weight},
            )
    session.flush()

    for scenario_id, scenario in scenarios.items():
        situation = session.get(Situation, scenario_id)
        if situation is None:
            situation = Situation(id=scenario_id, title=scenario["title"])
            session.add(situation)
        situation.title = scenario["title"]
        situation.scenario_type = scenario["title"]
        situation.description = scenario["situation"]
        situation.context_data = {
            "decision_conflict": scenario["decision_conflict"],
            "required_units": scenario["required_units"],
            "required_within_days": scenario["required_within_days"],
            "stockout_risk": scenario["stockout_risk"],
            "location": scenario["location"],
            "product": scenario["product"],
            "operational_weights": scenario["operational_weights"],
        }
        situation.is_synthetic = True
        situation.reported_by = "synthetic training fixture"
        situation_key = node_key("Situation", scenario_id)
        add_node(
            session,
            situation_key,
            "Situation",
            scenario["title"],
            {**situation.context_data, "description": scenario["situation"], "scenario_id": scenario_id},
        )

        for parameter_name, parameter_value in {
            "product": scenario["product"],
            "required_units": scenario["required_units"],
            "required_within_days": scenario["required_within_days"],
            "stockout_risk": scenario["stockout_risk"],
            "location": scenario["location"],
        }.items():
            parameter_id = f"{scenario_id}:{parameter_name}"
            parameter_key = node_key("Parameter", parameter_id)
            add_node(
                session,
                parameter_key,
                "Parameter",
                f"{parameter_name.replace('_', ' ').title()}: {parameter_value}",
                {"key": parameter_name, "value": parameter_value, "scenario_id": scenario_id},
            )
            add_relationship(session, situation_key, parameter_key, "HAS_PARAMETER", {"key": parameter_name})

        context_id = f"{scenario_id}:decision-conflict"
        context_key = node_key("Context", context_id)
        add_node(
            session,
            context_key,
            "Context",
            scenario["decision_conflict"],
            {
                "context_type": "external_condition",
                "description": scenario["decision_conflict"],
                "scenario_id": scenario_id,
            },
        )
        add_relationship(
            session,
            situation_key,
            context_key,
            "HAS_CONTEXT",
            {"context_type": "external_condition"},
        )

        stage1_rows = []
        for option_data in scenario["options"]:
            alternative_id = f"{scenario_id}:{option_data['id']}"
            alternative = session.get(Alternative, alternative_id)
            if alternative is None:
                alternative = Alternative(
                    id=alternative_id,
                    situation_id=scenario_id,
                    code=option_data["id"],
                    name=option_data["name"],
                )
                session.add(alternative)
            alternative.situation_id = scenario_id
            alternative.code = option_data["id"]
            alternative.name = option_data["name"]
            alternative.description = option_data["notes"]
            alternative.attributes = {
                **option_data["attributes"],
                "metrics": option_data["metrics"],
            }
            alternative.is_available = option_data["available"]
            alternative_key = node_key("Alternative", alternative_id)
            add_node(
                session,
                alternative_key,
                "Alternative",
                option_data["name"],
                {
                    "scenario_id": scenario_id,
                    "code": option_data["id"],
                    "metrics": option_data["metrics"],
                    "attributes": option_data["attributes"],
                    "available": option_data["available"],
                    "notes": option_data["notes"],
                },
            )
            add_relationship(
                session,
                situation_key,
                alternative_key,
                "HAS_ALTERNATIVE",
                {"available": option_data["available"]},
            )
            for criterion_key in criterion_keys:
                add_relationship(
                    session,
                    alternative_key,
                    node_key("Criterion", criterion_key),
                    "EVALUATED_AGAINST",
                    {
                        "score": option_data["metrics"][criterion_key],
                        "operational_weight": scenario["operational_weights"][criterion_key],
                    },
                )

            if option_data["available"] and option_data["attributes"]["capacity_units"] >= scenario["required_units"]:
                score = sum(
                    option_data["metrics"][key] * scenario["operational_weights"][key]
                    for key in criterion_keys
                )
                stage1_rows.append((option_data, score))
        stage1_rows.sort(key=lambda row: row[1], reverse=True)

        for index, (option_data, score) in enumerate(stage1_rows, start=1):
            alternative_id = f"{scenario_id}:{option_data['id']}"
            assessment = session.scalar(
                select(Assessment).where(
                    Assessment.situation_id == scenario_id,
                    Assessment.alternative_id == alternative_id,
                    Assessment.assessment_version == 1,
                )
            )
            if assessment is None:
                assessment = Assessment(
                    situation_id=scenario_id,
                    alternative_id=alternative_id,
                    assessment_version=1,
                    criterion_scores=option_data["metrics"],
                    criterion_weights=scenario["operational_weights"],
                    overall_score=round(score, 2),
                )
                session.add(assessment)
                session.flush()
            else:
                assessment.criterion_scores = option_data["metrics"]
                assessment.criterion_weights = scenario["operational_weights"]
                assessment.overall_score = round(score, 2)
            assessment_key = node_key("Assessment", f"{scenario_id}:{option_data['id']}:stage1-v1")
            add_node(
                session,
                assessment_key,
                "Assessment",
                f"Stage 1 suitability for {option_data['name']}",
                {
                    "scenario_id": scenario_id,
                    "alternative_id": alternative_id,
                    "overall_score": round(score, 2),
                    "rank": index,
                    "weights": scenario["operational_weights"],
                    "criterion_scores": option_data["metrics"],
                    "stage": 1,
                },
            )
            add_relationship(session, situation_key, assessment_key, "HAS_ASSESSMENT")
            add_relationship(
                session,
                assessment_key,
                node_key("Alternative", alternative_id),
                "EVALUATES",
                {"score": round(score, 2), "rank": index},
            )

        for profile in profiles:
            decision = decision_by_pair[(scenario_id, profile["id"])]
            event_id = f"{scenario_id}:{profile['id']}:training"
            preference_id = profile_preference_ids[profile["id"]]
            decision_event = session.get(DecisionEvent, event_id)
            if decision_event is None:
                decision_event = DecisionEvent(
                    id=event_id,
                    situation_id=scenario_id,
                    decision_maker_id=profile["id"],
                    preference_id=preference_id,
                    event_number=1,
                    status="completed",
                    event_context={"source": "synthetic_training", "scenario_id": scenario_id},
                )
                session.add(decision_event)
                session.flush()
            else:
                decision_event.preference_id = preference_id
                decision_event.status = "completed"
                decision_event.event_context = {"source": "synthetic_training", "scenario_id": scenario_id}

            available_options = [
                option for option in scenario["options"]
                if option["available"] and option["attributes"]["capacity_units"] >= scenario["required_units"]
            ]
            utilities = {
                option["id"]: sum(
                    option["metrics"][key] * profile["preferences"][key]
                    for key in criterion_keys
                )
                for option in available_options
            }
            probabilities = _softmax(utilities)
            predicted_option = max(utilities, key=utilities.get)
            event_key = node_key("DecisionEvent", event_id)
            add_node(
                session,
                event_key,
                "DecisionEvent",
                f"{profile['name']} decision event for {scenario_id}",
                {
                    "scenario_id": scenario_id,
                    "decision_maker_id": profile["id"],
                    "event_number": 1,
                    "status": "completed",
                    "source": "synthetic_training",
                },
            )
            add_relationship(session, situation_key, event_key, "HAS_DECISION_EVENT")
            add_relationship(
                session,
                event_key,
                node_key("Preference", profile["id"]),
                "INFLUENCED_BY",
                {"factor": "decision-maker preference weights"},
            )
            add_relationship(
                session,
                event_key,
                context_key,
                "INFLUENCED_BY",
                {"factor": "scenario decision conflict"},
            )

            for option in available_options:
                alternative_id = f"{scenario_id}:{option['id']}"
                prediction = session.scalar(
                    select(Prediction).where(
                        Prediction.decision_event_id == event_id,
                        Prediction.alternative_id == alternative_id,
                        Prediction.model_version == "synthetic-v1",
                    )
                )
                if prediction is None:
                    prediction = Prediction(
                        decision_event_id=event_id,
                        alternative_id=alternative_id,
                        utility_score=utilities[option["id"]],
                        probability=probabilities[option["id"]],
                        model_name="illustrative_softmax",
                        model_version="synthetic-v1",
                    )
                    session.add(prediction)
                else:
                    prediction.utility_score = utilities[option["id"]]
                    prediction.probability = probabilities[option["id"]]
                prediction_key = node_key(
                    "Prediction",
                    f"{event_id}:{option['id']}",
                )
                add_node(
                    session,
                    prediction_key,
                    "Prediction",
                    f"Predicted probability for {option['name']}",
                    {
                        "decision_event_id": event_id,
                        "alternative_id": alternative_id,
                        "probability": probabilities[option["id"]],
                        "utility_score": utilities[option["id"]],
                        "model_name": "illustrative_softmax",
                        "model_version": "synthetic-v1",
                    },
                )
                add_relationship(
                    session,
                    event_key,
                    prediction_key,
                    "HAS_PREDICTION",
                    {"probability": probabilities[option["id"]]},
                )
                add_relationship(
                    session,
                    prediction_key,
                    node_key("Alternative", alternative_id),
                    "EVALUATES",
                    {"probability": probabilities[option["id"]]},
                )
                if option["id"] == predicted_option:
                    add_relationship(
                        session,
                        event_key,
                        prediction_key,
                        "PREDICTED_AS",
                        {"probability": probabilities[option["id"]]},
                    )
                    add_relationship(
                        session,
                        prediction_key,
                        node_key("Alternative", alternative_id),
                        "PREDICTED_AS",
                        {"probability": probabilities[option["id"]]},
                    )

            selected_alternative_id = f"{scenario_id}:{decision['recorded_choice_id']}"
            db_decision = session.scalar(
                select(Decision).where(Decision.decision_event_id == event_id)
            )
            if db_decision is None:
                db_decision = Decision(
                    decision_event_id=event_id,
                    chosen_alternative_id=selected_alternative_id,
                    rationale=decision["rationale"],
                    label_source="synthetic_example",
                )
                session.add(db_decision)
                session.flush()
            else:
                db_decision.chosen_alternative_id = selected_alternative_id
                db_decision.rationale = decision["rationale"]
                db_decision.label_source = "synthetic_example"
            decision_key = node_key("Decision", decision["decision_id"])
            selected_name = next(
                option["name"] for option in scenario["options"]
                if option["id"] == decision["recorded_choice_id"]
            )
            add_node(
                session,
                decision_key,
                "Decision",
                f"{profile['name']} selected {selected_name}",
                {
                    "scenario_id": scenario_id,
                    "decision_maker_id": profile["id"],
                    "recorded_choice_id": decision["recorded_choice_id"],
                    "rationale": decision["rationale"],
                    "label_source": "synthetic_example",
                    "predicted_choice_id": predicted_option,
                    "prediction_matches_recorded": predicted_option == decision["recorded_choice_id"],
                },
            )
            add_relationship(session, event_key, decision_key, "HAS_DECISION")
            add_relationship(
                session,
                decision_key,
                node_key("Alternative", selected_alternative_id),
                "SELECTED",
                {"rationale": decision["rationale"], "label_source": "synthetic_example"},
            )

            outcome_data = decision["outcome"]
            db_outcome = session.scalar(
                select(Outcome).where(Outcome.decision_id == db_decision.id)
            )
            outcome_metrics = {
                "actual_delivery_days": float(outcome_data["actual_delivery_days"]),
                "cost_variance_percent": float(outcome_data["cost_variance_percent"]),
                "stockout_units": int(outcome_data["stockout_units"]),
            }
            if db_outcome is None:
                db_outcome = Outcome(
                    decision_id=db_decision.id,
                    status=outcome_data["status"],
                    summary=outcome_data["notes"],
                    metrics=outcome_metrics,
                    actual_observed=False,
                )
                session.add(db_outcome)
                session.flush()
            else:
                db_outcome.status = outcome_data["status"]
                db_outcome.summary = outcome_data["notes"]
                db_outcome.metrics = outcome_metrics
                db_outcome.actual_observed = False
            outcome_key = node_key("Outcome", decision["decision_id"])
            add_node(
                session,
                outcome_key,
                "Outcome",
                f"{outcome_data['status'].title()} outcome for {decision['decision_id']}",
                {
                    **outcome_metrics,
                    "status": outcome_data["status"],
                    "summary": outcome_data["notes"],
                    "actual_observed": False,
                    "source": "synthetic_training",
                },
            )
            add_relationship(session, decision_key, outcome_key, "RESULTED_IN")

    for scenario_id, scenario in scenarios.items():
        situation_key = node_key("Situation", scenario_id)
        for other_id, other in scenarios.items():
            if scenario_id >= other_id:
                continue
            same_risk = scenario["stockout_risk"] == other["stockout_risk"]
            similar_horizon = abs(
                scenario["required_within_days"] - other["required_within_days"]
            ) <= 3
            shared_keywords = set(
                scenario["decision_conflict"].lower().replace("-", " ").split()
            ) & set(other["decision_conflict"].lower().replace("-", " ").split())
            score = (
                0.5 * same_risk
                + 0.3 * similar_horizon
                + 0.2 * min(len(shared_keywords) / 3, 1)
            )
            if score >= 0.5:
                add_relationship(
                    session,
                    situation_key,
                    node_key("Situation", other_id),
                    "SIMILAR_TO",
                    {"score": round(score, 3), "shared_risk_level": same_risk, "similar_horizon": similar_horizon},
                )

    session.commit()
    return {
        "scenario_count": len(scenarios),
        "decision_maker_count": len(profiles),
        "decision_count": len(decisions),
        "seeded": True,
    }


def build_networkx_graph(session: Session) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    nodes = session.scalars(select(ContextNode)).all()
    relationships = session.scalars(select(ContextRelationship)).all()
    for node in nodes:
        graph.add_node(node.node_key, **_node_payload(node))
    for relationship in relationships:
        graph.add_edge(
            relationship.source_node_key,
            relationship.target_node_key,
            key=relationship.id,
            **_relationship_payload(relationship),
        )
    return graph


def graph_snapshot(session: Session, scenario_id: str | None = None, depth: int = 2) -> dict:
    graph = build_networkx_graph(session)
    root = node_key("Situation", scenario_id) if scenario_id else None
    if root and root not in graph:
        return {"directed": False, "node_count": 0, "relationship_count": 0, "nodes": [], "relationships": []}

    if root:
        distances = {root: 0}
        pending = deque([root])
        while pending:
            current = pending.popleft()
            current_distance = distances[current]
            if current_distance >= depth:
                continue
            for neighbor, edges in graph[current].items():
                candidate_distance = current_distance + 1
                if candidate_distance >= distances.get(neighbor, depth + 1):
                    continue
                neighbor_data = graph.nodes[neighbor]
                if neighbor_data["type"] == "Situation" and neighbor != root:
                    distances[neighbor] = candidate_distance
                    continue
                neighbor_scenario_id = neighbor_data["properties"].get("scenario_id")
                if neighbor_scenario_id and neighbor_scenario_id != scenario_id:
                    continue
                distances[neighbor] = candidate_distance
                if any(edge["type"] != "SIMILAR_TO" for edge in edges.values()):
                    pending.append(neighbor)
        selected_nodes = set(distances)
    else:
        selected_nodes = set(graph.nodes)
    nodes = [_node_payload_from_graph(graph.nodes[key]) for key in selected_nodes]
    relationships = [
        dict(data)
        for source, target, _, data in graph.edges(keys=True, data=True)
        if source in selected_nodes and target in selected_nodes
    ]
    return {
        "directed": False,
        "traversal": "bidirectional",
        "root": root,
        "depth": depth,
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "nodes": nodes,
        "relationships": relationships,
    }


def _node_payload_from_graph(data: dict) -> dict:
    return {
        key: data[key]
        for key in ("id", "type", "name", "properties", "provenance", "is_synthetic", "created_at", "updated_at")
    }


def neighbors(session: Session, key: str, depth: int = 1) -> dict:
    graph = build_networkx_graph(session)
    if key not in graph:
        return {"node": None, "depth": depth, "nodes": [], "relationships": []}
    distances = nx.single_source_shortest_path_length(graph, key, cutoff=depth)
    selected_nodes = set(distances)
    relationships = [
        {
            **dict(data),
            "traversal_direction": "outgoing" if data["source"] == key else "incoming",
        }
        for source, target, _, data in graph.edges(keys=True, data=True)
        if source in selected_nodes and target in selected_nodes
    ]
    return {
        "node": _node_payload_from_graph(graph.nodes[key]),
        "depth": depth,
        "nodes": [
            {**_node_payload_from_graph(graph.nodes[node]), "distance": distances[node]}
            for node in selected_nodes
        ],
        "relationships": relationships,
    }


def similar_situations(session: Session, scenario_id: str, limit: int = 5) -> dict:
    key = node_key("Situation", scenario_id)
    graph = build_networkx_graph(session)
    if key not in graph:
        return {"scenario_id": scenario_id, "similar_situations": []}
    similarities = []
    for neighbor, edges in graph[key].items():
        for edge in edges.values():
            if edge["type"] == "SIMILAR_TO":
                similarities.append({
                    "scenario_id": graph.nodes[neighbor]["id"].split(":", 1)[1],
                    "name": graph.nodes[neighbor]["name"],
                    "score": edge["properties"].get("score", 0),
                    "relationship": edge["type"],
                    "provenance": edge["provenance"],
                })
    similarities.sort(key=lambda row: row["score"], reverse=True)
    return {"scenario_id": scenario_id, "similar_situations": similarities[:limit]}
