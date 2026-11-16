"""Idempotently load the synthetic decision-training fixture into PostgreSQL."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.context_graph import seed_context_graph
from app.database import Base, SessionLocal, engine
from app.init_db import create_tables
from app.main import CRITERIA, TRAINING_DATA
from app.models import (
    Alternative,
    Assessment,
    ContextNode,
    ContextRelationship,
    Decision,
    DecisionEvent,
    DecisionMaker,
    Outcome,
    Prediction,
    Preference,
    Situation,
)


def _count(session: Session, model: type[Base], *criteria: ColumnElement[bool]) -> int:
    statement = select(func.count()).select_from(model)
    if criteria:
        statement = statement.where(*criteria)
    return int(session.scalar(statement))


def seed_postgresql() -> dict[str, int]:
    if engine.dialect.name != "postgresql":
        raise RuntimeError(
            "The training seed must run against PostgreSQL; check backend/.env DATABASE_URL."
        )

    create_tables()
    scenario_ids = tuple(TRAINING_DATA["scenarios"])
    decision_maker_ids = tuple(profile["id"] for profile in TRAINING_DATA["profiles"])
    event_ids = tuple(
        f"{scenario_id}:{decision_maker_id}:training"
        for scenario_id in scenario_ids
        for decision_maker_id in decision_maker_ids
    )

    with SessionLocal() as session:
        seed_context_graph(session, TRAINING_DATA, CRITERIA)
        counts = {
            "situations": _count(
                session,
                Situation,
                Situation.id.in_(scenario_ids),
                Situation.data_source == "synthetic",
            ),
            "alternatives": _count(
                session,
                Alternative,
                Alternative.situation_id.in_(scenario_ids),
                Alternative.data_source == "synthetic",
            ),
            "decision_makers": _count(
                session,
                DecisionMaker,
                DecisionMaker.id.in_(decision_maker_ids),
                DecisionMaker.data_source == "synthetic",
            ),
            "preferences": _count(
                session,
                Preference,
                Preference.decision_maker_id.in_(decision_maker_ids),
                Preference.version == 1,
                Preference.data_source == "synthetic",
            ),
            "assessments": _count(
                session,
                Assessment,
                Assessment.situation_id.in_(scenario_ids),
                Assessment.assessment_version == 1,
                Assessment.data_source == "synthetic",
            ),
            "decision_events": _count(
                session,
                DecisionEvent,
                DecisionEvent.id.in_(event_ids),
                DecisionEvent.data_source == "synthetic",
            ),
            "predictions": _count(
                session,
                Prediction,
                Prediction.decision_event_id.in_(event_ids),
                Prediction.model_version == "synthetic-v1",
                Prediction.data_source == "synthetic",
            ),
            "decisions": _count(
                session,
                Decision,
                Decision.decision_event_id.in_(event_ids),
                Decision.label_source == "synthetic",
                Decision.data_source == "synthetic",
            ),
            "outcomes": _count(
                session,
                Outcome,
                Outcome.decision_id.in_(
                    select(Decision.id).where(
                        Decision.decision_event_id.in_(event_ids),
                        Decision.data_source == "synthetic",
                    )
                ),
                Outcome.actual_observed.is_(False),
                Outcome.data_source == "synthetic",
            ),
            "context_nodes": _count(
                session,
                ContextNode,
                ContextNode.provenance == "synthetic",
                ContextNode.data_source == "synthetic",
            ),
            "context_relationships": _count(
                session,
                ContextRelationship,
                ContextRelationship.provenance == "synthetic",
                ContextRelationship.data_source == "synthetic",
            ),
        }

    expected = {
        "situations": 8,
        "alternatives": 24,
        "decision_makers": 4,
        "preferences": 4,
        "assessments": 24,
        "decision_events": 32,
        "predictions": 96,
        "decisions": 32,
        "outcomes": 32,
    }
    mismatches = {
        table: {"expected": expected_count, "actual": counts[table]}
        for table, expected_count in expected.items()
        if counts[table] != expected_count
    }
    if mismatches:
        raise RuntimeError(f"PostgreSQL training seed verification failed: {mismatches}")
    if counts["context_nodes"] == 0 or counts["context_relationships"] == 0:
        raise RuntimeError("PostgreSQL training seed created no synthetic context graph.")
    return counts


if __name__ == "__main__":
    for table, count in seed_postgresql().items():
        print(f"{table}: {count}")
