import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import main
from app.context_graph import graph_snapshot, seed_context_graph
from app.database import Base
from app.init_db import apply_postgresql_migrations, verify_postgresql_schema
from app.models import (
    ContextNode,
    ContextRelationship,
    Decision,
    DecisionEvent,
    DecisionRule,
    ExtractedRule,
    Outcome,
    RuleApplication,
    RuleEvidence,
    RuleFeedback,
    RuleExtraction,
    RuleReview,
)
from app.rule_api import (
    RuleReviewRequest,
    publish_rule,
    record_rule_feedback,
    review_rule,
)
from app.rule_memory import (
    evaluate_conditions,
    extract_candidate_rules,
    relevant_rules,
)


POSTGRES_TEST_URL = os.getenv("POSTGRES_TEST_DATABASE_URL")


@unittest.skipUnless(
    POSTGRES_TEST_URL,
    "Set POSTGRES_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)
class PostgreSQLWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sqlalchemy import create_engine

        cls.engine = create_engine(POSTGRES_TEST_URL, pool_pre_ping=True)
        if cls.engine.dialect.name != "postgresql":
            raise RuntimeError("POSTGRES_TEST_DATABASE_URL must use PostgreSQL.")
        cls.connection = cls.engine.connect()
        cls.outer_transaction = cls.connection.begin()
        cls.schema = f"di_test_{uuid4().hex}"
        cls.connection.execute(text(f'CREATE SCHEMA "{cls.schema}"'))
        cls.connection.execute(text(f'SET LOCAL search_path TO "{cls.schema}"'))
        Base.metadata.create_all(cls.connection)
        apply_postgresql_migrations(cls.connection)
        verify_postgresql_schema(cls.connection)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "outer_transaction", None) is not None:
            cls.outer_transaction.rollback()
        if getattr(cls, "connection", None) is not None:
            cls.connection.close()
        if getattr(cls, "engine", None) is not None:
            cls.engine.dispose()

    def setUp(self):
        self.test_transaction = self.connection.begin_nested()
        self.db = Session(
            bind=self.connection,
            join_transaction_mode="create_savepoint",
        )
        seed_context_graph(self.db, main.TRAINING_DATA, main.CRITERIA)

    def tearDown(self):
        self.db.close()
        if self.test_transaction.is_active:
            self.test_transaction.rollback()

    def _mark_seeded_decisions_observed(self):
        for decision in self.db.scalars(select(Decision)).all():
            decision.data_source = "observed_human"
            decision.label_source = "observed_human"
        self.db.flush()

    def _record_human_decision(
        self,
        scenario_id="SCN-001",
        decision_maker_id="DM-01",
        option_id="SUP-A",
        prediction_event_id=None,
    ):
        return main.record_human_decision(
            main.HumanDecisionRequest(
                scenario_id=scenario_id,
                decision_maker_id=decision_maker_id,
                chosen_option_id=option_id,
                prediction_event_id=prediction_event_id,
                operational_objective="Replenish stock before the delivery deadline.",
                rationale="Selected this supplier to meet the budget and timing constraints.",
                constraints=["Remain within the available budget."],
                influencing_factors=["Supplier capacity is confirmed."],
                reasoning_steps=["Check capacity.", "Compare the delivery deadline."],
                confidence=0.8,
            ),
            self.db,
        )

    def test_postgresql_schema_constraints_and_graph_reconstruction(self):
        verify_postgresql_schema(self.connection)
        self.assertIn("decision_rules", self.connection.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")
        ).scalars().all())
        graph = graph_snapshot(self.db, "SCN-001", depth=5)
        self.assertGreater(graph["node_count"], 0)
        self.assertGreater(graph["relationship_count"], 0)
        self.assertTrue(any(
            relationship["source"] in {node["id"] for node in graph["nodes"]}
            and relationship["target"] in {node["id"] for node in graph["nodes"]}
            for relationship in graph["relationships"]
        ))

    def test_decision_without_predictions_snapshot_immutability_and_outcome(self):
        record = self._record_human_decision()
        self.assertEqual(record["probabilities"], [])
        self.assertIsNone(record["predicted_choice"])
        trace = main.get_decision_trace(record["id"], self.db)
        self.assertEqual(trace["operational_objective"], "Replenish stock before the delivery deadline.")
        self.assertEqual(trace["recorded_choice"]["alternative_code"], "SUP-A")
        self.assertEqual(len(trace["alternative_snapshot"]), 3)
        self.assertEqual(trace["human_rationale"]["source"], "human_stated")
        self.assertEqual(trace["predictions"], [])
        self.assertIsNone(trace["model_explanation"])
        self.assertIsNone(trace["outcome"])

        event = self.db.get(DecisionEvent, record["prediction_event_id"])
        original_snapshot = event.event_context["reasoning_snapshot"]
        situation = self.db.get(main.Situation, "SCN-001")
        situation.title = "Changed live title after the decision"
        self.db.commit()
        self.db.expire_all()
        trace_after_live_update = main.get_decision_trace(record["id"], self.db)
        self.assertEqual(
            trace_after_live_update["situation_snapshot"]["title"],
            original_snapshot["situation"]["title"],
        )

        event = self.db.get(DecisionEvent, record["prediction_event_id"])
        updated_context = dict(event.event_context)
        updated_snapshot = dict(updated_context["reasoning_snapshot"])
        updated_snapshot["situation"] = {"title": "attempted rewrite"}
        updated_context["reasoning_snapshot"] = updated_snapshot
        event.event_context = updated_context
        with self.assertRaises((ValueError, SQLAlchemyError)):
            self.db.commit()
        self.db.rollback()
        self.db.expire_all()
        self.assertEqual(
            main.get_decision_trace(record["id"], self.db)["situation_snapshot"]["title"],
            original_snapshot["situation"]["title"],
        )

        outcome_result = main.persist_decision_outcome(
            self.db,
            record["id"],
            main.OutcomeFeedbackRequest(
                status="delayed",
                notes="The delivery arrived after the requested deadline.",
            ),
        )
        self.assertEqual(outcome_result["outcome"]["status"], "delayed")
        self.assertEqual(
            main.get_decision_trace(record["id"], self.db)["outcome"]["status"],
            "delayed",
        )

    def test_prediction_explanation_human_rationale_and_outcome_remain_separate(self):
        self._mark_seeded_decisions_observed()
        with tempfile.TemporaryDirectory() as directory:
            from app.ml.train import train_model
            from unittest.mock import patch

            model_path = Path(directory) / "random_forest.joblib"
            train_model(self.db, model_path, model_name="random_forest")
            with patch(
                "app.ml.predict.MODEL_PATHS",
                {"random_forest": model_path},
            ):
                prediction = main.predict_ml_choice(
                    main.MLPredictionRequest(
                        scenario_id="SCN-001",
                        decision_maker_id="DM-01",
                        model_name="random_forest",
                    ),
                    self.db,
                )

        event = self.db.get(DecisionEvent, prediction["prediction_event_id"])
        record = self._record_human_decision(
            prediction_event_id=event.id,
        )
        trace = main.get_decision_trace(record["id"], self.db)
        self.assertEqual(len(trace["predictions"]), 3)
        self.assertEqual(
            trace["human_rationale"]["text"],
            "Selected this supplier to meet the budget and timing constraints.",
        )
        self.assertIsNotNone(trace["model_explanation"])
        self.assertEqual(trace["model_explanation"]["source"], "model")
        self.assertIn("model_version", trace["model_explanation"])
        self.assertTrue(trace["model_explanation"]["extracted_rules"])
        self.assertTrue(all(
            rule["decision_id"] == self.db.get(Decision, event.decision.id).id
            for rule in trace["model_explanation"]["extracted_rules"]
        ))
        self.assertNotIn(
            trace["human_rationale"]["text"],
            str(trace["model_explanation"]),
        )
        outcome = main.persist_decision_outcome(
            self.db,
            record["id"],
            main.OutcomeFeedbackRequest(
                status="fulfilled",
                notes="The order arrived within the required window.",
            ),
        )
        self.assertEqual(outcome["outcome"]["status"], "fulfilled")
        rebuilt = graph_snapshot(self.db, "SCN-001", depth=5)
        self.assertTrue(any(node["type"] == "ModelExplanation" for node in rebuilt["nodes"]))
        self.assertTrue(any(node["type"] == "Outcome" for node in rebuilt["nodes"]))

    def test_observed_rule_review_publication_retrieval_and_feedback(self):
        self._mark_seeded_decisions_observed()
        extraction_result = extract_candidate_rules(
            self.db,
            min_matching_decisions=2,
            min_scenarios=1,
            min_decision_makers=1,
            max_rules=5,
        )
        self.db.commit()
        self.assertGreater(extraction_result["candidate_count"], 0)
        rule = self.db.get(DecisionRule, extraction_result["candidate_rule_ids"][0])
        self.assertIsNotNone(rule)
        self.assertTrue(rule.evidences)
        self.assertTrue(all(
            evidence.evidence_type in {"supporting", "contradictory"}
            for evidence in rule.evidences
        ))

        review_rule(
            rule.id,
            RuleReviewRequest(
                reviewer_id="DM-03",
                review_stage="evidence",
                review_decision="approved",
                comments="Historical evidence reviewed.",
            ),
            self.db,
        )
        review_rule(
            rule.id,
            RuleReviewRequest(
                reviewer_id="DM-04",
                review_stage="publication",
                review_decision="approved",
                comments="Approved for the shared memory graph.",
            ),
            self.db,
        )
        publish_rule(rule.id, self.db)
        self.db.expire_all()
        rule = self.db.get(DecisionRule, rule.id)
        self.assertIsNotNone(rule.published_at)
        graph_rule_key = f"DecisionRule:{rule.id}:v{rule.version}"
        graph = graph_snapshot(self.db, "SCN-001", depth=5)
        self.assertTrue(any(node["id"] == graph_rule_key for node in graph["nodes"]))
        self.assertTrue(any(
            edge["type"] == "SUPPORTED_BY"
            for edge in graph["relationships"]
        ))
        self.assertTrue(any(
            edge["type"] == "APPROVED_BY"
            for edge in graph["relationships"]
        ))

        evidence = rule.evidences[0]
        evidence_event = evidence.decision.decision_event
        from app.ml.train import load_training_dataset

        dataset = load_training_dataset(self.db, decision_source="observed_human")
        for rule_evidence in rule.evidences:
            row_index = dataset.decision_ids.index(rule_evidence.decision_id)
            self.assertTrue(
                evaluate_conditions(
                    rule.conditions_json,
                    dataset.rows[row_index],
                    rule.extraction.extraction_parameters["numeric_medians"],
                ),
                msg=(
                    f"Extracted condition did not match evidence "
                    f"{rule_evidence.decision_id}: conditions={rule.conditions_json!r}; "
                    f"row={dataset.rows[row_index]!r}"
                ),
            )
        matching = relevant_rules(
            self.db,
            evidence_event.situation_id,
            evidence_event.decision_maker_id,
        )
        self.assertIn(rule.id, {item["id"] for item in matching})

        decision_record = self._record_human_decision(
            scenario_id=evidence_event.situation_id,
            decision_maker_id=evidence_event.decision_maker_id,
            option_id=evidence.decision.chosen_alternative.code,
        )
        application_result = record_rule_feedback(
            rule.id,
            main_rule_feedback_request(
                scenario_id=evidence_event.situation_id,
                decision_id=decision_record["id"],
            ),
            self.db,
        )
        self.assertEqual(
            application_result["application"]["relevance_status"],
            "relevant",
        )
        outcome_result = main.persist_decision_outcome(
            self.db,
            decision_record["id"],
            main.OutcomeFeedbackRequest(
                status="fulfilled",
                notes="Recorded outcome for rule feedback.",
            ),
        )
        decision = self.db.scalar(
            select(Decision).where(Decision.confidence_note == decision_record["id"])
        )
        feedback_result = record_rule_feedback(
            rule.id,
            main_rule_feedback_request(
                scenario_id=evidence_event.situation_id,
                decision_id=decision_record["id"],
                outcome_id=decision.outcome.id,
                feedback_type="supporting",
            ),
            self.db,
        )
        self.assertEqual(feedback_result["feedback"]["feedback_type"], "supporting")
        self.assertTrue(self.db.query(RuleApplication).filter_by(rule_id=rule.id).count() >= 1)
        self.assertTrue(self.db.query(RuleFeedback).filter_by(rule_id=rule.id).count() >= 1)

        rule = self.db.get(DecisionRule, rule.id)
        original_conditions = rule.conditions_json
        rule.conditions_json = []
        with self.assertRaises((ValueError, SQLAlchemyError)):
            self.db.flush()
        self.db.rollback()
        self.db.expire_all()
        self.assertEqual(self.db.get(DecisionRule, rule.id).conditions_json, original_conditions)


def main_rule_feedback_request(
    scenario_id: str,
    decision_id: str | None = None,
    outcome_id: int | None = None,
    feedback_type: str | None = None,
):
    from app.rule_api import RuleFeedbackRequest

    return RuleFeedbackRequest(
        scenario_id=scenario_id,
        decision_id=decision_id,
        outcome_id=outcome_id,
        relevance_status="relevant",
        considered_by_human=True,
        feedback="This verified pattern was reviewed in the decision workflow.",
        feedback_type=feedback_type,
    )


if __name__ == "__main__":
    unittest.main()
