import unittest

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.context_graph import (
    build_networkx_graph,
    graph_snapshot,
    neighbors,
    seed_context_graph,
    similar_situations,
)
from app.database import Base
from app.main import (
    CRITERIA,
    TRAINING_DATA,
    ContextNodeRequest,
    ContextRelationshipRequest,
    create_context_node,
    create_context_relationship,
)


class ContextGraphTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        seed_context_graph(self.session, TRAINING_DATA, CRITERIA)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_seed_persists_pipeline_nodes_and_relationships(self):
        result = graph_snapshot(self.session, "SCN-001", depth=5)
        node_types = {node["type"] for node in result["nodes"]}
        relationship_types = {relationship["type"] for relationship in result["relationships"]}

        self.assertIn("Situation", node_types)
        self.assertIn("Alternative", node_types)
        self.assertIn("Criterion", node_types)
        self.assertIn("DecisionMaker", node_types)
        self.assertIn("Preference", node_types)
        self.assertIn("Assessment", node_types)
        self.assertIn("DecisionEvent", node_types)
        self.assertIn("Prediction", node_types)
        self.assertIn("Decision", node_types)
        self.assertIn("Outcome", node_types)
        self.assertIn("Context", node_types)
        self.assertTrue({
            "HAS_PARAMETER",
            "HAS_ALTERNATIVE",
            "EVALUATED_AGAINST",
            "HAS_PREFERENCE",
            "INFLUENCED_BY",
            "PREDICTED_AS",
            "HAS_PREDICTION",
            "SELECTED",
            "RESULTED_IN",
            "PARTICIPATED_IN",
            "OCCURRED_IN",
        }.issubset(relationship_types))
        self.assertTrue(all(relationship["bidirectional"] for relationship in result["relationships"]))

    def test_seed_persists_complete_synthetic_training_records(self):
        counts = {
            model.__tablename__: self.session.scalar(
                select(func.count()).select_from(model)
            )
            for model in (
                models.Situation,
                models.Alternative,
                models.DecisionMaker,
                models.Preference,
                models.Assessment,
                models.DecisionEvent,
                models.Prediction,
                models.Decision,
                models.Outcome,
            )
        }
        self.assertEqual(counts, {
            "situations": len(TRAINING_DATA["scenarios"]),
            "alternatives": len(TRAINING_DATA["scenarios"]) * 3,
            "decision_makers": 4,
            "preferences": 4,
            "assessments": len(TRAINING_DATA["scenarios"]) * 3,
            "decision_events": len(TRAINING_DATA["decisions"]),
            "predictions": len(TRAINING_DATA["decisions"]) * 3,
            "decisions": len(TRAINING_DATA["decisions"]),
            "outcomes": len(TRAINING_DATA["decisions"]),
        })

        scenario = self.session.get(models.Situation, "SCN-001")
        self.assertEqual(scenario.context_data["demand_increase_percent"], 30)
        self.assertEqual(scenario.context_data["current_inventory_days"], 5)
        self.assertEqual(scenario.context_data["required_inventory_days"], 20)
        self.assertEqual(scenario.context_data["available_budget_lakh"], 20.0)
        self.assertEqual(scenario.context_data["data_source"], "synthetic")

        expected_scores = {"SUP-A": 75.25, "SUP-B": 82.25, "SUP-C": 82.25}
        for option_id, expected_score in expected_scores.items():
            assessment = self.session.scalar(
                select(models.Assessment).where(
                    models.Assessment.situation_id == "SCN-001",
                    models.Assessment.alternative_id == f"SCN-001:{option_id}",
                )
            )
            self.assertAlmostEqual(assessment.overall_score, expected_score)

        expected_probabilities = {
            "DM-01": {"SUP-A": 0.65, "SUP-B": 0.20, "SUP-C": 0.15},
            "DM-02": {"SUP-A": 0.10, "SUP-B": 0.30, "SUP-C": 0.60},
            "DM-03": {"SUP-A": 0.10, "SUP-B": 0.70, "SUP-C": 0.20},
            "DM-04": {"SUP-A": 0.15, "SUP-B": 0.45, "SUP-C": 0.40},
        }
        expected_choices = {"DM-01": "SUP-A", "DM-02": "SUP-C", "DM-03": "SUP-B", "DM-04": "SUP-C"}
        for decision_maker_id, expected in expected_probabilities.items():
            event_id = f"SCN-001:{decision_maker_id}:training"
            predictions = self.session.scalars(
                select(models.Prediction).where(models.Prediction.decision_event_id == event_id)
            ).all()
            self.assertEqual(
                {row.alternative_id.rsplit(":", 1)[1]: row.probability for row in predictions},
                expected,
            )
            decision = self.session.scalar(
                select(models.Decision).where(models.Decision.decision_event_id == event_id)
            )
            self.assertEqual(decision.chosen_alternative_id, f"SCN-001:{expected_choices[decision_maker_id]}")
            self.assertEqual(decision.label_source, "synthetic")

        self.assertTrue(all(
            node.data_source == "synthetic"
            and node.provenance == "synthetic"
            and node.properties["data_source"] == "synthetic"
            for node in self.session.scalars(select(models.ContextNode)).all()
        ))
        self.assertTrue(all(
            relationship.data_source == "synthetic"
            and relationship.provenance == "synthetic"
            and relationship.properties["data_source"] == "synthetic"
            for relationship in self.session.scalars(select(models.ContextRelationship)).all()
        ))
        for model in (
            models.Situation,
            models.Alternative,
            models.DecisionMaker,
            models.Preference,
            models.Assessment,
            models.DecisionEvent,
            models.Prediction,
            models.Decision,
            models.Outcome,
        ):
            self.assertTrue(all(
                row.data_source == "synthetic"
                for row in self.session.scalars(select(model)).all()
            ), model.__tablename__)

    def test_networkx_traversal_is_bidirectional(self):
        graph = build_networkx_graph(self.session)
        alternative = "Alternative:SCN-001:SUP-C"
        situation = "Situation:SCN-001"

        self.assertTrue(graph.has_edge(alternative, situation))
        result = neighbors(self.session, alternative, depth=1)
        situation_node = next(node for node in result["nodes"] if node["id"] == situation)
        self.assertEqual(situation_node["distance"], 1)
        link = next(
            relationship for relationship in result["relationships"]
            if relationship["type"] == "HAS_ALTERNATIVE"
        )
        self.assertEqual(link["traversal_direction"], "incoming")

    def test_similar_situations_are_retrievable(self):
        result = similar_situations(self.session, "SCN-001", limit=3)

        self.assertTrue(result["similar_situations"])
        self.assertLessEqual(len(result["similar_situations"]), 3)
        self.assertTrue(all(item["relationship"] == "SIMILAR_TO" for item in result["similar_situations"]))
        scenario_graph = graph_snapshot(self.session, "SCN-001", depth=5)
        situation_nodes = [node for node in scenario_graph["nodes"] if node["type"] == "Situation"]
        self.assertGreater(len(situation_nodes), 1)
        self.assertTrue(all(
            node["id"] == "Situation:SCN-001"
            or any(
                edge["type"] == "SIMILAR_TO"
                and {edge["source"], edge["target"]} == {"Situation:SCN-001", node["id"]}
                for edge in scenario_graph["relationships"]
            )
            for node in situation_nodes
        ))
        self.assertTrue(all(
            node["properties"].get("scenario_id", "SCN-001") == "SCN-001"
            for node in scenario_graph["nodes"]
            if node["type"] != "Situation"
        ))
        self.assertTrue(any(
            edge["type"] == "SIMILAR_TO"
            for edge in scenario_graph["relationships"]
        ))

    def test_context_nodes_and_typed_relationships_are_persisted_and_traversable(self):
        created = create_context_node(
            ContextNodeRequest(
                node_id="SCN-001:policy-01",
                name="Cold-chain handling policy",
                context_type="policy",
                situation_id="SCN-001",
                properties={"description": "Require controlled transport."},
                provenance="test:context-node",
            ),
            self.session,
        )
        relationship = create_context_relationship(
            ContextRelationshipRequest(
                source_node_id=created["node"]["id"],
                target_node_id="Situation:SCN-001",
                relationship_type="INFLUENCED_BY",
                properties={"reason": "policy applies to this situation"},
                provenance="test:relationship",
            ),
            self.session,
        )
        result = neighbors(self.session, "Situation:SCN-001", depth=1)

        self.assertEqual(relationship["type"], "INFLUENCED_BY")
        self.assertTrue(relationship["bidirectional_traversal"])
        self.assertIn(created["node"]["id"], {node["id"] for node in result["nodes"]})
        self.assertTrue(any(
            edge["type"] == "INFLUENCED_BY" and edge["traversal_direction"] == "incoming"
            for edge in result["relationships"]
        ))

    def test_seeding_is_idempotent(self):
        counts_before = (
            self.session.scalar(select(func.count()).select_from(models.ContextNode)),
            self.session.scalar(select(func.count()).select_from(models.ContextRelationship)),
        )
        seed_context_graph(self.session, TRAINING_DATA, CRITERIA)
        node_count = self.session.scalar(select(func.count()).select_from(models.ContextNode))
        relationship_count = self.session.scalar(select(func.count()).select_from(models.ContextRelationship))
        situation_count = self.session.scalar(select(func.count()).select_from(models.Situation))
        decision_count = self.session.scalar(select(func.count()).select_from(models.Decision))
        outcome_count = self.session.scalar(select(func.count()).select_from(models.Outcome))

        self.assertGreater(node_count, 0)
        self.assertGreater(relationship_count, 0)
        self.assertEqual((node_count, relationship_count), counts_before)
        self.assertEqual(situation_count, len(TRAINING_DATA["scenarios"]))
        self.assertEqual(decision_count, len(TRAINING_DATA["decisions"]))
        self.assertEqual(outcome_count, len(TRAINING_DATA["decisions"]))


if __name__ == "__main__":
    unittest.main()
