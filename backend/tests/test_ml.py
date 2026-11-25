import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import main
from app.context_graph import graph_snapshot, seed_context_graph
from app.database import Base
from app.ml.predict import _load_model_cached, load_model, predict_choice
from app.ml.train import TrainingDataError, load_training_dataset, train_model
from app.models import ContextNode, Decision, DecisionEvent, ExtractedRule, ModelExplanation, Prediction


class MLChoiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        seed_context_graph(self.session, main.TRAINING_DATA, main.CRITERIA)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temporary_directory.name) / "choice_model.joblib"

    def tearDown(self):
        _load_model_cached.cache_clear()
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def mark_seeded_choices_as_observed(self):
        for decision in self.session.query(Decision).all():
            decision.data_source = "observed_human"
            decision.label_source = "observed_human"
        self.session.flush()

    def test_feature_rows_include_preferences_and_all_alternative_criterion_scores(self):
        dataset = load_training_dataset(self.session)

        self.assertIn("preference_cost", dataset.numeric_features)
        self.assertIn("preference_reliability", dataset.numeric_features)
        self.assertIn("alternative_SUP-A_cost", dataset.numeric_features)
        self.assertIn("alternative_SUP-C_risk", dataset.numeric_features)
        self.assertIn("decision_maker_role", dataset.categorical_features)
        self.assertEqual(len(dataset.rows), len(main.TRAINING_DATA["decisions"]))
        self.assertEqual(len(set(dataset.groups)), len(main.TRAINING_DATA["scenarios"]))

    def test_grouped_multinomial_model_predicts_probabilities_for_eligible_alternatives(self):
        metadata = train_model(self.session, self.model_path)

        split = metadata["group_split"]
        self.assertFalse(
            set(split["training_scenario_ids"]) & set(split["holdout_scenario_ids"])
        )
        self.assertEqual(metadata["model_name"], "multinomial_logistic_regression")
        artifact = load_model(self.model_path)
        model = artifact["pipeline"]
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps["classifier"], LogisticRegression)
        self.assertIsInstance(model.named_steps["preprocessor"], ColumnTransformer)
        numeric_pipeline = model.named_steps["preprocessor"].named_transformers_["numeric"]
        self.assertIsInstance(numeric_pipeline.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(numeric_pipeline.named_steps["scaler"], StandardScaler)
        categorical_transformer = model.named_steps["preprocessor"].named_transformers_["categorical"]
        self.assertIsInstance(categorical_transformer, OneHotEncoder)

        result = predict_choice(
            self.session,
            "SCN-002",
            "DM-02",
            model_path=self.model_path,
        )
        self.assertEqual(
            {item["alternative_code"] for item in result["probabilities"]},
            {"SUP-B", "SUP-C"},
        )
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["probabilities"]),
            1.0,
        )
        self.assertEqual(
            result["feature_inputs"]["preference_weights"],
            main.TRAINING_DATA["profiles"][1]["preferences"],
        )
        self.assertIn("SUP-A", result["feature_inputs"]["alternative_criterion_scores"])
        self.assertTrue(all(0 <= item["probability"] <= 1 for item in result["probabilities"]))

    def test_grouped_random_forest_model_predicts_probabilities(self):
        with self.assertRaisesRegex(TrainingDataError, "No observed_human human decision records"):
            train_model(self.session, self.model_path, model_name="random_forest")

        self.mark_seeded_choices_as_observed()
        metadata = train_model(
            self.session,
            self.model_path,
            model_name="random_forest",
        )

        split = metadata["group_split"]
        self.assertFalse(
            set(split["training_scenario_ids"]) & set(split["holdout_scenario_ids"])
        )
        self.assertEqual(metadata["model_name"], "random_forest")
        artifact = load_model(self.model_path)
        model = artifact["pipeline"]
        self.assertIsInstance(model.named_steps["classifier"], RandomForestClassifier)
        self.assertEqual(model.named_steps["classifier"].n_estimators, 300)
        numeric_pipeline = model.named_steps["preprocessor"].named_transformers_["numeric"]
        self.assertIsInstance(numeric_pipeline.named_steps["imputer"], SimpleImputer)
        self.assertNotIn("scaler", numeric_pipeline.named_steps)

        result = predict_choice(
            self.session,
            "SCN-002",
            "DM-02",
            model_path=self.model_path,
            model_name="random_forest",
        )
        self.assertEqual(result["model"]["name"], "random_forest")
        self.assertEqual(
            {item["alternative_code"] for item in result["probabilities"]},
            {"SUP-B", "SUP-C"},
        )
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["probabilities"]),
            1.0,
        )
        self.assertTrue(all(0 <= item["probability"] <= 1 for item in result["probabilities"]))
        explanation = result["explanation"]
        self.assertEqual(explanation["kind"], "random_forest_pattern_analysis")
        self.assertEqual(sum(
            item["votes"] for item in explanation["tree_vote_summary"].values()
        ), 300)
        self.assertTrue(explanation["rule_paths"])
        self.assertTrue(explanation["top_features"])
        self.assertIn("not evidence", explanation["rule_path_note"])

    def test_random_forest_can_train_and_evaluate_synthetic_scenarios_explicitly(self):
        metadata = train_model(
            self.session,
            self.model_path,
            model_name="random_forest",
            training_data_source="synthetic",
        )

        self.assertEqual(metadata["training_data_source"], "synthetic")
        self.assertEqual(
            metadata["training_scenario_count"],
            len(main.TRAINING_DATA["scenarios"]),
        )
        holdout = metadata["group_split"]
        self.assertFalse(
            set(holdout["training_scenario_ids"]) & set(holdout["holdout_scenario_ids"])
        )
        self.assertGreater(holdout["holdout_sample_count"], 0)
        self.assertIn("macro_f1", holdout["holdout_metrics"])
        self.assertIn("multiclass_brier_score", holdout["holdout_metrics"])

        with patch(
            "app.ml.predict.MODEL_PATHS",
            {"random_forest": self.model_path},
        ):
            prediction = predict_choice(
                self.session,
                "SCN-009",
                "DM-01",
                model_name="random_forest",
            )
        self.assertEqual(prediction["model"]["training_data_source"], "synthetic")
        self.assertIn("synthetic demonstration choices", prediction["disclaimer"])

    def test_fastapi_prediction_endpoint_uses_the_trained_model(self):
        train_model(self.session, self.model_path)
        with patch("app.ml.predict.MODEL_PATH", self.model_path):
            result = main.predict_ml_choice(
                main.MLPredictionRequest(
                    scenario_id="SCN-001",
                    decision_maker_id="DM-01",
                ),
                self.session,
            )

        self.assertEqual(result["scenario_id"], "SCN-001")
        self.assertEqual(
            result["model"]["training_sample_count"],
            len(main.TRAINING_DATA["decisions"]),
        )
        self.assertTrue(result["model"]["version"])
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertTrue(result["persisted"])
        self.assertTrue(result["prediction_event_id"])

        event = self.session.get(DecisionEvent, result["prediction_event_id"])
        self.assertIsNotNone(event)
        self.assertEqual(event.status, "predicted")
        self.assertEqual(event.data_source, "model")
        self.assertIsNone(event.decision)
        self.assertEqual(len(event.predictions), 3)
        self.assertTrue(all(item.data_source == "model" for item in event.predictions))
        self.assertTrue(all(
            item.model_version == f"ml-{result['model']['version']}"
            for item in event.predictions
        ))

        with self.assertRaises(main.HTTPException):
            main.record_ml_decision(
                main.MLDecisionRequest(
                    prediction_event_id=result["prediction_event_id"],
                    chosen_option_id="NOT-IN-PREDICTION",
                    rationale="This option was not presented.",
                    operational_objective="Meet the replenishment requirement.",
                ),
                self.session,
            )
        self.assertIsNone(event.decision)

        record = main.record_ml_decision(
            main.MLDecisionRequest(
                prediction_event_id=result["prediction_event_id"],
                chosen_option_id="SUP-A",
                rationale="Supplier A meets the cost requirement.",
                operational_objective="Replenish filters within the delivery deadline and budget.",
                constraints=["Remain within the available budget."],
                influencing_factors=["Supplier A has the lowest cost."],
                reasoning_steps=["Check the budget.", "Confirm the supplier can meet the order."],
                considered_alternative_codes=["SUP-A", "SUP-B", "SUP-C"],
                confidence=0.8,
            ),
            self.session,
        )
        self.assertEqual(record["chosen_option_id"], "SUP-A")
        self.assertEqual(record["reasoning_context"]["rationale_source"], "human_stated")
        self.assertEqual(
            record["reasoning_context"]["operational_objective"],
            "Replenish filters within the delivery deadline and budget.",
        )
        self.assertEqual(len(record["probabilities"]), 3)
        self.assertEqual(
            record["preference_weights"],
            main.TRAINING_DATA["profiles"][0]["preferences"],
        )
        self.assertIn("model_name", record)
        self.assertEqual(
            record["choice_matches_prediction"],
            result["predicted_choice"] == "SUP-A",
        )
        self.assertEqual(event.decision.rationale, "Supplier A meets the cost requirement.")
        decision_node = self.session.query(ContextNode).filter_by(
            node_type="HumanStatedReason"
        ).one()
        self.assertEqual(decision_node.properties["source"], "human_stated")
        graph = graph_snapshot(self.session, "SCN-001", depth=3)
        self.assertTrue(any(node["type"] == "Objective" for node in graph["nodes"]))
        self.assertTrue(any(
            relationship["type"] == "HAS_REASONING_STEP"
            for relationship in graph["relationships"]
        ))
        self.assertEqual(len(self.session.query(Prediction).filter_by(
            decision_event_id=event.id
        ).all()), 3)
        history = main.get_decision_feedback(self.session)["records"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], record["id"])
        self.assertEqual(history[0]["rationale"], "Supplier A meets the cost requirement.")

        outcome = main.record_decision_outcome(
            record["id"],
            main.OutcomeFeedbackRequest(
                status="fulfilled",
                notes="Replacement shipment arrived within the required window.",
            ),
            self.session,
        )
        self.assertEqual(outcome["outcome"]["status"], "fulfilled")
        self.assertEqual(
            outcome["outcome"]["notes"],
            "Replacement shipment arrived within the required window.",
        )

        with self.assertRaises(main.HTTPException) as duplicate_error:
            main.record_ml_decision(
                main.MLDecisionRequest(
                    prediction_event_id=result["prediction_event_id"],
                    chosen_option_id="SUP-B",
                    rationale="A second choice for the same event.",
                    operational_objective="Meet the delivery requirement.",
                ),
                self.session,
            )
        self.assertEqual(duplicate_error.exception.status_code, 409)

    def test_fastapi_prediction_endpoint_selects_random_forest(self):
        self.mark_seeded_choices_as_observed()
        train_model(
            self.session,
            self.model_path,
            model_name="random_forest",
        )
        with patch(
            "app.ml.predict.MODEL_PATHS",
            {"random_forest": self.model_path},
        ):
            result = main.predict_ml_choice(
                main.MLPredictionRequest(
                    scenario_id="SCN-001",
                    decision_maker_id="DM-01",
                    model_name="random_forest",
                ),
                self.session,
            )

        self.assertEqual(result["model"]["name"], "random_forest")
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertTrue(result["persisted"])
        explanation = self.session.scalar(
            select(ModelExplanation).where(
                ModelExplanation.prediction_event_id == result["prediction_event_id"]
            )
        )
        self.assertIsNotNone(explanation)
        self.assertGreater(len(explanation.rule_paths), 0)
        self.assertGreater(
            self.session.query(ExtractedRule).filter_by(
                prediction_event_id=result["prediction_event_id"]
            ).count(),
            0,
        )
        graph = graph_snapshot(self.session, "SCN-001", depth=3)
        self.assertTrue(any(node["type"] == "ModelExplanation" for node in graph["nodes"]))
        self.assertTrue(any(node["type"] == "ExtractedRule" for node in graph["nodes"]))
        event = self.session.get(DecisionEvent, result["prediction_event_id"])
        self.assertEqual(event.event_context["model_name"], "random_forest")
        self.assertTrue(all(
            item.model_name == "random_forest"
            for item in event.predictions
        ))
        record = main.record_ml_decision(
            main.MLDecisionRequest(
                prediction_event_id=event.id,
                chosen_option_id="SUP-A",
                rationale="Selected Supplier A to meet the budget.",
                operational_objective="Replenish the stock before the delivery deadline.",
            ),
            self.session,
        )
        self.assertEqual(record["reasoning_context"]["rationale_source"], "human_stated")
        self.assertTrue(record["reasoning_snapshot"]["alternatives"])
        linked_rules = self.session.query(ExtractedRule).filter_by(
            prediction_event_id=event.id
        ).all()
        self.assertTrue(linked_rules)
        self.assertTrue(all(rule.decision_id == event.decision.id for rule in linked_rules))
        history_record = main.get_decision_feedback(self.session)["records"][0]
        self.assertIsNotNone(history_record["model_explanation"])
        self.assertTrue(history_record["reasoning_snapshot"]["alternatives"])


if __name__ == "__main__":
    unittest.main()
