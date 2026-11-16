import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import main
from app.context_graph import seed_context_graph
from app.database import Base
from app.ml.predict import _load_model_cached, load_model, predict_choice
from app.ml.train import load_training_dataset, train_model


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

    def test_feature_rows_include_preferences_and_all_alternative_criterion_scores(self):
        dataset = load_training_dataset(self.session)

        self.assertIn("preference_cost", dataset.numeric_features)
        self.assertIn("preference_reliability", dataset.numeric_features)
        self.assertIn("alternative_SUP-A_cost", dataset.numeric_features)
        self.assertIn("alternative_SUP-C_risk", dataset.numeric_features)
        self.assertIn("decision_maker_role", dataset.categorical_features)
        self.assertEqual(len(dataset.rows), 32)
        self.assertEqual(len(set(dataset.groups)), 8)

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
        self.assertEqual(result["model"]["training_sample_count"], 32)
        self.assertEqual(len(result["probabilities"]), 3)


if __name__ == "__main__":
    unittest.main()
