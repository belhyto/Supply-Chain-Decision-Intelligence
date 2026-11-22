from pathlib import Path
from typing import Literal

ModelName = Literal["multinomial_logistic_regression", "random_forest"]

DEFAULT_MODEL_NAME: ModelName = "multinomial_logistic_regression"
MODEL_NAMES: tuple[ModelName, ...] = (DEFAULT_MODEL_NAME, "random_forest")
MODEL_ARTIFACT_PATHS = {
    "multinomial_logistic_regression": Path(__file__).parent / "artifacts" / "choice_model.joblib",
    "random_forest": Path(__file__).parent / "artifacts" / "random_forest_model.joblib",
}
