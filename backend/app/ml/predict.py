from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import FeatureDataError, extract_feature_row, feature_matrix
from app.models import Alternative, Assessment, DecisionMaker, Preference, Situation

MODEL_PATH = Path(__file__).parent / "artifacts" / "choice_model.joblib"


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the model artifact has not been generated yet."""


class ScenarioNotFoundError(LookupError):
    """Raised when a prediction request references an unknown situation."""


class DecisionMakerNotFoundError(LookupError):
    """Raised when a prediction request references an unknown decision-maker."""


class PreferenceNotFoundError(LookupError):
    """Raised when no current preference profile exists for a decision-maker."""


class NoAvailableAlternativesError(ValueError):
    """Raised when no alternative is currently eligible for selection."""


def load_model(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise ModelNotTrainedError(
            f"No trained choice model found at {path}. Run `python -m app.ml.train` from backend first."
        )
    return _load_model_cached(str(path.resolve()), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_model_cached(model_path: str, modified_at_ns: int) -> dict[str, Any]:
    current_modified_at_ns = Path(model_path).stat().st_mtime_ns
    if current_modified_at_ns != modified_at_ns:
        return _load_model_cached(model_path, current_modified_at_ns)
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {"pipeline", "metadata"}.issubset(artifact):
        raise ValueError(f"Invalid choice model artifact: {model_path}")
    return artifact


def predict_choice(
    session: Session,
    scenario_id: str,
    decision_maker_id: str,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    artifact = load_model(model_path or MODEL_PATH)
    metadata = artifact["metadata"]
    pipeline = artifact["pipeline"]

    situation = session.get(Situation, scenario_id)
    if situation is None:
        raise ScenarioNotFoundError(f"Unknown situation: {scenario_id}")
    decision_maker = session.get(DecisionMaker, decision_maker_id)
    if decision_maker is None or not decision_maker.is_active:
        raise DecisionMakerNotFoundError(f"Unknown active decision-maker: {decision_maker_id}")
    preference = session.scalar(
        select(Preference)
        .where(
            Preference.decision_maker_id == decision_maker_id,
            Preference.is_current.is_(True),
        )
        .order_by(Preference.version.desc())
    )
    if preference is None:
        raise PreferenceNotFoundError(
            f"No current preference profile exists for decision-maker {decision_maker_id}."
        )

    alternatives = session.scalars(
        select(Alternative)
        .where(Alternative.situation_id == scenario_id)
        .order_by(Alternative.code)
    ).all()
    if not alternatives:
        raise NoAvailableAlternativesError(
            f"Situation {scenario_id} has no stored alternatives."
        )

    assessments = session.scalars(
        select(Assessment)
        .where(Assessment.situation_id == scenario_id)
        .order_by(Assessment.assessment_version.desc(), Assessment.id.desc())
    ).all()
    scores_by_alternative: dict[str, dict[str, float]] = {}
    for assessment in assessments:
        scores_by_alternative.setdefault(assessment.alternative_id, assessment.criterion_scores)

    criteria = metadata["criteria"]
    alternative_codes = metadata["alternative_codes"]
    feature_alternatives = []
    for alternative in alternatives:
        scores = scores_by_alternative.get(alternative.id)
        if scores is None:
            raise FeatureDataError(
                f"Alternative {alternative.id} has no persisted suitability assessment."
            )
        feature_alternatives.append({
            "code": alternative.code,
            "scores": scores,
            "available": alternative.is_available,
            "attributes": alternative.attributes,
        })

    feature_row = extract_feature_row(
        situation.context_data,
        decision_maker.role,
        preference.weights,
        feature_alternatives,
        criteria,
        alternative_codes,
    )
    matrix = feature_matrix(
        [feature_row],
        metadata["numeric_features"],
        metadata["categorical_features"],
    )
    classifier = pipeline.named_steps["classifier"]
    raw_probabilities = pipeline.predict_proba(matrix)[0]
    probability_by_code = {
        str(code): float(probability)
        for code, probability in zip(classifier.classes_, raw_probabilities)
    }

    required_units = situation.context_data.get("required_units")
    eligible_alternatives = []
    for alternative in alternatives:
        if not alternative.is_available:
            continue
        capacity = alternative.attributes.get("capacity_units")
        if required_units is not None:
            if capacity is None:
                raise FeatureDataError(
                    f"Alternative {alternative.id} is missing capacity_units for eligibility filtering."
                )
            try:
                capacity_value = float(capacity)
                required_value = float(required_units)
            except (TypeError, ValueError) as error:
                raise FeatureDataError(
                    f"Alternative {alternative.id} has invalid capacity or required quantity."
                ) from error
            if not math.isfinite(capacity_value) or not math.isfinite(required_value):
                raise FeatureDataError(
                    f"Alternative {alternative.id} has non-finite capacity or required quantity."
                )
            if capacity_value < required_value:
                continue
        eligible_alternatives.append(alternative)
    if not eligible_alternatives:
        raise NoAvailableAlternativesError(
            f"Situation {scenario_id} has no available alternatives with sufficient capacity."
        )

    total_probability = sum(
        probability_by_code.get(alternative.code, 0.0)
        for alternative in eligible_alternatives
    )
    if total_probability <= 0:
        raise FeatureDataError("The trained model assigned zero probability to every available alternative.")

    probabilities = [
        {
            "alternative_id": alternative.id,
            "alternative_code": alternative.code,
            "alternative_name": alternative.name,
            "probability": probability_by_code.get(alternative.code, 0.0) / total_probability,
            "probability_percent": round(
                probability_by_code.get(alternative.code, 0.0) / total_probability * 100,
                2,
            ),
            "criterion_scores": scores_by_alternative[alternative.id],
        }
        for alternative in eligible_alternatives
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "scenario_id": situation.id,
        "scenario_title": situation.title,
        "decision_maker": {
            "id": decision_maker.id,
            "name": decision_maker.name,
            "role": decision_maker.role,
        },
        "predicted_choice": probabilities[0]["alternative_code"],
        "probabilities": probabilities,
        "feature_inputs": {
            "preference_weights": preference.weights,
            "alternative_criterion_scores": {
                item["code"]: item["scores"]
                for item in feature_alternatives
            },
        },
        "model": {
            "name": metadata["model_name"],
            "training_data_source": metadata["training_data_source"],
            "training_sample_count": metadata["training_sample_count"],
            "training_scenario_count": metadata["training_scenario_count"],
            "group_split": metadata["group_split"],
        },
        "disclaimer": (
            "Choice probabilities are illustrative model estimates trained on synthetic "
            "decisions; they are not validated predictions of real human behavior."
        ),
    }
