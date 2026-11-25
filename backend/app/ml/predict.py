from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import FeatureDataError, extract_feature_row, feature_matrix
from app.ml.explain import explain_random_forest
from app.ml.model_registry import (
    DEFAULT_MODEL_NAME,
    MODEL_ARTIFACT_PATHS,
    ModelName,
)
from app.models import Alternative, Assessment, DecisionMaker, Preference, Situation

MODEL_PATH = MODEL_ARTIFACT_PATHS[DEFAULT_MODEL_NAME]
MODEL_PATHS = MODEL_ARTIFACT_PATHS


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
    model_name: ModelName = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    artifact_path = model_path or (
        MODEL_PATH if model_name == DEFAULT_MODEL_NAME else MODEL_PATHS[model_name]
    )
    artifact = load_model(artifact_path)
    metadata = artifact["metadata"]
    pipeline = artifact["pipeline"]
    if model_name == "random_forest" and metadata.get("training_data_source") not in {
        "observed_human",
        "synthetic",
    }:
        raise ModelNotTrainedError(
            "The experimental Random Forest artifact has an unsupported training-data source."
        )

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
    assessment_by_alternative: dict[str, Assessment] = {}
    for assessment in assessments:
        scores_by_alternative.setdefault(assessment.alternative_id, assessment.criterion_scores)
        assessment_by_alternative.setdefault(assessment.alternative_id, assessment)

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
    exclusion_reasons: dict[str, str] = {}
    for alternative in alternatives:
        if not alternative.is_available:
            exclusion_reasons[alternative.id] = "Alternative is unavailable."
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
                exclusion_reasons[alternative.id] = (
                    f"Capacity ({capacity_value:g}) is below required quantity ({required_value:g})."
                )
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
    alternatives_snapshot = [
        {
            "id": alternative.id,
            "code": alternative.code,
            "name": alternative.name,
            "available": alternative.is_available,
            "eligible": alternative in eligible_alternatives,
            "exclusion_reason": exclusion_reasons.get(alternative.id),
            "attributes": alternative.attributes,
            "criterion_scores": scores_by_alternative[alternative.id],
            "overall_suitability": (
                assessment_by_alternative[alternative.id].overall_score
                if alternative.id in assessment_by_alternative
                else None
            ),
        }
        for alternative in alternatives
    ]
    reasoning_snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "situation": {
            "id": situation.id,
            "title": situation.title,
            "description": situation.description,
            "context_data": situation.context_data,
        },
        "decision_maker": {
            "id": decision_maker.id,
            "name": decision_maker.name,
            "role": decision_maker.role,
        },
        "operational_objective": situation.context_data.get("operational_objective"),
        "preference_weights": preference.weights,
        "alternatives": alternatives_snapshot,
    }
    explanation = None
    if model_name == "random_forest":
        explanation = explain_random_forest(
            pipeline,
            matrix,
            probabilities[0]["alternative_code"],
        )

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
        "reasoning_snapshot": reasoning_snapshot,
        "explanation": explanation,
        "feature_inputs": {
            "preference_weights": preference.weights,
            "alternative_criterion_scores": {
                item["code"]: item["scores"]
                for item in feature_alternatives
            },
        },
        "model": {
            "name": metadata["model_name"],
            "version": metadata["trained_at"],
            "training_data_source": metadata["training_data_source"],
            "training_sample_count": metadata["training_sample_count"],
            "training_scenario_count": metadata["training_scenario_count"],
            "group_split": metadata["group_split"],
        },
        "disclaimer": (
            "The experimental Random Forest was trained on synthetic demonstration choices; "
            "its outputs are not validated predictions of human behavior."
            if model_name == "random_forest"
            and metadata["training_data_source"] == "synthetic"
            else (
                "Choice probabilities are illustrative model estimates trained on synthetic "
                "decisions; they are not validated predictions of real human behavior."
                if metadata["training_data_source"] == "synthetic"
                else (
                    "Experimental Random Forest patterns are based on observed human choices "
                    "but are not validated predictions of future behavior."
                )
            )
        ),
    }
