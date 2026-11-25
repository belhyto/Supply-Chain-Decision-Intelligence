from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import extract_feature_row, feature_columns, feature_matrix
from app.ml.model_registry import (
    DEFAULT_MODEL_NAME,
    MODEL_ARTIFACT_PATHS,
    MODEL_NAMES,
    ModelName,
)
from app.models import (
    Alternative,
    Assessment,
    Decision,
    DecisionEvent,
    DecisionMaker,
    Preference,
    Situation,
)

MODEL_PATH = MODEL_ARTIFACT_PATHS[DEFAULT_MODEL_NAME]


class TrainingDataError(ValueError):
    """Raised when the persisted synthetic records cannot train the classifier."""


@dataclass(frozen=True)
class TrainingDataset:
    rows: list[dict[str, Any]]
    labels: list[str]
    groups: list[str]
    criteria: tuple[str, ...]
    alternative_codes: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    decision_ids: list[int]
    decision_maker_ids: list[str]
    decision_times: list[str]


def load_training_dataset(
    session: Session,
    decision_source: str = "synthetic",
) -> TrainingDataset:
    decisions = session.scalars(
        select(Decision).where(
            Decision.data_source == decision_source,
            Decision.label_source == decision_source,
        )
    ).all()
    if not decisions:
        raise TrainingDataError(f"No {decision_source} human decision records are available.")

    event_ids = {decision.decision_event_id for decision in decisions}
    events = {
        row.id: row
        for row in session.scalars(
            select(DecisionEvent).where(DecisionEvent.id.in_(event_ids))
        ).all()
    }
    situation_ids = {event.situation_id for event in events.values()}
    decision_maker_ids = {event.decision_maker_id for event in events.values()}
    preference_ids = {
        event.preference_id for event in events.values()
        if event.preference_id is not None
    }
    situations = {
        row.id: row
        for row in session.scalars(
            select(Situation).where(Situation.id.in_(situation_ids))
        ).all()
    }
    decision_makers = {
        row.id: row
        for row in session.scalars(
            select(DecisionMaker).where(DecisionMaker.id.in_(decision_maker_ids))
        ).all()
    }
    preferences = {
        row.id: row
        for row in session.scalars(
            select(Preference).where(Preference.id.in_(preference_ids))
        ).all()
    }
    alternatives_query = select(Alternative).where(Alternative.situation_id.in_(situation_ids))
    assessments_query = select(Assessment).where(Assessment.situation_id.in_(situation_ids))
    if decision_source == "synthetic":
        alternatives_query = alternatives_query.where(Alternative.data_source == "synthetic")
        assessments_query = assessments_query.where(
            Assessment.data_source == "synthetic",
            Assessment.assessment_version == 1,
        )
    alternatives = session.scalars(
        alternatives_query.order_by(Alternative.situation_id, Alternative.code)
    ).all()
    assessments = session.scalars(
        assessments_query.order_by(
            Assessment.alternative_id,
            Assessment.assessment_version.desc(),
            Assessment.id.desc(),
        )
    ).all()

    alternatives_by_situation: dict[str, list[Alternative]] = {}
    for alternative in alternatives:
        alternatives_by_situation.setdefault(alternative.situation_id, []).append(alternative)
    scores_by_alternative = {}
    for assessment in assessments:
        scores_by_alternative.setdefault(assessment.alternative_id, assessment.criterion_scores)
    alternative_codes = tuple(sorted({alternative.code for alternative in alternatives}))
    if len(alternative_codes) != 3:
        raise TrainingDataError("The training records must define exactly three alternative classes.")
    if not scores_by_alternative:
        raise TrainingDataError("No suitability assessments are stored for the selected decision records.")
    criteria = tuple(sorted(next(iter(scores_by_alternative.values()))))
    if not criteria:
        raise TrainingDataError("Suitability assessments do not contain criterion scores.")

    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    groups: list[str] = []
    decision_ids: list[int] = []
    decision_maker_ids: list[str] = []
    decision_times: list[str] = []
    for decision in decisions:
        event = events.get(decision.decision_event_id)
        if event is None:
            raise TrainingDataError(
                f"Decision {decision.id} has no linked decision event."
            )
        situation = situations.get(event.situation_id)
        decision_maker = decision_makers.get(event.decision_maker_id)
        preference = preferences.get(event.preference_id) if event.preference_id is not None else None
        if situation is None or decision_maker is None or preference is None:
            raise TrainingDataError(
                f"Decision event {event.id} is missing its situation, decision-maker, or preference."
            )

        scenario_alternatives = alternatives_by_situation.get(situation.id, [])
        if len(scenario_alternatives) != len(alternative_codes) or {
            alternative.code for alternative in scenario_alternatives
        } != set(alternative_codes):
            raise TrainingDataError(
                f"Situation {situation.id} does not have the same three alternatives as the training set."
            )

        feature_alternatives = []
        decision_snapshot = event.event_context.get("reasoning_snapshot", {})
        scenario_snapshot = decision_snapshot.get("situation", {})
        human_reasoning = event.event_context.get("human_reasoning", {})
        role_at_decision = decision_snapshot.get("decision_maker", {}).get(
            "role",
            decision_maker.role,
        )
        preferences_at_decision = human_reasoning.get(
            "preference_weights",
            decision_snapshot.get("preference_weights", preference.weights),
        )
        snapshot_alternatives = {
            item["code"]: item
            for item in decision_snapshot.get("alternatives", [])
        }
        for alternative in scenario_alternatives:
            snapshot_alternative = snapshot_alternatives.get(alternative.code, {})
            criterion_scores = snapshot_alternative.get(
                "criterion_scores",
                scores_by_alternative.get(alternative.id),
            )
            if criterion_scores is None:
                raise TrainingDataError(
                    f"Alternative {alternative.id} has no suitability assessment."
                )
            feature_alternatives.append({
                "code": alternative.code,
                "scores": criterion_scores,
                "available": snapshot_alternative.get("available", alternative.is_available),
                "attributes": snapshot_alternative.get("attributes", alternative.attributes),
            })

        chosen_alternative = next(
            (
                alternative for alternative in scenario_alternatives
                if alternative.id == decision.chosen_alternative_id
            ),
            None,
        )
        if chosen_alternative is None:
            raise TrainingDataError(
                f"Synthetic decision {decision.id} selects an alternative outside its situation."
            )

        rows.append(extract_feature_row(
            scenario_snapshot.get("context_data", situation.context_data),
            role_at_decision,
            preferences_at_decision,
            feature_alternatives,
            criteria,
            alternative_codes,
        ))
        labels.append(chosen_alternative.code)
        groups.append(situation.id)
        decision_ids.append(decision.id)
        decision_maker_ids.append(event.decision_maker_id)
        decision_times.append(decision.decided_at.isoformat())

    if not rows:
        raise TrainingDataError("No synthetic decisions are available for model training.")
    if set(labels) != set(alternative_codes):
        raise TrainingDataError("Every alternative must appear as a recorded training choice.")

    numeric, categorical = feature_columns(criteria, alternative_codes)
    return TrainingDataset(
        rows=rows,
        labels=labels,
        groups=groups,
        criteria=criteria,
        alternative_codes=alternative_codes,
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        decision_ids=decision_ids,
        decision_maker_ids=decision_maker_ids,
        decision_times=decision_times,
    )


def _build_pipeline(dataset: TrainingDataset, model_name: ModelName) -> Pipeline:
    numeric_indices = list(range(len(dataset.numeric_features)))
    categorical_indices = list(range(
        len(dataset.numeric_features),
        len(dataset.numeric_features) + len(dataset.categorical_features),
    ))
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if model_name == "multinomial_logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        classifier = LogisticRegression(solver="lbfgs", max_iter=2000)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    numeric_pipeline = Pipeline(steps=numeric_steps)
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_indices),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model(
    session: Session,
    model_path: Path | None = None,
    model_name: ModelName = DEFAULT_MODEL_NAME,
    training_data_source: Literal["synthetic", "observed_human"] | None = None,
) -> dict[str, Any]:
    if model_name not in MODEL_ARTIFACT_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_path = model_path or MODEL_ARTIFACT_PATHS[model_name]
    decision_source = training_data_source or (
        "observed_human" if model_name == "random_forest" else "synthetic"
    )
    if decision_source not in {"synthetic", "observed_human"}:
        raise ValueError("Training data source must be synthetic or observed_human.")
    if model_name != "random_forest" and decision_source != "synthetic":
        raise ValueError("Multinomial logistic regression uses the synthetic fixture.")
    dataset = load_training_dataset(session, decision_source=decision_source)
    feature_rows = feature_matrix(
        dataset.rows,
        dataset.numeric_features,
        dataset.categorical_features,
    )
    labels = np.asarray(dataset.labels)
    groups = np.asarray(dataset.groups)
    if len(set(groups)) < 4:
        raise TrainingDataError("Group holdout requires at least four distinct scenarios.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, test_indices = next(splitter.split(feature_rows, labels, groups))
    training_scenarios = sorted(set(groups[train_indices]))
    holdout_scenarios = sorted(set(groups[test_indices]))
    if set(training_scenarios) & set(holdout_scenarios):
        raise TrainingDataError("Group split placed a scenario in both training and test sets.")
    if set(labels[train_indices]) != set(dataset.alternative_codes):
        raise TrainingDataError(
            "The grouped training partition must contain choices for every alternative."
        )

    evaluation_model = _build_pipeline(dataset, model_name)
    evaluation_model.fit(feature_rows[train_indices], labels[train_indices])
    holdout_predictions = evaluation_model.predict(feature_rows[test_indices])
    holdout_probabilities = evaluation_model.predict_proba(feature_rows[test_indices])
    classifier_classes = evaluation_model.named_steps["classifier"].classes_
    holdout_actual = labels[test_indices]
    holdout_one_hot = np.asarray([
        [float(label == class_name) for class_name in classifier_classes]
        for label in holdout_actual
    ])
    holdout_metrics = {
        "accuracy": float(accuracy_score(holdout_actual, holdout_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(holdout_actual, holdout_predictions)
        ),
        "macro_f1": float(
            f1_score(
                holdout_actual,
                holdout_predictions,
                labels=dataset.alternative_codes,
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_brier_score": float(
            np.mean(np.sum((holdout_probabilities - holdout_one_hot) ** 2, axis=1))
        ),
        "per_class": classification_report(
            holdout_actual,
            holdout_predictions,
            labels=dataset.alternative_codes,
            output_dict=True,
            zero_division=0,
        ),
    }

    model = _build_pipeline(dataset, model_name)
    model.fit(feature_rows, labels)
    trained_at = datetime.now(timezone.utc).isoformat()
    dataset_digest = hashlib.sha256(json.dumps(
        {
            "rows": [
                {
                    key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
                for row in dataset.rows
            ],
            "labels": dataset.labels,
            "groups": dataset.groups,
            "decision_ids": dataset.decision_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    metadata = {
        "model_name": model_name,
        "model_version": f"{model_name}:{trained_at}",
        "training_data_source": decision_source,
        "dataset_version": f"sha256:{dataset_digest}",
        "training_sample_count": len(dataset.rows),
        "training_scenario_count": len(set(dataset.groups)),
        "criteria": list(dataset.criteria),
        "alternative_codes": list(dataset.alternative_codes),
        "numeric_features": list(dataset.numeric_features),
        "categorical_features": list(dataset.categorical_features),
        "trained_at": trained_at,
        "group_split": {
            "strategy": "GroupShuffleSplit",
            "test_size": 0.25,
            "random_state": 42,
            "training_scenario_ids": training_scenarios,
            "holdout_scenario_ids": holdout_scenarios,
            "training_sample_count": int(len(train_indices)),
            "holdout_sample_count": int(len(test_indices)),
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_metrics": holdout_metrics,
        },
    }
    if model_name == "random_forest":
        metadata["model_parameters"] = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42,
        }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump({"pipeline": model, "metadata": metadata}, temporary_path)
    os.replace(temporary_path, model_path)
    return metadata


def main() -> None:
    from app.database import SessionLocal
    from app.init_db import create_tables

    parser = argparse.ArgumentParser(description="Train the decision-choice models.")
    parser.add_argument(
        "--random-forest-source",
        choices=("observed_human", "synthetic"),
        default="observed_human",
        help=(
            "Training source for the experimental Random Forest. Synthetic is for "
            "demonstrations only; observed_human remains the rule-memory source."
        ),
    )
    arguments = parser.parse_args()
    create_tables()
    with SessionLocal() as session:
        metadata = {}
        for model_name in MODEL_NAMES:
            try:
                metadata[model_name] = train_model(
                    session,
                    model_name=model_name,
                    training_data_source=(
                        arguments.random_forest_source
                        if model_name == "random_forest"
                        else "synthetic"
                    ),
                )
            except TrainingDataError as error:
                if model_name != "random_forest":
                    raise
                metadata[model_name] = {
                    "status": "not_trained",
                    "reason": str(error),
                }
                print(f"Random Forest remains experimental: {error}", file=sys.stderr)
    print(json.dumps(metadata, indent=2))
    for model_name in MODEL_NAMES:
        print(f"{model_name} artifact: {MODEL_ARTIFACT_PATHS[model_name]}")


if __name__ == "__main__":
    main()
