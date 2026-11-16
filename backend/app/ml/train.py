from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import extract_feature_row, feature_columns, feature_matrix
from app.models import (
    Alternative,
    Assessment,
    Decision,
    DecisionEvent,
    DecisionMaker,
    Preference,
    Situation,
)

MODEL_PATH = Path(__file__).parent / "artifacts" / "choice_model.joblib"


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


def load_training_dataset(session: Session) -> TrainingDataset:
    situations = {
        row.id: row
        for row in session.scalars(
            select(Situation).where(Situation.data_source == "synthetic")
        ).all()
    }
    decision_makers = {
        row.id: row
        for row in session.scalars(
            select(DecisionMaker).where(DecisionMaker.data_source == "synthetic")
        ).all()
    }
    preferences = {
        row.id: row
        for row in session.scalars(
            select(Preference).where(Preference.data_source == "synthetic")
        ).all()
    }
    events = {
        row.id: row
        for row in session.scalars(
            select(DecisionEvent).where(DecisionEvent.data_source == "synthetic")
        ).all()
    }
    alternatives = session.scalars(
        select(Alternative)
        .where(Alternative.data_source == "synthetic")
        .order_by(Alternative.situation_id, Alternative.code)
    ).all()
    assessments = session.scalars(
        select(Assessment)
        .where(
            Assessment.data_source == "synthetic",
            Assessment.assessment_version == 1,
        )
        .order_by(Assessment.alternative_id)
    ).all()
    decisions = session.scalars(
        select(Decision).where(
            Decision.data_source == "synthetic",
            Decision.label_source == "synthetic",
        )
    ).all()

    alternatives_by_situation: dict[str, list[Alternative]] = {}
    for alternative in alternatives:
        alternatives_by_situation.setdefault(alternative.situation_id, []).append(alternative)
    scores_by_alternative = {
        assessment.alternative_id: assessment.criterion_scores
        for assessment in assessments
    }
    alternative_codes = tuple(sorted({alternative.code for alternative in alternatives}))
    if len(alternative_codes) != 3:
        raise TrainingDataError("Training data must define exactly three alternative classes.")
    if not scores_by_alternative:
        raise TrainingDataError("No synthetic suitability assessments are stored.")
    criteria = tuple(sorted(next(iter(scores_by_alternative.values()))))
    if not criteria:
        raise TrainingDataError("Synthetic assessments do not contain criterion scores.")

    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    groups: list[str] = []
    for decision in decisions:
        event = events.get(decision.decision_event_id)
        if event is None:
            raise TrainingDataError(
                f"Synthetic decision {decision.id} has no synthetic decision event."
            )
        situation = situations.get(event.situation_id)
        decision_maker = decision_makers.get(event.decision_maker_id)
        preference = preferences.get(event.preference_id) if event.preference_id is not None else None
        if situation is None or decision_maker is None or preference is None:
            raise TrainingDataError(
                f"Synthetic decision event {event.id} is missing its situation, decision-maker, or preference."
            )

        scenario_alternatives = alternatives_by_situation.get(situation.id, [])
        if len(scenario_alternatives) != len(alternative_codes) or {
            alternative.code for alternative in scenario_alternatives
        } != set(alternative_codes):
            raise TrainingDataError(
                f"Situation {situation.id} does not have the same three alternatives as the training set."
            )

        feature_alternatives = []
        for alternative in scenario_alternatives:
            criterion_scores = scores_by_alternative.get(alternative.id)
            if criterion_scores is None:
                raise TrainingDataError(
                    f"Alternative {alternative.id} has no synthetic suitability assessment."
                )
            feature_alternatives.append({
                "code": alternative.code,
                "scores": criterion_scores,
                "available": alternative.is_available,
                "attributes": alternative.attributes,
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
            situation.context_data,
            decision_maker.role,
            preference.weights,
            feature_alternatives,
            criteria,
            alternative_codes,
        ))
        labels.append(chosen_alternative.code)
        groups.append(situation.id)

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
    )


def _build_pipeline(dataset: TrainingDataset) -> Pipeline:
    numeric_indices = list(range(len(dataset.numeric_features)))
    categorical_indices = list(range(
        len(dataset.numeric_features),
        len(dataset.numeric_features) + len(dataset.categorical_features),
    ))
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_indices),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=2000)),
    ])


def train_model(session: Session, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    dataset = load_training_dataset(session)
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

    evaluation_model = _build_pipeline(dataset)
    evaluation_model.fit(feature_rows[train_indices], labels[train_indices])
    holdout_accuracy = float(evaluation_model.score(
        feature_rows[test_indices],
        labels[test_indices],
    ))

    model = _build_pipeline(dataset)
    model.fit(feature_rows, labels)
    metadata = {
        "model_name": "multinomial_logistic_regression",
        "training_data_source": "synthetic",
        "training_sample_count": len(dataset.rows),
        "training_scenario_count": len(set(dataset.groups)),
        "criteria": list(dataset.criteria),
        "alternative_codes": list(dataset.alternative_codes),
        "numeric_features": list(dataset.numeric_features),
        "categorical_features": list(dataset.categorical_features),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "group_split": {
            "strategy": "GroupShuffleSplit",
            "test_size": 0.25,
            "random_state": 42,
            "training_scenario_ids": training_scenarios,
            "holdout_scenario_ids": holdout_scenarios,
            "training_sample_count": int(len(train_indices)),
            "holdout_sample_count": int(len(test_indices)),
            "holdout_accuracy": holdout_accuracy,
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump({"pipeline": model, "metadata": metadata}, temporary_path)
    os.replace(temporary_path, model_path)
    return metadata


def main() -> None:
    from app.database import SessionLocal
    from app.init_db import create_tables

    create_tables()
    with SessionLocal() as session:
        metadata = train_model(session)
    print(json.dumps(metadata, indent=2))
    print(f"Model artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()
