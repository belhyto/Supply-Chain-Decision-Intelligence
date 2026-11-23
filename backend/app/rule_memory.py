from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ml.features import extract_feature_row, feature_matrix
from app.ml.train import TrainingDataError, load_training_dataset, train_model
from app.models import (
    Alternative,
    Assessment,
    Decision,
    DecisionMaker,
    DecisionRule,
    Preference,
    RuleApplication,
    RuleEvidence,
    RuleExtraction,
    RuleFeedback,
    RuleReview,
    Situation,
)


def _encoded_condition(
    transformed_name: str,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    categorical_start_index: int,
    operator: str,
    threshold: float,
) -> dict[str, Any]:
    if transformed_name.startswith("numeric__"):
        encoded_name = transformed_name.removeprefix("numeric__")
        feature_name = encoded_name
        if encoded_name.startswith("x") and encoded_name[1:].isdigit():
            numeric_index = int(encoded_name[1:])
            if numeric_index < len(numeric_features):
                feature_name = numeric_features[numeric_index]
        return {
            "tree_feature_name": transformed_name,
            "kind": "numeric",
            "feature_name": feature_name,
            "condition_operator": operator,
            "threshold": round(float(threshold), 6),
        }

    encoded_name = transformed_name.removeprefix("categorical__")
    for column_index, column in enumerate(categorical_features):
        prefix = f"x{categorical_start_index + column_index}_"
        if encoded_name.startswith(prefix):
            return {
                "tree_feature_name": transformed_name,
                "kind": "categorical",
                "feature_name": column,
                "category": encoded_name[len(prefix):],
                "condition_operator": operator,
                "threshold": round(float(threshold), 6),
            }
    raise ValueError(f"Unable to decode Random Forest feature: {transformed_name}")


def _tree_paths(
    tree,
    feature_names: list[str],
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    categorical_start_index: int,
):
    pending = [(0, [])]
    while pending:
        node_id, path = pending.pop()
        if path:
            yield node_id, path
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == right:
            continue

        feature_index = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        feature_name = feature_names[feature_index]
        left_condition = _encoded_condition(
            feature_name,
            numeric_features,
            categorical_features,
            categorical_start_index,
            "<=",
            threshold,
        )
        right_condition = _encoded_condition(
            feature_name,
            numeric_features,
            categorical_features,
            categorical_start_index,
            ">",
            threshold,
        )
        pending.append((right, [*path, right_condition]))
        pending.append((left, [*path, left_condition]))


def _condition_signature(conditions: list[dict[str, Any]], choice: str) -> str:
    return json.dumps(
        {"choice": choice, "conditions": conditions},
        sort_keys=True,
        separators=(",", ":"),
    )


def _dataset_version(dataset) -> str:
    return hashlib.sha256(json.dumps(
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


def _rule_name(choice: str, conditions: list[dict[str, Any]]) -> str:
    if conditions:
        first = conditions[0]
        feature = first.get("feature_name", "context").replace("_", " ")
        return f"Observed {choice} pattern under {feature}"
    return f"Observed pattern for {choice}"


def _condition_description(conditions: list[dict[str, Any]]) -> str:
    descriptions = []
    for condition in conditions:
        value = (
            repr(condition["category"])
            if condition["kind"] == "categorical"
            else f"{condition['condition_operator']} {condition['threshold']:g}"
        )
        if condition["kind"] == "categorical":
            relation = "is" if condition["condition_operator"] == ">" else "is not"
            descriptions.append(
                f"{condition['feature_name'].replace('_', ' ')} {relation} {value}"
            )
        else:
            descriptions.append(
                f"{condition['feature_name'].replace('_', ' ')} "
                f"{condition['condition_operator']} {condition['threshold']:g}"
            )
    return " AND ".join(descriptions)


def extract_candidate_rules(
    session: Session,
    min_matching_decisions: int = 3,
    min_scenarios: int = 2,
    min_decision_makers: int = 2,
    max_rules: int = 25,
) -> dict[str, Any]:
    if min_matching_decisions < 1 or min_scenarios < 1 or min_decision_makers < 1:
        raise ValueError("Evidence thresholds must be positive integers.")
    if max_rules < 1 or max_rules > 100:
        raise ValueError("max_rules must be between 1 and 100.")

    dataset = load_training_dataset(session, decision_source="observed_human")
    with tempfile.TemporaryDirectory(prefix="decision-rule-extraction-") as directory:
        artifact_path = Path(directory) / "random_forest.joblib"
        metadata = train_model(
            session,
            model_path=artifact_path,
            model_name="random_forest",
        )
        artifact = joblib.load(artifact_path)

    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    matrix = feature_matrix(
        dataset.rows,
        dataset.numeric_features,
        dataset.categorical_features,
    )
    transformed = preprocessor.transform(matrix)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    grouped_paths: dict[str, dict[str, Any]] = {}
    for tree_id, estimator in enumerate(classifier.estimators_):
        node_samples = estimator.decision_path(transformed)
        for node_id, conditions in _tree_paths(
            estimator.tree_,
            feature_names,
            dataset.numeric_features,
            dataset.categorical_features,
            len(dataset.numeric_features),
        ):
            matched_indices = node_samples[:, node_id].nonzero()[0].tolist()
            if not matched_indices:
                continue
            tree_prediction = int(estimator.predict(transformed[matched_indices[:1]])[0])
            choice = str(classifier.classes_[tree_prediction])
            signature = _condition_signature(conditions, choice)
            grouped = grouped_paths.setdefault(signature, {
                "conditions": conditions,
                "choice": choice,
                "indices": set(),
                "tree_ids": set(),
            })
            grouped["indices"].update(matched_indices)
            grouped["tree_ids"].add(tree_id)

    candidates = []
    for signature, pattern in grouped_paths.items():
        indices = sorted(pattern["indices"])
        if len(indices) < min_matching_decisions:
            continue
        matching_labels = [dataset.labels[index] for index in indices]
        choice = pattern["choice"]
        supporting_indices = [
            index for index in indices
            if dataset.labels[index] == choice
        ]
        contradicting_indices = [
            index for index in indices
            if dataset.labels[index] != choice
        ]
        scenario_count = len({dataset.groups[index] for index in indices})
        decision_maker_count = len({
            dataset.decision_maker_ids[index]
            for index in indices
        })
        if scenario_count < min_scenarios or decision_maker_count < min_decision_makers:
            continue

        conditions = pattern["conditions"]
        candidates.append({
            "signature": signature,
            "conditions": conditions,
            "choice": choice,
            "indices": indices,
            "supporting_indices": supporting_indices,
            "contradicting_indices": contradicting_indices,
            "matching_count": len(matching_labels),
            "supporting_count": len(supporting_indices),
            "contradictory_count": len(contradicting_indices),
            "support_proportion": len(supporting_indices) / len(matching_labels),
            "scenario_count": scenario_count,
            "decision_maker_count": decision_maker_count,
            "tree_ids": sorted(pattern["tree_ids"]),
        })

    candidates.sort(
        key=lambda item: (
            -item["supporting_count"],
            -item["matching_count"],
            -item["support_proportion"],
            item["signature"],
        )
    )
    candidates = candidates[:max_rules]
    version = metadata["model_version"]
    extraction_id = str(uuid4())
    extraction_parameters = {
        "minimum_matching_decisions": min_matching_decisions,
        "minimum_scenarios": min_scenarios,
        "minimum_decision_makers": min_decision_makers,
        "maximum_rules": max_rules,
        "numeric_features": list(dataset.numeric_features),
        "categorical_features": list(dataset.categorical_features),
        "criteria": list(dataset.criteria),
        "alternative_codes": list(dataset.alternative_codes),
        "feature_names": feature_names,
        "categorical_values": {
            feature: sorted({
                str(row[feature])
                for row in dataset.rows
            })
            for feature in dataset.categorical_features
        },
        "numeric_medians": {
            feature: float(value)
            for feature, value in zip(
                dataset.numeric_features,
                preprocessor.named_transformers_["numeric"]
                .named_steps["imputer"].statistics_,
            )
        },
    }
    extraction = RuleExtraction(
        id=extraction_id,
        model_name="random_forest",
        model_version=version,
        dataset_version=metadata["dataset_version"],
        extraction_method="grouped_random_forest_leaf_paths",
        extraction_parameters=extraction_parameters,
        status="completed",
        data_source="model",
    )
    session.add(extraction)
    session.flush()

    rules = []
    decisions_by_id = {
        decision.id: decision
        for decision in session.scalars(
            select(Decision).where(
                Decision.id.in_(dataset.decision_ids),
                Decision.data_source == "observed_human",
                Decision.label_source == "observed_human",
            )
        ).all()
    }
    if len(decisions_by_id) != len(dataset.decision_ids):
        raise TrainingDataError("Observed decisions changed during rule extraction.")

    for candidate in candidates:
        signature = candidate["signature"]
        version_group = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:64]
        previous = session.scalar(
            select(DecisionRule)
            .where(DecisionRule.version_group == version_group)
            .order_by(DecisionRule.version.desc())
            .limit(1)
        )
        rule = DecisionRule(
            id=str(uuid4()),
            rule_name=_rule_name(candidate["choice"], candidate["conditions"]),
            description=(
                f"Observed pattern: decisions matching "
                f"{_condition_description(candidate['conditions']) or 'the available context'} "
                f"selected {candidate['choice']} in "
                f"{candidate['supporting_count']} of {candidate['matching_count']} matching records. "
                "This is historical evidence, not a recommendation."
            ),
            conditions_json=candidate["conditions"],
            observed_pattern={
                "choice": candidate["choice"],
                "matching_decisions": candidate["matching_count"],
                "supporting_decisions": candidate["supporting_count"],
                "contradictory_decisions": candidate["contradictory_count"],
                "support_proportion": candidate["support_proportion"],
                "distinct_scenarios": candidate["scenario_count"],
                "distinct_decision_makers": candidate["decision_maker_count"],
                "supporting_tree_count": len(candidate["tree_ids"]),
                "tree_ids": candidate["tree_ids"],
                "first_observed_at": min(
                    dataset.decision_times[index]
                    for index in candidate["indices"]
                ),
                "last_observed_at": max(
                    dataset.decision_times[index]
                    for index in candidate["indices"]
                ),
                "quality_checks": {
                    "minimum_evidence_met": candidate["matching_count"] >= min_matching_decisions,
                    "scenario_representation_met": candidate["scenario_count"] >= min_scenarios,
                    "decision_maker_representation_met": (
                        candidate["decision_maker_count"] >= min_decision_makers
                    ),
                },
            },
            status="PENDING_REVIEW",
            version=(previous.version + 1) if previous else 1,
            version_group=version_group,
            extraction_id=extraction_id,
            supersedes_rule_id=previous.id if previous else None,
            data_source="model",
        )
        session.add(rule)
        session.flush()
        for index in candidate["indices"]:
            decision = decisions_by_id[dataset.decision_ids[index]]
            evidence_type = (
                "supporting"
                if dataset.labels[index] == candidate["choice"]
                else "contradictory"
            )
            session.add(RuleEvidence(
                id=str(uuid4()),
                rule_id=rule.id,
                decision_id=decision.id,
                evidence_type=evidence_type,
                evidence_description=(
                    f"Decision {decision.confidence_note or decision.id} selected "
                    f"{dataset.labels[index]}; extracted pattern observes "
                    f"{candidate['choice']}."
                ),
                data_source="observed_human",
            ))
        rules.append(rule)

    session.flush()
    return {
        "extraction_id": extraction.id,
        "model_name": extraction.model_name,
        "model_version": extraction.model_version,
        "dataset_version": extraction.dataset_version,
        "status": extraction.status,
        "candidate_count": len(rules),
        "candidate_rule_ids": [rule.id for rule in rules],
        "quality_thresholds": extraction_parameters,
    }


def evaluate_conditions(
    conditions: list[dict[str, Any]],
    feature_row: dict[str, Any],
    numeric_medians: dict[str, float],
) -> bool:
    for condition in conditions:
        if condition["kind"] == "categorical":
            observed_value = float(
                feature_row.get(condition["feature_name"])
                == condition["category"]
            )
        else:
            observed_value = feature_row.get(condition["feature_name"])
            if observed_value is None or (
                isinstance(observed_value, float) and math.isnan(observed_value)
            ):
                observed_value = numeric_medians.get(condition["feature_name"])
            if observed_value is None:
                return False
            observed_value = float(observed_value)
        threshold = float(condition["threshold"])
        if condition["condition_operator"] == "<=":
            matches = observed_value <= threshold
        else:
            matches = observed_value > threshold
        if not matches:
            return False
    return True


def validate_conditions(
    conditions: list[dict[str, Any]],
    extraction_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    if not conditions or len(conditions) > 40:
        raise ValueError("A rule must contain between 1 and 40 conditions.")
    numeric_features = set(extraction_parameters["numeric_features"])
    categorical_features = set(extraction_parameters["categorical_features"])
    categorical_values = extraction_parameters.get("categorical_values", {})
    validated = []
    for condition in conditions:
        kind = condition.get("kind")
        feature_name = condition.get("feature_name")
        operator = condition.get("condition_operator")
        threshold = condition.get("threshold")
        if kind not in {"numeric", "categorical"}:
            raise ValueError("Rule condition kind must be numeric or categorical.")
        expected = numeric_features if kind == "numeric" else categorical_features
        if feature_name not in expected or operator not in {"<=", ">"}:
            raise ValueError("Rule condition feature or operator is not in the extraction schema.")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError("Rule condition thresholds must be finite numbers.")
        normalized = {
            "kind": kind,
            "feature_name": feature_name,
            "condition_operator": operator,
            "threshold": float(threshold),
        }
        if kind == "categorical":
            category = condition.get("category")
            if category not in categorical_values.get(feature_name, []):
                raise ValueError(
                    f"Rule condition category is unknown for {feature_name}."
                )
            normalized["category"] = category
        validated.append(normalized)
    return validated


def replace_rule_conditions(
    session: Session,
    rule: DecisionRule,
    conditions: list[dict[str, Any]],
) -> None:
    if rule.published_at is not None:
        raise ValueError("Published rule versions are immutable.")
    extraction = rule.extraction
    normalized = validate_conditions(conditions, extraction.extraction_parameters)
    dataset = load_training_dataset(session, decision_source="observed_human")
    choice = rule.observed_pattern["choice"]
    matched_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if evaluate_conditions(
            normalized,
            row,
            extraction.extraction_parameters.get("numeric_medians", {}),
        )
    ]
    if not matched_indices:
        raise ValueError("Edited conditions match no observed human decision records.")

    for evidence in list(rule.evidences):
        session.delete(evidence)
    supporting_indices = [
        index for index in matched_indices
        if dataset.labels[index] == choice
    ]
    contradicting_indices = [
        index for index in matched_indices
        if dataset.labels[index] != choice
    ]
    distinct_scenarios = len({
        dataset.groups[index] for index in matched_indices
    })
    distinct_decision_makers = len({
        dataset.decision_maker_ids[index] for index in matched_indices
    })
    thresholds = extraction.extraction_parameters
    rule.conditions_json = normalized
    rule.description = (
        f"Observed pattern: decisions matching "
        f"{_condition_description(normalized)} selected {choice} in "
        f"{len(supporting_indices)} of {len(matched_indices)} matching records. "
        "This is historical evidence, not a recommendation."
    )
    rule.observed_pattern = {
        **rule.observed_pattern,
        "matching_decisions": len(matched_indices),
        "supporting_decisions": len(supporting_indices),
        "contradictory_decisions": len(contradicting_indices),
        "support_proportion": len(supporting_indices) / len(matched_indices),
        "distinct_scenarios": distinct_scenarios,
        "distinct_decision_makers": distinct_decision_makers,
        "quality_checks": {
            "minimum_evidence_met": (
                len(matched_indices) >= thresholds["minimum_matching_decisions"]
            ),
            "scenario_representation_met": (
                distinct_scenarios >= thresholds["minimum_scenarios"]
            ),
            "decision_maker_representation_met": (
                distinct_decision_makers >= thresholds["minimum_decision_makers"]
            ),
        },
        "first_observed_at": min(dataset.decision_times[index] for index in matched_indices),
        "last_observed_at": max(dataset.decision_times[index] for index in matched_indices),
    }
    for index in matched_indices:
        decision_id = dataset.decision_ids[index]
        decision = session.get(Decision, decision_id)
        if decision is None:
            continue
        evidence_type = (
            "supporting" if dataset.labels[index] == choice else "contradictory"
        )
        session.add(RuleEvidence(
            id=str(uuid4()),
            rule_id=rule.id,
            decision_id=decision_id,
            evidence_type=evidence_type,
            evidence_description=(
                f"Decision {decision.confidence_note or decision.id} selected "
                f"{dataset.labels[index]}; extracted pattern observes {choice}."
            ),
            data_source="observed_human",
        ))


def relevant_rules(
    session: Session,
    scenario_id: str,
    decision_maker_id: str,
) -> list[dict[str, Any]]:
    situation = session.get(Situation, scenario_id)
    decision_maker = session.get(DecisionMaker, decision_maker_id)
    if situation is None or decision_maker is None or not decision_maker.is_active:
        return []
    rules = session.scalars(
        select(DecisionRule)
        .where(
            DecisionRule.status == "APPROVED",
            DecisionRule.published_at.is_not(None),
        )
        .order_by(DecisionRule.rule_name, DecisionRule.version.desc())
    ).all()
    if not rules:
        return []

    active_extractions = {
        extraction.id: extraction
        for extraction in session.scalars(
            select(RuleExtraction).where(
                RuleExtraction.id.in_({rule.extraction_id for rule in rules})
            )
        ).all()
    }
    preferences = session.scalar(
        select(Preference)
        .where(
            Preference.decision_maker_id == decision_maker_id,
            Preference.is_current.is_(True),
        )
        .order_by(Preference.version.desc())
    )
    if preferences is None:
        return []

    assessments = session.scalars(
        select(Assessment)
        .where(Assessment.situation_id == scenario_id)
        .order_by(Assessment.assessment_version.desc(), Assessment.id.desc())
    ).all()
    score_by_alternative: dict[str, dict[str, float]] = {}
    for assessment in assessments:
        score_by_alternative.setdefault(
            assessment.alternative_id,
            assessment.criterion_scores,
        )
    alternatives = session.scalars(
        select(Alternative)
        .where(Alternative.situation_id == scenario_id)
        .order_by(Alternative.code)
    ).all()
    results = []
    for rule in rules:
        extraction = active_extractions.get(rule.extraction_id)
        if extraction is None:
            continue
        parameters = extraction.extraction_parameters
        criteria = parameters["criteria"]
        alternative_codes = parameters["alternative_codes"]
        feature_alternatives = []
        valid = True
        for alternative in alternatives:
            scores = score_by_alternative.get(alternative.id)
            if scores is None:
                valid = False
                break
            feature_alternatives.append({
                "code": alternative.code,
                "scores": scores,
                "available": alternative.is_available,
                "attributes": alternative.attributes,
            })
        if not valid:
            continue
        try:
            feature_row = extract_feature_row(
                situation.context_data,
                decision_maker.role,
                preferences.weights,
                feature_alternatives,
                criteria,
                alternative_codes,
            )
        except ValueError:
            continue
        if not evaluate_conditions(
            rule.conditions_json,
            feature_row,
            parameters.get("numeric_medians", {}),
        ):
            continue
        evidence = rule.evidences
        results.append({
            **serialize_rule(rule),
            "matching_conditions": rule.conditions_json,
            "supporting_decisions": sum(
                item.evidence_type == "supporting" for item in evidence
            ),
            "contradictory_decisions": sum(
                item.evidence_type == "contradictory" for item in evidence
            ),
        })
    return results


def serialize_rule(rule: DecisionRule) -> dict[str, Any]:
    extraction = rule.extraction
    return {
        "id": rule.id,
        "rule_name": rule.rule_name,
        "description": rule.description,
        "conditions": rule.conditions_json,
        "observed_pattern": rule.observed_pattern,
        "status": rule.status,
        "version": rule.version,
        "version_group": rule.version_group,
        "supersedes_rule_id": rule.supersedes_rule_id,
        "published_at": rule.published_at.isoformat() if rule.published_at else None,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "source": "model",
        "extraction": {
            "id": extraction.id,
            "model_name": extraction.model_name,
            "model_version": extraction.model_version,
            "dataset_version": extraction.dataset_version,
            "method": extraction.extraction_method,
            "created_at": extraction.created_at.isoformat() if extraction.created_at else None,
        },
    }


def serialize_rule_evidence(evidence: RuleEvidence) -> dict[str, Any]:
    decision = evidence.decision
    event = decision.decision_event
    return {
        "id": evidence.id,
        "evidence_type": evidence.evidence_type,
        "description": evidence.evidence_description,
        "decision_id": decision.confidence_note or f"DEC-{decision.id}",
        "database_decision_id": decision.id,
        "decision_maker_id": event.decision_maker_id,
        "decision_maker_name": event.decision_maker.name,
        "scenario_id": event.situation_id,
        "choice": decision.chosen_alternative.code,
        "rationale": decision.rationale,
        "decision_source": decision.data_source,
        "timestamp": decision.decided_at.isoformat(),
    }


def serialize_rule_review(review: RuleReview) -> dict[str, Any]:
    return {
        "id": review.id,
        "reviewer_id": review.reviewer_id,
        "reviewer_role": review.reviewer_role,
        "review_stage": review.review_stage,
        "review_decision": review.review_decision,
        "comments": review.comments,
        "modified_conditions": review.modified_conditions,
        "reviewed_at": review.reviewed_at.isoformat(),
    }


def serialize_rule_application(application: RuleApplication) -> dict[str, Any]:
    return {
        "id": application.id,
        "rule_id": application.rule_id,
        "rule_version": application.rule_version,
        "scenario_id": application.situation_id,
        "decision_id": (
            application.decision.confidence_note or f"DEC-{application.decision_id}"
            if application.decision_id
            else None
        ),
        "relevance_status": application.relevance_status,
        "considered_by_human": application.considered_by_human,
        "feedback": application.feedback,
        "created_at": application.created_at.isoformat() if application.created_at else None,
    }


def serialize_rule_feedback(feedback: RuleFeedback) -> dict[str, Any]:
    return {
        "id": feedback.id,
        "outcome_id": feedback.outcome_id,
        "feedback_type": feedback.feedback_type,
        "comments": feedback.comments,
        "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
    }


def rule_history(rule: DecisionRule) -> list[dict[str, Any]]:
    history = []
    seen = set()
    current: DecisionRule | None = rule
    while current is not None and current.id not in seen:
        seen.add(current.id)
        history.append(serialize_rule(current))
        current = current.supersedes
    return list(reversed(history))

