from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def explain_random_forest(
    pipeline: Pipeline,
    feature_matrix: Any,
    predicted_choice: str,
    max_sampled_paths: int = 8,
    max_top_features: int = 10,
) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Random Forest explanations require a RandomForestClassifier pipeline.")

    transformed = preprocessor.transform(feature_matrix)
    feature_names = preprocessor.get_feature_names_out()
    feature_importances = sorted(
        (
            {"feature_name": str(name), "importance": float(importance)}
            for name, importance in zip(feature_names, classifier.feature_importances_)
            if importance > 0
        ),
        key=lambda item: (-item["importance"], item["feature_name"]),
    )
    votes: Counter[str] = Counter()
    vote_paths: list[dict[str, Any]] = []
    condition_support: dict[tuple[str, str, float, str], set[int]] = defaultdict(set)
    for tree_id, estimator in enumerate(classifier.estimators_):
        tree = estimator.tree_
        node_id = 0
        conditions = []
        tree_class_index = int(estimator.predict(transformed)[0])
        tree_choice = str(classifier.classes_[tree_class_index])
        votes[tree_choice] += 1
        while tree.children_left[node_id] != tree.children_right[node_id]:
            feature_index = int(tree.feature[node_id])
            value = float(transformed[0, feature_index])
            threshold = float(tree.threshold[node_id])
            operator = "<=" if value <= threshold else ">"
            branch = "left" if operator == "<=" else "right"
            condition = {
                "feature_name": str(feature_names[feature_index]),
                "condition_operator": operator,
                "threshold": threshold,
                "observed_value": value,
                "branch_taken": branch,
            }
            conditions.append(condition)
            if tree_choice == predicted_choice:
                condition_key = (
                    condition["feature_name"],
                    operator,
                    round(threshold, 4),
                    branch,
                )
                condition_support[condition_key].add(tree_id)
            node_id = int(
                tree.children_left[node_id]
                if operator == "<="
                else tree.children_right[node_id]
            )

        if tree_choice == predicted_choice:
            vote_paths.append({
                "tree_id": tree_id,
                "tree_vote": tree_choice,
                "conditions": conditions,
            })

    tree_vote_summary = {
        str(choice): {
            "votes": votes.get(str(choice), 0),
            "share": votes.get(str(choice), 0) / len(classifier.estimators_),
        }
        for choice in classifier.classes_
    }
    target_vote_count = votes.get(predicted_choice, 0)
    support_threshold = max(2, math.ceil(target_vote_count * 0.1))
    recurring_rules = []
    support_by_condition = {
        key: len(tree_ids)
        for key, tree_ids in condition_support.items()
    }
    for (feature_name, operator, threshold, branch), support_count in support_by_condition.items():
        if support_count < support_threshold:
            continue
        recurring_rules.append({
            "feature_name": feature_name,
            "condition_operator": operator,
            "threshold": threshold,
            "branch_taken": branch,
            "support_count": support_count,
            "support_confidence": support_count / target_vote_count if target_vote_count else 0,
            "tree_vote": predicted_choice,
        })
    recurring_rules.sort(
        key=lambda rule: (-rule["support_count"], rule["feature_name"], rule["threshold"])
    )

    for path in vote_paths:
        for condition in path["conditions"]:
            key = (
                condition["feature_name"],
                condition["condition_operator"],
                round(condition["threshold"], 4),
                condition["branch_taken"],
            )
            condition["support_count"] = support_by_condition.get(key, 0)
            condition["support_confidence"] = (
                condition["support_count"] / target_vote_count
                if target_vote_count
                else 0
            )

    return {
        "kind": "random_forest_pattern_analysis",
        "predicted_choice": predicted_choice,
        "tree_vote_summary": tree_vote_summary,
        "top_features": feature_importances[:max_top_features],
        "recurring_rules": recurring_rules[:10],
        "rule_paths": vote_paths[:max_sampled_paths],
        "rule_path_note": (
            "Paths are model-derived patterns for this feature row, not evidence of "
            "the human decision-maker's actual reasoning."
        ),
    }
