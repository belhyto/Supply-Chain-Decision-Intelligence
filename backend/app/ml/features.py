from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

SCENARIO_NUMERIC_KEYS = (
    "required_units",
    "required_within_days",
    "demand_increase_percent",
    "current_inventory_days",
    "required_inventory_days",
    "available_budget_lakh",
)


class FeatureDataError(ValueError):
    """Raised when persisted records cannot form the model's feature schema."""


def feature_columns(
    criterion_keys: Sequence[str],
    alternative_codes: Sequence[str],
) -> tuple[list[str], list[str]]:
    numeric = [f"situation_{key}" for key in SCENARIO_NUMERIC_KEYS]
    numeric.extend(f"preference_{key}" for key in criterion_keys)
    for code in alternative_codes:
        numeric.append(f"alternative_{code}_available")
        numeric.append(f"alternative_{code}_capacity_units")
        numeric.extend(
            f"alternative_{code}_{criterion}"
            for criterion in criterion_keys
        )
    categorical = [
        "decision_maker_role",
        "situation_stockout_risk",
        "situation_threat_level",
    ]
    return numeric, categorical


def _number(value: Any, feature_name: str) -> float:
    if value is None or value == "":
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FeatureDataError(f"{feature_name} must be numeric.") from error
    if not math.isfinite(number):
        raise FeatureDataError(f"{feature_name} must be finite.")
    return number


def extract_feature_row(
    situation_context: Mapping[str, Any],
    decision_maker_role: str | None,
    preferences: Mapping[str, Any],
    alternatives: Sequence[Mapping[str, Any]],
    criterion_keys: Sequence[str],
    alternative_codes: Sequence[str],
) -> dict[str, Any]:
    expected_criteria = set(criterion_keys)
    if set(preferences) != expected_criteria:
        raise FeatureDataError("Preference weights must match the model criteria.")

    alternatives_by_code = {str(item["code"]): item for item in alternatives}
    if len(alternatives_by_code) != len(alternatives):
        raise FeatureDataError("Alternative codes must be unique within a situation.")
    if set(alternatives_by_code) != set(alternative_codes):
        raise FeatureDataError("Alternative codes do not match the trained model.")

    row: dict[str, Any] = {
        f"situation_{key}": _number(situation_context.get(key), f"situation_{key}")
        for key in SCENARIO_NUMERIC_KEYS
    }
    weights = {
        key: _number(preferences[key], f"preference_{key}")
        for key in criterion_keys
    }
    if any(value < 0 or value > 1 for value in weights.values()) or not math.isclose(
        sum(weights.values()),
        1.0,
        abs_tol=1e-6,
    ):
        raise FeatureDataError("Preference weights must be between zero and one and sum to one.")
    row.update({f"preference_{key}": value for key, value in weights.items()})

    for code in alternative_codes:
        alternative = alternatives_by_code[code]
        scores = alternative.get("scores", {})
        if set(scores) != expected_criteria:
            raise FeatureDataError(f"Criterion scores for {code} do not match the model criteria.")
        for criterion in criterion_keys:
            score = _number(scores[criterion], f"alternative_{code}_{criterion}")
            if not 0 <= score <= 100:
                raise FeatureDataError(f"Criterion scores for {code} must be between zero and 100.")
            row[f"alternative_{code}_{criterion}"] = score

        attributes = alternative.get("attributes", {})
        row[f"alternative_{code}_available"] = float(bool(alternative.get("available", False)))
        row[f"alternative_{code}_capacity_units"] = _number(
            attributes.get("capacity_units"),
            f"alternative_{code}_capacity_units",
        )

    row["decision_maker_role"] = decision_maker_role or "unknown"
    row["situation_stockout_risk"] = str(situation_context.get("stockout_risk") or "unknown")
    row["situation_threat_level"] = str(situation_context.get("threat_level") or "unknown")
    return row


def feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> np.ndarray:
    feature_names = [*numeric_features, *categorical_features]
    if not rows:
        raise FeatureDataError("At least one feature row is required.")
    try:
        return np.asarray(
            [[row[name] for name in feature_names] for row in rows],
            dtype=object,
        )
    except KeyError as error:
        raise FeatureDataError(f"Feature row is missing {error.args[0]}.") from error
