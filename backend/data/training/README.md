# Synthetic multi-decision-maker training scenarios

This fixture is synthetic. It is intended for exercising scenario selection, suitability scoring, preference rankings, illustrative choice probabilities, disagreement analysis, and prediction-versus-recorded-choice evaluation. It does not represent observed human behavior and must not be used to claim real-world predictive accuracy.

## Dataset size

- 8 scenarios (`SCN-001` through `SCN-008`)
- 3 supplier alternatives per scenario (24 rows)
- 4 decision-maker profiles
- 32 recorded decisions (one per scenario/profile pair)
- 32 corresponding observed-outcome examples (one per decision)

## Files and keys

- `scenarios.csv`: situation, decision conflict, demand/lead-time parameters, and operational criterion weights. Primary key: `scenario_id`.
- `options.csv`: per-scenario supplier availability, 0-100 criterion scores, attributes, and notes. Composite key: `scenario_id`, `option_id`. JSON-valued cells use RFC 4180 CSV escaping.
- `decision_maker_profiles.csv`: the four illustrative profiles and cost/reliability/speed/risk preference weights. Primary key: `decision_maker_id`. Each profile's weights sum to 1.
- `decisions.csv`: the selected option and rationale for each person/scenario pair. Primary key: `decision_id`; includes `scenario_id`, `decision_maker_id`, and `recorded_choice_id`.
- `outcomes.csv`: outcome details joined to decisions by `decision_id`; includes fulfillment status, actual delivery days, cost variance, stockout units, and notes.

The pipeline keeps three concepts separate:

1. **Predicted choice**: highest utility under the selected profile, with probabilities estimated by the illustrative softmax model.
2. **Recorded choice**: the synthetic decision in `decisions.csv`.
3. **Outcome**: the synthetic result in `outcomes.csv`.

The `/api/training/evaluation` endpoint reports top-choice agreement, multiclass Brier score, and how many decision-maker pairs recorded different choices in each scenario. These metrics describe this fixture only. Do not treat them as model validation, training evidence, or calibration results for real decision-makers.

## Use in the prototype

Select one of the eight scenarios in the dashboard. Its operational context and alternatives feed the existing three-stage pipeline. Stage 3 displays that scenario's four recorded choices and outcomes alongside descriptive evaluation metrics across all 32 records. The API also exposes the underlying joined decision data.

The normal decision-feedback forms remain separate from this static fixture; interactive feedback is in-memory and resets when the backend restarts.
