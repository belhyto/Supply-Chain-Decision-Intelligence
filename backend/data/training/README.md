# Synthetic multi-decision-maker training scenarios

This fixture is synthetic. It is intended for exercising scenario selection, suitability scoring, preference rankings, illustrative choice probabilities, disagreement analysis, and prediction-versus-recorded-choice evaluation. It does not represent observed human behavior and must not be used to claim real-world predictive accuracy.

## Dataset size

- 8 scenarios (`SCN-001` through `SCN-008`)
- 3 supplier alternatives per scenario (24 rows)
- 4 decision-maker profiles
- 32 recorded decisions (one per scenario/profile pair)
- 24 suitability assessments and 96 alternative-level choice probabilities when seeded to PostgreSQL
- 32 corresponding synthetic outcome examples (one per decision)

`SCN-001` is the supplier-disruption example: 30% demand increase, five inventory days, a 20-day inventory target, a seven-day delivery deadline, and a ₹20 lakh budget. Its three suitability scores are 75.25, 82.25, and 82.25. Its decision probabilities are supplied illustrative values and are not calibrated human-behavior predictions.

## Files and keys

- `scenarios.csv`: situation, decision conflict, demand/lead-time parameters, and operational criterion weights. Primary key: `scenario_id`.
- `options.csv`: per-scenario supplier availability, 0-100 criterion scores, attributes, and notes. Composite key: `scenario_id`, `option_id`. JSON-valued cells use RFC 4180 CSV escaping.
- `decision_maker_profiles.csv`: the four illustrative profiles and cost/reliability/speed/risk preference weights. Primary key: `decision_maker_id`. Each profile's weights sum to 1.
- `decisions.csv`: the selected option and rationale for each person/scenario pair. Primary key: `decision_id`; includes `scenario_id`, `decision_maker_id`, and `recorded_choice_id`. `predicted_probabilities_json` optionally provides an explicitly illustrative distribution; otherwise the seed computes a softmax distribution.
- `outcomes.csv`: outcome details joined to decisions by `decision_id`; includes fulfillment status, actual delivery days, cost variance, stockout units, and notes.

The pipeline keeps three concepts separate:

1. **Predicted choice**: highest utility under the selected profile, with probabilities estimated by the illustrative softmax model.
2. **Recorded choice**: the synthetic decision in `decisions.csv`.
3. **Outcome**: the synthetic result in `outcomes.csv`.

The `/api/training/evaluation` endpoint reports top-choice agreement, multiclass Brier score, and how many decision-maker pairs recorded different choices in each scenario. These metrics describe this fixture only. Do not treat them as model validation, training evidence, or calibration results for real decision-makers.

## Use in the prototype

Select one of the eight scenarios in the dashboard. Its operational context and alternatives feed the existing three-stage pipeline. Stage 3 displays that scenario's four recorded choices and outcomes alongside descriptive evaluation metrics across all 32 records. The API also exposes the underlying joined decision data.

The normal decision-feedback forms remain separate from this static fixture; interactive decisions and outcomes are persisted in PostgreSQL and linked into the context graph.

## ML models and decision explanations

With the PostgreSQL fixture seeded, train the logistic regression prototype from the `backend` directory. The command also attempts to train the experimental Random Forest from observed human decisions:

```powershell
python -m app.ml.train
```

Logistic regression trains on the 32 synthetic decision records. Its features include scenario numeric parameters, each decision-maker's criterion weights, each alternative's criterion scores, availability and capacity, and categorical decision-maker role and situation risk. It uses median imputation, `StandardScaler`, and `OneHotEncoder`. A `GroupShuffleSplit` holds out complete scenario IDs for a prototype score, then the production pipeline is fit on all synthetic records.

Random Forest is not trained on the synthetic fixture. It requires observed human decisions across at least four distinct scenarios and recorded examples for all alternatives. Until those records exist, the training command reports that it remains untrained, and the API will not load a stale model trained on synthetic labels. Once sufficient observed data is available, the model uses the same pre-choice situation, preference, alternative-score, availability, capacity, and role features; human-stated post-choice factors and rationale are deliberately excluded to avoid label leakage. It uses median imputation and `OneHotEncoder`, with no numeric scaling.

The decision-feedback dashboard keeps **Multinomial logistic regression** as the default. **Experimental historical pattern analysis · Random Forest** is available only after training on observed human choices. Random Forest's output includes forest vote distribution, global feature importance, recurring split conditions, and representative tree paths. Those are patterns in the model—not the actual reasoning behind any individual decision. SHAP is not currently calculated.

Request a decision-maker-specific prediction:

```http
POST /api/ml/predict
Content-Type: application/json

{"scenario_id":"SCN-001","decision_maker_id":"DM-01"}
```

The existing logistic-regression request remains compatible. To request experimental Random Forest analysis after it is trained, add `"model_name":"random_forest"`. The response also contains the situation and eligibility snapshot used by the model. A Random Forest response includes the model explanation; the explanation is stored separately from any human rationale.

Record a human choice separately by posting the returned `prediction_event_id` and the selected alternative code:

```http
POST /api/ml/decisions
Content-Type: application/json

{"prediction_event_id":"SCN-001:DM-01:ml:<run-id>","chosen_option_id":"SUP-C","rationale":"Supplier C meets the deadline.","operational_objective":"Replenish 1,200 units within seven days.","constraints":["Stay within budget.","Deliver the full quantity."],"influencing_factors":["Only five inventory days remain."],"reasoning_steps":["Compare eligible suppliers.","Check lead time and capacity.","Select the option that meets the deadline."],"considered_alternative_codes":["SUP-B","SUP-C"],"confidence":0.8}
```

The human decision references the prediction event and is linked to the selected alternative independently of the model's probabilities. The decision snapshot also preserves objective, constraints, considered alternatives, the situation and supplier suitability at that time, preferences, human-stated influencing factors, ordered reasoning steps, optional confidence, and the human's own rationale. `GET /api/feedback` returns this context, the distinct model explanation if present, and any later operational outcome. Model explanations do not replace or infer a person's rationale.

The PostgreSQL `ModelExplanation` and `ExtractedRule` tables persist Random Forest vote summaries, feature importance, representative tree paths, and recurring split conditions. Important rules and human-stated `Objective`, `Constraint`, `DecisionFactor`, `ReasoningStep`, and `HumanStatedReason` nodes are connected to the existing NetworkX context graph with model or human provenance. No SHAP values are stored because SHAP is not implemented. The grouped holdout scores describe the underlying data only; the tiny synthetic fixture cannot establish real-world predictive behavior.

## Seed PostgreSQL

From the `backend` directory, with `DATABASE_URL` configured in `backend/.env`, run:

```powershell
python seed_postgresql.py
```

The script creates any missing tables, adds the `data_source` column to existing PostgreSQL tables when needed, and upserts the synthetic fixture in foreign-key order. It is safe to rerun: the scenario, profile, assessment, event, prediction, decision, outcome, and graph records use stable identifiers or uniqueness keys. Seeded records use `data_source = "synthetic"`; outcomes remain marked `actual_observed = false`. The script verifies the expected dataset counts before reporting success.
