# Multi-Decision-Maker Decision Intelligence System with Supply Chain Application

This system incorporates a three-stage process:
1. Situation assessment and option suitability scoring.
2. Decision-maker-specific preference weighting and utility scoring.
3. Choice probability estimates, comparison, and explanations.

The supply-chain workflow is presented in a four-layer architecture: physical operations, digital twin, the three-stage decision pipeline, and decision/outcome feedback.

## Synthetic scenario training fixture
The prototype includes 24 synthetic scenario records with three alternatives each, four decision-maker profiles, 96 recorded example choices, and 96 linked synthetic outcomes in `backend/data/training/`. Select any scenario in the dashboard to assess it through all three decision stages. Stage 3 compares estimated choices and probabilities with the recorded synthetic examples and displays their linked example outcomes.

The fixture can be downloaded as [multi_decision_maker_training_scenarios.zip](./backend/data/multi_decision_maker_training_scenarios.zip).

The dataset is synthetic and for workflow/model testing only. Reported scores describe these generated records; they are not evidence of real-world behavior, validated predictions, or model calibration. Predicted choice, recorded choice, and observed outcome remain distinct fields. The feedback screen persists ML predictions first, then stores the human decision separately so preference weights, rationale, and any later outcome can be compared without treating a difference as an error. See `backend/data/training/README.md` for the CSV schema and interpretation.
## Stack
- Backend: Python 3.10+, FastAPI, Pydantic
- Frontend: React + Vite
- Configuration-driven mock data plus a 24-scenario synthetic training fixture with 72 alternatives, four decision-maker profiles, 96 recorded choices, and 96 example outcomes

## Important interpretation
Suitability scores are weighted evaluations of option fit, separate from a person's actual choice. Logistic regression uses the synthetic fixture. Random Forest is experimental: by default it trains only on observed human choices, while an explicit synthetic-demo option is available for testing with generated records. Its feature importance and tree paths are model-derived patterns, never evidence of a particular person's reasoning. Rule-memory extraction remains restricted to observed human choices. Human objectives, constraints, considered alternatives, preferences, stated factors, ordered reasoning steps, confidence, rationale, and later outcomes are preserved separately in the decision context and graph. Neither model is validated for operational use.

## Configure another domain
The pipeline reads its domain, situation, criteria, weights, available options, and decision-maker profiles from `backend/app/config/decision_pipeline.json`. Edit that file to create another domain without changing the scoring implementation:
- Set `domain_name` and the scenario's `id`, `title`, `situation`, and optional `context` fields.
- Define criteria with unique `key` and display `label` values. Option metrics are suitability scores from 0 to 100, where higher values always indicate a better fit.
- Provide one `operational_weights` value per criterion. Weights are normalized by the API, so they do not have to sum to 1.
- Define options with unique `id` and `name` values, a `metrics` score for every criterion, and optional `notes` and display `attributes`.
- Define decision-maker profiles with unique IDs and one preference weight per criterion. The interface supports any number of criteria, options, and profiles.
- `attribute_labels` optionally maps option attribute keys to readable labels. `probability_temperature` controls the default illustrative softmax estimate.
- For the supply-chain twin, define `physical_network` collections for warehouses, inventory, orders, and shipments, plus a `digital_twin.state` snapshot with supplier `id`, `capacity_units`, `is_available`, and criterion `metrics`, along with inventory, shipment, and order states. Supplier IDs must match option IDs.

Invalid configuration fails at backend startup with a clear error. Request weights must include exactly the configured criteria and be finite, non-negative values.

## Run the backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
__
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Run the frontend
In a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (usually http://localhost:5173).
In development, Vite proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:8000`.

## Default training scenario
`SCN-001` models a high-demand stockout risk with three supplier alternatives. The selector includes 23 further synthetic cases covering supplier disruption, cost and quality risk, capacity constraints, transport interruptions, perishable goods, customs, launches, and sustainability requirements. Four decision-maker profiles—Procurement Manager, Operations Manager, Risk & Quality Manager, and Supply Chain Director—evaluate each situation. Scenario-specific operational weights and supplier scores are loaded from CSV.

## API routes
- `GET /api/health` — health check
- `GET /api/config` — active domain configuration
- `GET /api/mock-data` — compatibility alias for the active configuration
- `POST /api/assess` — calculate suitability, utilities, probabilities, and comparisons
- `GET /api/training/decisions` — read synthetic decision records and linked outcomes; optionally filter with `scenario_id`
- `GET /api/training/evaluation` — descriptive top-choice agreement, multiclass Brier score, and recorded-choice disagreement counts against the synthetic records
- `GET /api/physical` — physical network: warehouses, inventory, suppliers, shipments, and orders
- `GET /api/twin` — current digital-twin snapshot and recent update events
- `POST /api/twin/updates` — validate and apply an operational update to inventory, supplier, shipment, or order state
- `POST /api/ml/predict` — generate and persist an ML prediction event and its alternative probabilities without recording a human decision
- `POST /api/ml/decisions` — record a human-selected alternative and rationale separately against a prediction event
- `GET /api/feedback` — recorded ML and interactive decisions, full probability distributions, preference weights, and captured outcomes
- `POST /api/feedback/decisions` — legacy interactive decision route using the illustrative softmax probabilities
- `POST /api/feedback/decisions/{decision_id}/outcome` — record fulfillment status and optional twin updates from observed operations
- `GET /api/context-graph?scenario_id=SCN-001&depth=3` — read a scenario-centered graph projection; omit the scenario ID for the full graph
- `GET /api/context-graph/nodes/{node_key}/neighbors?depth=1` — traverse a node's relationships in either direction
- `GET /api/context-graph/similar/{scenario_id}` — retrieve historical situations connected with `SIMILAR_TO`
- `POST /api/context-graph/nodes` — persist a constraint, event, policy, external condition, evidence item, or source and connect it to a situation
- `POST /api/context-graph/relationships` — persist a typed relationship between existing graph nodes

## Dashboard demonstration
1. Select `SCN-001 · Supplier disruption`, then open **Decision feedback**.
2. Select **Procurement Manager** and use the sample decision: choose `Supplier A` with the rationale “Supplier A meets the cost requirement while staying within the ₹20 lakh budget.”
3. Record the human choice, then use the sample outcome “1,200 units arrived before the 7-day deadline; no stockout occurred.”
4. Open **Knowledge graph**, keep the `Alternative` node filter to see the supplier-choice network, or choose **All types** to explore more of the scenario. The sample context input is “Primary supplier unavailable; demand is up 30% and only five inventory days remain.”

The charts compare the model's most-likely alternative with recorded example choices and show confidence on logged decisions. These are descriptive views of synthetic prototype data, not claims of validated model accuracy.

## Train and evaluate the experimental Random Forest
First seed PostgreSQL with the fixture, then explicitly select the synthetic source when you want a demonstration model:

```powershell
cd backend
python seed_postgresql.py
python -m app.ml.train --random-forest-source synthetic
```

Training evaluates a grouped holdout: complete scenarios are held out from fitting, and the JSON output includes accuracy, balanced accuracy, macro-F1, multiclass Brier score, and per-class precision/recall/F1. The trained artifact then fits the final model on all 96 synthetic choice examples. These holdout metrics measure agreement with the generated choice rules only; they do not estimate how real people decide. To keep the default observed-human workflow, omit the flag (or use `--random-forest-source observed_human`). The reviewed rule-memory extractor always uses observed human decisions and does not consume synthetic records.

## Run backend tests
From the `backend` directory:
```bash
python -m unittest discover -s tests
```

## Knowledge and memory graph
The backend persists the existing situation, alternative, preference, assessment, decision-event, prediction, decision, and outcome records in PostgreSQL. `ContextNode` and `ContextRelationship` add extensible graph entities and typed edges with JSON properties, timestamps, and provenance. On startup, the graph is idempotently seeded from the synthetic training fixture.

NetworkX builds an in-memory undirected `MultiGraph` from the persisted records for depth-limited traversal, neighbor lookup, and similar-situation retrieval. A relationship is stored once with its semantic source and target; the graph supports traversing it from either endpoint. Prediction distributions use `HAS_PREDICTION`/`EVALUATES`, while the modeled top choice is explicitly linked with `PREDICTED_AS`. Interactive decisions and their outcomes are also persisted, including links to the prediction, selected alternative, outcome, and any supplied operational context.

The graph includes Situation, Parameter, Alternative, Criterion, DecisionMaker, Preference, Assessment, DecisionEvent, Prediction, Decision, Outcome, and extensible Context nodes. Seed provenance is marked as synthetic; interactive user-entered context and decisions use distinct provenance labels.

## Decision trace and reviewed rule memory
`GET /api/decisions/{decision_id}` reconstructs a decision from its persisted decision-time situation, alternative, preference, and human-reasoning snapshots. It returns predictions and Random Forest explanations separately from human rationale, includes model/dataset versions and extracted-rule evidence, and reports an outcome as `null` until one is recorded. Decision-time snapshots and published rule versions are protected from later mutation by SQLAlchemy guards and PostgreSQL triggers; application startup creates missing tables and applies the versioned trigger migrations.

The experimental rule-memory API is under `/api/rules`: `POST /extract` creates candidate patterns only from observed human decisions; `GET /candidates`, `GET /{rule_id}/evidence`, and `GET /{rule_id}/history` expose candidates and provenance; `POST /{rule_id}/review` supports separate evidence and publication review; `POST /{rule_id}/publish` publishes approved immutable versions; `GET /relevant/{scenario_id}` retrieves matching published patterns without treating them as recommendations; and `POST /{rule_id}/feedback` records human relevance and outcome feedback. Rules retain model version, dataset hash, supporting and contradictory decision records, reviewer actions, and graph links. The reviewer-ID/role check is a prototype workflow control, not authentication: the application has no identity provider, so it must not be used as an authorization boundary in a deployed system.

PostgreSQL integration tests are opt-in. Set `POSTGRES_TEST_DATABASE_URL` to a disposable PostgreSQL database before running `python -m unittest tests.test_postgresql` from `backend`; the suite creates a temporary schema and rolls it back. Do not point it at a production database. The regular backend test command continues to run without this variable, skipping PostgreSQL-only cases.

## Prototype layer behavior
- The physical network is the mock baseline in `physical_network` within the domain configuration.
- The digital twin starts from `digital_twin.state` and is updated in server memory by operational telemetry requests. Supplier availability and capacity gate candidate options, and the latest supplier performance scores feed the next assessment.
- Decision records capture the actual selected option alongside the modeled top choice, selected-option probability, and twin revision at decision time.
- Outcomes can carry validated twin updates, linking observed results back to the evolving representation. Updating the twin directly is also supported through the dashboard.
- Twin state and its operational update event list are in-memory and reset when the backend restarts. Interactive human decisions and outcomes are persisted in PostgreSQL and represented in the context graph. No live WMS, ERP, TMS, supplier, or IoT feeds are connected.

