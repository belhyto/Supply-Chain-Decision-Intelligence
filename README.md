# Configurable Decision Intelligence Pipeline — Mock Prototype

A three-stage prototype:
1. Situation assessment and option suitability scoring.
2. Decision-maker-specific preference weighting and utility scoring.
3. Choice probability estimates, comparison, and explanations.

The supply-chain workflow is presented in a four-layer architecture: physical operations, digital twin, the three-stage decision pipeline, and decision/outcome feedback.

## Synthetic scenario training fixture
The prototype includes eight scenario records with three alternatives each, four decision-maker profiles, 32 recorded decisions, and 32 linked example outcomes in `backend/data/training/`. Select a scenario in the dashboard to assess it through all three decision stages. Stage 3 compares predicted choices and probabilities with recorded synthetic choices and displays the linked outcomes.

The fixture can be downloaded as [multi_decision_maker_training_scenarios.zip](./backend/data/multi_decision_maker_training_scenarios.zip).

The dataset is synthetic and for workflow/model testing only. Reported top-choice agreement and multiclass Brier score describe this fixture; they are not evidence of real-world behavior, validated predictions, or model calibration. Predicted choice, recorded choice, and observed outcome remain distinct fields. See `backend/data/training/README.md` for the CSV schema and interpretation.
## Stack
- Backend: Python 3.10+, FastAPI, Pydantic
- Frontend: React + Vite
- Configuration-driven mock data plus an eight-scenario synthetic training fixture with 24 alternatives, four decision-maker profiles, 32 recorded decisions, and 32 outcomes

## Important interpretation
Suitability scores are weighted evaluations of option fit. Choice probabilities are illustrative model estimates produced by a softmax function, not empirical probabilities. Calibrate the probability model against observed historical choices before operational use.

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
`SCN-001` models a high-demand stockout risk with three supplier alternatives. The scenario selector also includes primary supplier disruption, procurement cost spike, supplier quality alert, seasonal demand surge, transport strike, perishable goods, and single-source dependency. Four decision-maker profiles—Procurement Manager, Operations Manager, Risk & Quality Manager, and Supply Chain Director—evaluate each situation. Scenario-specific operational weights and supplier scores are loaded from CSV.

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
- `GET /api/feedback` — decisions and any captured outcomes
- `POST /api/feedback/decisions` — record a human-selected option with its model prediction and probability
- `POST /api/feedback/decisions/{decision_id}/outcome` — record fulfillment status and optional twin updates from observed operations

## Run backend tests
From the `backend` directory:
```bash
python -m unittest discover -s tests
```

## Prototype layer behavior
- The physical network is the mock baseline in `physical_network` within the domain configuration.
- The digital twin starts from `digital_twin.state` and is updated in server memory by operational telemetry requests. Supplier availability and capacity gate candidate options, and the latest supplier performance scores feed the next assessment.
- Decision records capture the actual selected option alongside the modeled top choice, selected-option probability, and twin revision at decision time.
- Outcomes can carry validated twin updates, linking observed results back to the evolving representation. Updating the twin directly is also supported through the dashboard.
- Twin state, event history, decisions, and outcomes are in-memory prototype data and reset when the backend restarts. No live WMS, ERP, TMS, supplier, or IoT feeds are connected.

## Before production
Add authentication/RBAC, audit logs, validation of business constraints, persistent storage, tests, model calibration, monitoring, and human approval workflows. Do not use the sample weights or probability temperature as production policy without domain validation.
