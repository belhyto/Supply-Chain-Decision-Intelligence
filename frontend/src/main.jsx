import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "/api";

function ScoreBar({ value, max = 100 }) {
  return <div className="bar"><div style={{ width: `${Math.min(100, value / max * 100)}%` }} /></div>;
}

function App() {
  const [data, setData] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [temperature, setTemperature] = useState(8);
  const [selectedScenario, setSelectedScenario] = useState("SCN-001");
  const [activeLayer, setActiveLayer] = useState(3);
  const [activeStage, setActiveStage] = useState(1);
  const [physical, setPhysical] = useState(null);
  const [twin, setTwin] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [trainingEvaluation, setTrainingEvaluation] = useState(null);

  async function runAssessment(temp = temperature, scenarioId = selectedScenario) {
    setError("");
    try {
      const response = await fetch(`${API}/assess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ temperature: Number(temp), scenario_id: scenarioId })
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail ?? `API returned ${response.status}`);
      }
      setResult(await response.json());
    } catch (assessmentError) {
      setError(assessmentError instanceof Error && assessmentError.message
        ? assessmentError.message
        : "Could not connect to the API. Start the FastAPI backend on port 8000, then refresh.");
    }
  }

  async function refreshTwin() {
    const response = await fetch(`${API}/twin`);
    if (!response.ok) throw new Error(`Twin API returned ${response.status}`);
    setTwin(await response.json());
  }

  async function refreshFeedback() {
    const response = await fetch(`${API}/feedback`);
    if (!response.ok) throw new Error(`Feedback API returned ${response.status}`);
    setFeedback((await response.json()).records);
  }

  async function changeScenario(scenarioId) {
    setError("");
    try {
      const response = await fetch(`${API}/config?scenario_id=${encodeURIComponent(scenarioId)}`);
      if (!response.ok) throw new Error(`Scenario API returned ${response.status}`);
      const config = await response.json();
      setSelectedScenario(scenarioId);
      setData(config);
      await runAssessment(temperature, scenarioId);
    } catch (scenarioError) {
      setError(scenarioError instanceof Error ? scenarioError.message : "Could not load scenario.");
    }
  }

  useEffect(() => {
    (async () => {
      try {
        const [configResponse, physicalResponse] = await Promise.all([
          fetch(`${API}/config?scenario_id=SCN-001`),
          fetch(`${API}/physical`)
        ]);
        if (!configResponse.ok || !physicalResponse.ok) throw new Error("Could not load pipeline data.");
        const [config, network] = await Promise.all([configResponse.json(), physicalResponse.json()]);
        setData(config);
        setPhysical(network);
        setSelectedScenario("SCN-001");
        const initialTemperature = config.probability_temperature ?? 8;
        setTemperature(initialTemperature);
        const [evaluationResponse] = await Promise.all([
          fetch(`${API}/training/evaluation`),
          runAssessment(initialTemperature, "SCN-001"),
          refreshTwin(),
          refreshFeedback()
        ]);
        if (!evaluationResponse.ok) throw new Error("Training evaluation could not be loaded.");
        setTrainingEvaluation(await evaluationResponse.json());
      } catch {
        setError("Backend data could not be loaded. Start the FastAPI backend on port 8000, then refresh.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <main className="shell"><p>Loading prototype…</p></main>;

  return (
    <main className="shell">
      <header className="topbar">
        <div>
        
          <h1>{data?.domain_name ?? "Decision"} <span>Decision Pipeline</span></h1>
          <p className="subtitle">Situation assessment · Personal preferences · Choice probabilities</p>
        </div>
    
      </header>

      {error && <div className="error">{error}</div>}
      {data && (
        <>
          <section className="scenario panel">
            <div>
              <div className="eyebrow">CURRENT SITUATION · {data.scenario.id}</div>
              <h2>{data.scenario.title}</h2>
              <p>{data.scenario.situation}</p>
              <label className="scenario-picker">Training scenario
                <select value={selectedScenario} onChange={event => changeScenario(event.target.value)}>
                  {data.training_scenarios.map(scenario => <option key={scenario.id} value={scenario.id}>{scenario.id} · {scenario.title}</option>)}
                </select>
              </label>
              <p className="hint scenario-conflict"><b>Decision conflict:</b> {data.scenario.decision_conflict}</p>
              <div className="chips">
                {Object.entries(data.scenario.context ?? {}).map(([label, value]) => (
                  <span key={label} className={label.toLowerCase().includes("risk") ? "danger-chip" : ""}>
                    {label}: {value}
                  </span>
                ))}
              </div>
            </div>
            <div className="scenario-mark">DI<br/><b>01</b></div>
          </section>

          <nav className="stage-nav layer-nav">
            {[["1", "Physical supply chain"], ["2", "Digital twin"], ["3", "Decision pipeline"], ["4", "Decision feedback"]].map(([num, label]) =>
              <button key={num} className={activeLayer === Number(num) ? "active" : ""} onClick={() => setActiveLayer(Number(num))}>
                <b>{num}</b><span>{label}</span>
              </button>
            )}
          </nav>

          {activeLayer === 1 && <PhysicalLayer physical={physical} options={data.options} />}
          {activeLayer === 2 && <DigitalTwinLayer twin={twin} criteria={data.criteria} onUpdated={async () => { await refreshTwin(); await runAssessment(); }} />}
          {activeLayer === 3 && <>
            <nav className="stage-nav pipeline-nav">
              {[["1", "Situation & suitability"], ["2", "Decision preferences"], ["3", "Choice probabilities"]].map(([num, label]) =>
                <button key={num} className={activeStage === Number(num) ? "active" : ""} onClick={() => setActiveStage(Number(num))}>
                  <b>{num}</b><span>{label}</span>
                </button>
              )}
            </nav>
            {result && activeStage === 1 && <StageOne result={result} data={data} />}
            {result && activeStage === 2 && <StageTwo result={result} criteria={data.criteria} />}
            {result && activeStage === 3 && <StageThree result={result} temperature={temperature} setTemperature={(value) => { setTemperature(value); runAssessment(value); }} trainingEvaluation={trainingEvaluation} scenarioId={selectedScenario} />}
          </>}
          {activeLayer === 4 && <FeedbackLayer data={data} result={result} records={feedback} onChanged={refreshFeedback} scenarioId={selectedScenario} />}
          <footer>
            <span>DIGITAL TWIN / MOCK OPERATIONAL DATA</span>
            <span>Illustrative decision support · Human decision remains final</span>
          </footer>
        </>
      )}
    </main>
  );
}

function PhysicalLayer({ physical, options }) {
  if (!physical) return null;
  return <>
    <section className="section-heading"><div><div className="eyebrow">LAYER 01 · PHYSICAL SUPPLY CHAIN</div><h2>Operational network</h2></div><div className="badge">MOCK SOURCE RECORDS</div></section>
    <p className="intro">Physical-world entities represented by this prototype: facilities, inventory, supplier capacity, shipments, and open orders.</p>
    <section className="panel"><h3>Warehouses & distribution centers</h3><div className="cards">{physical.warehouses.map(warehouse => <article className="data-card" key={warehouse.id}><span className="eyebrow">{warehouse.id}</span><h4>{warehouse.name}</h4><p>{warehouse.location}</p></article>)}</div></section>
    <section className="panel"><h3>Suppliers</h3><div className="cards">{options.map(option => <article className="data-card" key={option.id}><span className="eyebrow">{option.id}</span><h4>{option.name}</h4><p>{option.notes}</p><DataGrid values={option.attributes} /></article>)}</div></section>
    <section className="panel"><h3>Inventory & demand</h3><div className="cards">{physical.inventory.map(item => <article className="data-card" key={item.id}><span className="eyebrow">{item.sku}</span><h4>{item.product}</h4><DataGrid values={{ "On hand": `${item.on_hand_units} units`, "Reserved": `${item.reserved_units} units`, "Daily demand": `${item.daily_demand_units} units`, "Reorder point": `${item.reorder_point_units} units` }} /></article>)}</div></section>
    <div className="cards">
      <section className="panel"><h3>Orders</h3>{physical.orders.map(order => <div className="entity-row" key={order.id}><b>{order.id}</b><span>{order.quantity_units.toLocaleString()} units · due in {order.due_in_days} days</span><small>{order.status}</small></div>)}</section>
      <section className="panel"><h3>Transportation & shipments</h3>{physical.shipments.map(shipment => <div className="entity-row" key={shipment.id}><b>{shipment.id}</b><span>{shipment.quantity_units} units · {shipment.location}</span><small>{shipment.status} · ETA {shipment.estimated_delivery_days} days</small></div>)}</section>
    </div>
  </>;
}

function DataGrid({ values }) {
  return <div className="twin-grid">{Object.entries(values).map(([label, value]) => <div key={label}><small>{label}</small><b>{value}</b></div>)}</div>;
}

const TWIN_FIELDS = {
  inventory: ["on_hand_units", "reserved_units", "daily_demand_units"],
  supplier: ["capacity_units", "is_available", "reliability", "cost", "speed", "risk"],
  shipment: ["status", "location", "estimated_delivery_days"],
  order: ["status"]
};

const TWIN_COLLECTION = {
  inventory: "inventory",
  supplier: "suppliers",
  shipment: "shipments",
  order: "orders"
};

function DigitalTwinLayer({ twin, criteria, onUpdated }) {
  const [entityType, setEntityType] = useState("inventory");
  const [entityId, setEntityId] = useState("");
  const [field, setField] = useState(TWIN_FIELDS.inventory[0]);
  const [value, setValue] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  if (!twin) return null;

  const entities = twin[TWIN_COLLECTION[entityType]] ?? [];
  const selectedEntityId = entities.some(entity => entity.id === entityId) ? entityId : entities[0]?.id ?? "";
  const fields = entityType === "supplier" ? [...TWIN_FIELDS.supplier.filter(key => !criteria.some(criterion => criterion.key === key)), ...criteria.map(criterion => criterion.key)] : TWIN_FIELDS[entityType];

  async function submitUpdate(event) {
    event.preventDefault();
    setError("");
    setMessage("");
    const numericField = ["on_hand_units", "reserved_units", "daily_demand_units", "capacity_units", "estimated_delivery_days", ...criteria.map(criterion => criterion.key)].includes(field);
    const response = await fetch("/api/twin/updates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "Dashboard manual telemetry", entity_type: entityType, entity_id: selectedEntityId, field, value: numericField ? Number(value) : field === "is_available" ? value === "true" : value })
    });
    const body = await response.json();
    if (!response.ok) {
      setError(body.detail ?? "Twin update failed.");
      return;
    }
    setMessage(`Twin updated to revision ${body.twin.revision}. Assessment refreshed.`);
    setValue("");
    await onUpdated();
  }

  const currentRecord = entities.find(entity => entity.id === selectedEntityId);
  const currentValue = entityType === "supplier" && criteria.some(criterion => criterion.key === field)
    ? currentRecord?.metrics?.[field]
    : currentRecord?.[field];

  return <>
    <section className="section-heading"><div><div className="eyebrow">LAYER 02 · DIGITAL TWIN</div><h2>Live operational representation</h2></div><div className="badge">REVISION {twin.revision}</div></section>
    <p className="intro">The twin holds a working snapshot of inventory, supplier capacity/performance, shipment state, and orders. Updates immediately change eligible suppliers and decision rankings.</p>
    <section className="panel twin-summary"><div><span className="eyebrow">LAST UPDATE</span><b>{new Date(twin.last_updated).toLocaleString()}</b></div><div><span className="eyebrow">SOURCE</span><b>{twin.last_source}</b></div><div><span className="eyebrow">RECENT EVENTS</span><b>{twin.recent_events.length}</b></div></section>
    <div className="cards">
      <TwinEntityCard title="Supplier capacity & performance" entities={twin.suppliers} fields={["capacity_units", "is_available", ...criteria.map(criterion => criterion.key)]} labels={{ capacity_units: "Capacity (units)", is_available: "Available" }} />
      <TwinEntityCard title="Inventory & demand state" entities={twin.inventory} fields={["on_hand_units", "reserved_units", "daily_demand_units"]} labels={{ on_hand_units: "On hand", reserved_units: "Reserved", daily_demand_units: "Daily demand" }} />
      <TwinEntityCard title="Shipment locations & ETAs" entities={twin.shipments} fields={["status", "location", "estimated_delivery_days"]} labels={{ estimated_delivery_days: "ETA (days)" }} />
      <TwinEntityCard title="Order status" entities={twin.orders} fields={["status"]} labels={{}} />
    </div>
    <section className="panel">
      <h3>Apply incoming operational data</h3>
      <p className="hint">Simulate a validated feed update. Supplier performance changes are reflected in the next assessment; unavailable or under-capacity suppliers are excluded.</p>
      <form className="update-form" onSubmit={submitUpdate}>
        <label>Entity<select value={entityType} onChange={event => { setEntityType(event.target.value); setField(TWIN_FIELDS[event.target.value][0]); setValue(""); }}><option value="inventory">Inventory</option><option value="supplier">Supplier</option><option value="shipment">Shipment</option><option value="order">Order</option></select></label>
        <label>Record<select value={selectedEntityId} onChange={event => setEntityId(event.target.value)}>{entities.map(entity => <option key={entity.id} value={entity.id}>{entity.id}</option>)}</select></label>
        <label>Field<select value={field} onChange={event => { setField(event.target.value); setValue(""); }}>{fields.map(name => <option key={name} value={name}>{name}</option>)}</select></label>
        <label>New value{field === "is_available" ? <select value={value || String(currentValue ?? true)} onChange={event => setValue(event.target.value)}><option value="true">Available</option><option value="false">Unavailable</option></select> : <input value={value} onChange={event => setValue(event.target.value)} placeholder={`Current: ${currentValue ?? ""}`} required />}</label>
        <button type="submit" className="primary-button">Apply update</button>
      </form>
      {message && <p className="success-message">{message}</p>}
      {error && <p className="inline-error">{error}</p>}
    </section>
    <section className="panel"><h3>Recent twin update events</h3>{twin.recent_events.length === 0 ? <p className="hint">No operational updates recorded this session.</p> : twin.recent_events.map((event, index) => <div className="entity-row" key={`${event.revision}-${index}`}><b>Revision {event.revision}: {event.entity_id}</b><span>{event.field} → {String(event.value)}</span><small>{event.source} · {new Date(event.timestamp).toLocaleString()}</small></div>)}</section>
  </>;
}

function TwinEntityCard({ title, entities, fields, labels }) {
  return <section className="panel"><h3>{title}</h3>{entities.map(entity => <div className="twin-entity" key={entity.id}><span className="eyebrow">{entity.id}</span><DataGrid values={Object.fromEntries(fields.map(field => [labels[field] ?? field, field in (entity.metrics ?? {}) ? entity.metrics[field] : entity[field]]))} /></div>)}</section>;
}

function FeedbackLayer({ data, result, records, onChanged, scenarioId }) {
  const [decisionMakerId, setDecisionMakerId] = useState(data.decision_makers[0]?.id ?? "");
  const [chosenOptionId, setChosenOptionId] = useState("");
  const [rationale, setRationale] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function createDecision(event) {
    event.preventDefault();
    setError("");
    setMessage("");
    const response = await fetch("/api/feedback/decisions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId, decision_maker_id: decisionMakerId, chosen_option_id: chosenOptionId, rationale })
    });
    const body = await response.json();
    if (!response.ok) {
      setError(body.detail ?? "Decision could not be recorded.");
      return;
    }
    setMessage(`Decision ${body.id} recorded against predicted choice "${body.predicted_choice}".`);
    setRationale("");
    await onChanged();
  }

  return <>
    <section className="section-heading"><div><div className="eyebrow">LAYER 04 · DECISION FEEDBACK</div><h2>Decision and outcome learning loop</h2></div><div className="badge">HUMAN DECISION RECORDED</div></section>
    <p className="intro">Record the option a person actually selected, compare it with the model's most likely choice, and later attach the observed operational outcome. This prototype records feedback in memory for this server session.</p>
    <section className="panel">
      <h3>Record an actual decision</h3>
      <form className="update-form feedback-form" onSubmit={createDecision}>
        <label>Decision-maker<select value={decisionMakerId} onChange={event => setDecisionMakerId(event.target.value)}>{data.decision_makers.map(person => <option key={person.id} value={person.id}>{person.name}</option>)}</select></label>
        <label>Chosen option<select value={chosenOptionId || result?.stage1.rankings[0]?.option_id || ""} onChange={event => setChosenOptionId(event.target.value)} required>{(result?.stage1.rankings ?? []).map(option => <option key={option.option_id} value={option.option_id}>{option.option_name}</option>)}</select></label>
        <label className="wide-field">Decision rationale<input value={rationale} onChange={event => setRationale(event.target.value)} placeholder="Why was this option selected?" /></label>
        <button type="submit" className="primary-button">Record decision</button>
      </form>
      {message && <p className="success-message">{message}</p>}
      {error && <p className="inline-error">{error}</p>}
    </section>
    <section className="panel"><h3>Recorded decisions & observed outcomes</h3>{records.filter(record => record.scenario_id === scenarioId).length === 0 ? <p className="hint">No interactive decisions have been recorded for this scenario yet.</p> : records.filter(record => record.scenario_id === scenarioId).map(record => <DecisionRecord key={record.id} record={record} onChanged={onChanged} />)}</section>
  </>;
}

function DecisionRecord({ record, onChanged }) {
  const [status, setStatus] = useState("fulfilled");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState("");

  async function recordOutcome(event) {
    event.preventDefault();
    setError("");
    const response = await fetch(`/api/feedback/decisions/${record.id}/outcome`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, notes })
    });
    const body = await response.json();
    if (!response.ok) {
      setError(body.detail ?? "Outcome could not be recorded.");
      return;
    }
    await onChanged();
  }

  return <article className="feedback-record">
    <div className="record-heading"><div><span className="eyebrow">{record.id} · {record.decision_maker_name}</span><h4>Selected: {record.chosen_option_name}</h4></div><span className={record.choice_matches_prediction ? "match" : "different"}>{record.choice_matches_prediction ? "MATCHED PREDICTION" : "DIFFERED FROM PREDICTION"}</span></div>
    <p className="hint">Predicted: {record.predicted_choice} ({record.predicted_choice_probability_percent.toFixed(1)}%); chosen option estimate: {record.chosen_option_probability_percent.toFixed(1)}% · Twin revision {record.twin_revision}</p>
    {record.rationale && <p className="record-notes">{record.rationale}</p>}
    {record.outcome ? <div className="callout"><b>Outcome: {record.outcome.status}</b>{record.outcome.notes && <><br/><small>{record.outcome.notes}</small></>}</div> : <form className="outcome-form" onSubmit={recordOutcome}><label>Observed result<select value={status} onChange={event => setStatus(event.target.value)}><option value="fulfilled">Fulfilled</option><option value="delayed">Delayed</option><option value="partial">Partial</option><option value="failed">Failed</option></select></label><label>Outcome notes<input value={notes} onChange={event => setNotes(event.target.value)} placeholder="Delivery time, quantity, or cost observed" /></label><button type="submit" className="secondary-button">Save outcome</button></form>}
    {error && <p className="inline-error">{error}</p>}
  </article>;
}

function StageOne({ result, data }) {
  const criteriaByKey = Object.fromEntries(data.criteria.map(criterion => [criterion.key, criterion]));
  return <>
    <section className="section-heading"><div><div className="eyebrow">STAGE 01</div><h2>Situation assessment & suitability</h2></div><div className="badge">WEIGHTED SCORE / 100</div></section>
    <section className="panel twin-summary"><div><span className="eyebrow">TWIN REVISION</span><b>{result.operational_snapshot.twin_revision}</b></div><div><span className="eyebrow">AVAILABLE INVENTORY</span><b>{result.operational_snapshot.available_inventory_units ?? "N/A"} units · {result.operational_snapshot.days_of_inventory ?? "N/A"} days cover</b></div><div><span className="eyebrow">ELIGIBLE SUPPLIERS</span><b>{result.operational_snapshot.available_supplier_count}</b></div></section>
    <div className="two-col">
      <section className="panel">
        <h3>Operational scoring weights</h3>
        {Object.entries(result.stage1.weights).map(([key, value]) => <div className="weight-row" key={key}><span>{criteriaByKey[key]?.label ?? key}</span><b>{Math.round(value * 100)}%</b><ScoreBar value={value * 100}/></div>)}
        <p className="hint">Weights are normalized by the API. Option scores are configured assessments.</p>
      </section>
      <section className="panel">
        <h3>Suitability ranking</h3>
        {result.stage1.rankings.map((option, index) => <div className="supplier-row" key={option.option_id}>
          <div className="rank">{String(index + 1).padStart(2, "0")}</div>
          <div className="supplier-main"><b>{option.option_name}</b><small>{option.option_id} · {Object.entries(option.attributes).map(([key, value]) => `${data.attribute_labels[key] ?? key}: ${value}`).join(" · ")}</small><ScoreBar value={option.score}/></div>
          <strong className="score">{option.score.toFixed(2)}</strong>
        </div>)}
        <div className="callout"><b>Top-ranked:</b> {result.stage1.top_option}<br/><small>{result.stage1.interpretation}</small></div>
      </section>
    </div>
    <section className="panel supplier-cards">
      <h3>Available options</h3>
      <div className="cards">{data.options.map(option => <article key={option.id}>
        <span className="eyebrow">{option.id}</span><h4>{option.name}</h4><span className={option.available === false ? "option-unavailable" : "option-available"}>{option.available === false ? "Unavailable in this scenario" : "Available alternative"}</span><p>{option.notes}</p>
        <div className="mini-stats">{data.criteria.map(criterion => <span key={criterion.key}>{criterion.label}<b>{option.metrics[criterion.key]}</b></span>)}</div>
        {Object.keys(option.attributes ?? {}).length > 0 && <div className="mini-stats">{Object.entries(option.attributes).map(([key, value]) => <span key={key}>{data.attribute_labels[key] ?? key}<b>{value}</b></span>)}</div>}
      </article>)}</div>
    </section>
  </>;
}

function StageTwo({ result, criteria }) {
  const labels = Object.fromEntries(criteria.map(criterion => [criterion.key, criterion.label]));
  return <>
    <section className="section-heading"><div><div className="eyebrow">STAGE 02</div><h2>Independent decision-maker preferences</h2></div><div className="badge">PERSONAL WEIGHTS</div></section>
    <p className="intro">Each decision-maker evaluates the same available options using their configured priorities. Personal weights are independent of the Stage 1 operational weights.</p>
    <div className="cards">
      {result.stage2.map(person => <section className="panel person-card" key={person.decision_maker_id}>
        <div className="person-top"><div className="avatar">{person.decision_maker_name.split(" ").map(word => word[0]).join("").slice(0,2)}</div><div><h3>{person.decision_maker_name}</h3><span className="role">{person.role}</span></div></div>
        <p className="hint">{person.description || "Personal preference profile"}</p>
        {Object.entries(person.weights).map(([key, value]) => <div className="weight-row" key={key}><span>{labels[key] ?? key}</span><b>{Math.round(value * 100)}%</b><ScoreBar value={value * 100}/></div>)}
        <h4 className="subhead">Ranking under personal preferences</h4>
        {person.rankings.map((ranking, index) => <div className="compact-rank" key={ranking.option_id}><span>{index + 1}. {ranking.option_name}</span><b>{ranking.score.toFixed(2)}</b></div>)}
        <div className="callout"><b>Personal top choice:</b> {person.top_choice}</div>
      </section>)}
    </div>
  </>;
}

function StageThree({ result, temperature, setTemperature, trainingEvaluation, scenarioId }) {
  const scenarioRecords = trainingEvaluation?.records.filter(record => record.scenario_id === scenarioId) ?? [];
  const scenarioDisagreement = trainingEvaluation?.scenario_disagreement.find(scenario => scenario.scenario_id === scenarioId);
  return <>
    <section className="section-heading"><div><div className="eyebrow">STAGE 03</div><h2>Choice probabilities & decision comparison</h2></div><div className="badge">SOFTMAX ESTIMATE</div></section>
    <section className="panel temp-panel">
      <div><h3>Probability model sensitivity</h3><p className="hint">Lower temperature concentrates probability on the highest utility; higher temperature spreads it across options.</p></div>
      <label>Temperature <b>{Number(temperature).toFixed(1)}</b><input type="range" min="2" max="20" step="1" value={temperature} onChange={event => setTemperature(event.target.value)} /></label>
    </section>
    <div className="cards">
      {result.stage3.decision_maker_probabilities.map(person => <section className="panel" key={person.decision_maker_id}>
        <div className="person-top"><div className="avatar">{person.decision_maker_name.split(" ").map(word => word[0]).join("").slice(0,2)}</div><div><h3>{person.decision_maker_name}</h3><span className="role">Modeled choice distribution</span></div></div>
        {person.probabilities.map(option => <div className="prob-row" key={option.option_id}>
          <div className="prob-label"><span>{option.option_name}</span><b>{option.probability_percent.toFixed(1)}%</b></div>
          <ScoreBar value={option.probability_percent} max={100}/>
          <small>Utility score: {option.utility_score.toFixed(2)}</small>
        </div>)}
        <div className="callout"><b>Most likely:</b> {person.most_likely_choice} <span className="prob-pill">{person.most_likely_choice_probability_percent.toFixed(1)}%</span></div>
      </section>)}
    </div>
    <section className="panel">
      <h3>Compare personal choices with Stage 1</h3>
      {result.stage3.comparisons.map(comparison => <div className="comparison" key={comparison.decision_maker}>
        <div className="comparison-icon">{comparison.matches_stage1 ? "✓" : "↔"}</div>
        <div><b>{comparison.decision_maker}</b><p>{comparison.explanation}</p></div>
        <span className={comparison.matches_stage1 ? "match" : "different"}>{comparison.matches_stage1 ? "ALIGNED" : "DIFFERS"}</span>
      </div>)}
      <p className="disclaimer">{result.stage3.interpretation} Do not treat these estimates as real-world likelihoods until calibrated against historical choices.</p>
    </section>
    <section className="panel training-panel">
      <div className="section-heading"><div><div className="eyebrow">SYNTHETIC TRAINING EVALUATION</div><h2>Recorded choices vs. model output</h2></div><div className="badge">{trainingEvaluation?.sample_count ?? 0} RECORDS</div></div>
      <p className="intro">These descriptive metrics compare model top choices and probabilities with the synthetic recorded decisions. They are workflow/test metrics, not validated estimates of real people.</p>
      <div className="twin-summary"><div><span className="eyebrow">TOP-CHOICE MATCH</span><b>{trainingEvaluation?.top_choice_accuracy_percent ?? "—"}%</b></div><div><span className="eyebrow">MEAN MULTICLASS BRIER SCORE</span><b>{trainingEvaluation?.mean_multiclass_brier_score ?? "—"}</b></div><div><span className="eyebrow">SCOPE</span><b>All 8 synthetic scenarios</b></div></div>
      {trainingEvaluation?.interpretation && <p className="disclaimer">{trainingEvaluation.interpretation}</p>}
      <h3>Decision records for {result.scenario.id}</h3>
      {scenarioDisagreement && <p className="hint disagreement-summary">Recorded decisions selected <b>{scenarioDisagreement.distinct_recorded_choices}</b> distinct alternatives, producing <b>{scenarioDisagreement.pairwise_disagreement_count}</b> differing decision-maker pairs out of {scenarioDisagreement.decision_count * (scenarioDisagreement.decision_count - 1) / 2}.</p>}
      <div className="training-table-wrap"><table className="training-table"><thead><tr><th>Decision-maker</th><th>Predicted choice</th><th>Recorded choice</th><th>Recorded-choice probability</th><th>Observed outcome</th></tr></thead><tbody>
        {scenarioRecords.map(record => <tr key={record.decision_id}><td>{record.decision_maker_name}</td><td>{record.predicted_choice}</td><td>{record.recorded_choice}<small>{record.rationale}</small></td><td>{record.recorded_choice_probability_percent.toFixed(1)}%<small>{record.prediction_matches_recorded ? "Top choice matched" : "Different from top choice"}</small></td><td>{record.outcome.status}<small>{record.outcome.actual_delivery_days} days · {record.outcome.stockout_units} stockout units</small></td></tr>)}
      </tbody></table></div>
    </section>
  </>;
}

createRoot(document.getElementById("root")).render(<App />);
