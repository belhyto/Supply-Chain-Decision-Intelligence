
-- 1. Decision makers
CREATE TABLE decision_makers (
    decision_maker_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Situations
CREATE TABLE situations (
    situation_id SERIAL PRIMARY KEY,
    scenario_name VARCHAR(200),
    description TEXT,
    situation_parameters JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Available alternatives
CREATE TABLE alternatives (
    alternative_id SERIAL PRIMARY KEY,
    situation_id INTEGER REFERENCES situations(situation_id),
    alternative_name VARCHAR(150) NOT NULL,
    attributes JSONB
);

-- 4. Preference snapshots
CREATE TABLE preferences (
    preference_id SERIAL PRIMARY KEY,
    decision_maker_id INTEGER REFERENCES decision_makers(decision_maker_id),
    situation_id INTEGER REFERENCES situations(situation_id),
    cost_weight NUMERIC(5,2),
    reliability_weight NUMERIC(5,2),
    speed_weight NUMERIC(5,2),
    risk_weight NUMERIC(5,2),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Suitability assessments
CREATE TABLE assessments (
    assessment_id SERIAL PRIMARY KEY,
    situation_id INTEGER REFERENCES situations(situation_id),
    alternative_id INTEGER REFERENCES alternatives(alternative_id),
    suitability_score NUMERIC(6,2),
    assessment_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Predicted choices
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    situation_id INTEGER REFERENCES situations(situation_id),
    decision_maker_id INTEGER REFERENCES decision_makers(decision_maker_id),
    alternative_id INTEGER REFERENCES alternatives(alternative_id),
    predicted_probability NUMERIC(7,6),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Actual human decisions
CREATE TABLE decisions (
    decision_id SERIAL PRIMARY KEY,
    situation_id INTEGER REFERENCES situations(situation_id),
    decision_maker_id INTEGER REFERENCES decision_makers(decision_maker_id),
    selected_alternative_id INTEGER REFERENCES alternatives(alternative_id),
    rationale TEXT,
    decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Operational outcomes
CREATE TABLE outcomes (
    outcome_id SERIAL PRIMARY KEY,
    decision_id INTEGER REFERENCES decisions(decision_id),
    actual_cost NUMERIC(14,2),
    actual_delivery_days NUMERIC(8,2),
    quality_score NUMERIC(5,2),
    outcome_description TEXT,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
