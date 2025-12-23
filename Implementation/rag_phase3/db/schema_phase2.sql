-- Learners
CREATE TABLE learners (
    id SERIAL PRIMARY KEY,
    learning_style TEXT,
    knowledge_level FLOAT,
    embedding FLOAT8[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Interactions
CREATE TABLE learner_interactions (
    id SERIAL PRIMARY KEY,
    learner_id INT REFERENCES learners(id),
    chunk_id INT REFERENCES chunks(id),
    interaction_type TEXT,
    rating INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommendation logs (XAI + evaluation)
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    learner_id INT REFERENCES learners(id),
    chunk_id INT REFERENCES chunks(id),
    content_score FLOAT,
    collaborative_score FLOAT,
    final_score FLOAT,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
