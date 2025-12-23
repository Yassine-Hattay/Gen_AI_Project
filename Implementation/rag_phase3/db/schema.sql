CREATE TABLE courses (
  id SERIAL PRIMARY KEY,
  name TEXT,
  domain TEXT
);

CREATE TABLE concepts (
  id SERIAL PRIMARY KEY,
  name TEXT,
  course_id INT,
  difficulty TEXT,
  prerequisites INT[]
);

CREATE TABLE resources (
  id SERIAL PRIMARY KEY,
  type TEXT,
  source TEXT,
  course_id INT
);

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  resource_id INT,
  concept_id INT,
  chunk_type TEXT,
  difficulty TEXT,
  faiss_index INT,
  text TEXT
);

