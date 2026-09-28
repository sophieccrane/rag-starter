-- Run by ingest.py on every run (drops and rebuilds the table, so re-running is safe).
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
  id        bigserial PRIMARY KEY,
  doc_id    text NOT NULL,              -- file name, e.g. transformer-maintenance
  title     text NOT NULL,
  section   text NOT NULL,              -- the "## " heading the chunk came from
  access    text NOT NULL DEFAULT 'field',  -- who may see it: 'field' or 'engineering'
  content   text NOT NULL,
  embedding vector(384) NOT NULL,       -- all-MiniLM-L6-v2 produces 384 numbers per text
  -- keyword-search column, kept in sync by Postgres itself
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', section || ' ' || content)) STORED
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator)
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
-- Full-text index for the keyword half of hybrid search
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
