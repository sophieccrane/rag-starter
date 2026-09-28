"""Shared pieces: database connection and the embedding model."""
import os

import psycopg
from pgvector.psycopg import register_vector

# Port 5433 so it doesn't clash with a Postgres you may already run on 5432.
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:pw@localhost:5433/postgres")

# Small, free, runs on your laptop. Downloads ~90 MB the first time.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def embed(texts):
    """Turn a list of strings into a list of 384-dim vectors (normalised, so cosine works well)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(texts, normalize_embeddings=True)


def connect():
    conn = psycopg.connect(DB_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets psycopg send/receive numpy arrays as pgvector values
    return conn
