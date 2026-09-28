# RAG starter: pgvector + Python

A small retrieval-augmented generation app over 8 **fictional** field manuals for a made-up utility. The documents are written to show the things you'll be asked about: part numbers that look alike (hybrid search), manager-only documents (permissions), and questions phrased differently from the text (embeddings).

Budget: about 90 minutes. Stop there even if something's unfinished.

## Files

| File | What it does |
|---|---|
| `schema.sql` | Table with a `vector(384)` column, an HNSW index, and a full-text column for keyword search |
| `ingest.py` | **Index time.** Reads `docs/`, chunks by `##` section, embeds, and inserts |
| `ask.py` | **Query time.** Embeds the question, retrieves (vector or hybrid), and asks Claude for a cited answer |
| `eval.py` | Measures recall@k on 10 golden questions, vector vs hybrid |
| `rag_common.py` | DB connection and embedding model |
| `docs/` | The fictional manuals. Two are `access: engineering` |

## Setup (about 15 minutes)

```bash
# 1. Postgres with pgvector, on port 5433 so it won't clash with a local Postgres
docker run -d --name rag-pg -e POSTGRES_PASSWORD=pw -p 5433:5432 pgvector/pgvector:pg17

# 2. Python environment
cd rag-starter
uv venv && source .venv/bin/activate          # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt            # or: pip install -r requirements.txt
                                              # Windows: .venv\Scripts\activate

# 3. Optional, for real answers (without it you'll see the prompt that would be sent)
export ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

```bash
python ingest.py                                              # build the index (safe to re-run)
python ask.py "How often should transformer oil be sampled?"  # vector search + answer
python ask.py "PX-4471-B" --mode vector --no-answer           # retrieval only
python ask.py "PX-4471-B" --mode hybrid --no-answer
python eval.py                                                # recall@3, vector vs hybrid
psql postgresql://postgres:pw@localhost:5433/postgres         # look inside the table
```

Useful in `psql`:

```sql
SELECT doc_id, section, access, left(content, 60) FROM chunks;
SELECT vector_dims(embedding), left(embedding::text, 80) FROM chunks LIMIT 1;  -- what an embedding looks like
```

## Step 6: add the permission filter yourself

In `ask.py`, replace the `TODO(you, step 6)` line in `VECTOR_SQL` with a `WHERE` clause so that `--role field` can't retrieve `engineering` chunks. `role` is already passed in as `%(role)s`.

Test it:

```bash
python ask.py "What relay settings need approval?" --role field --no-answer        # should show no relay-settings chunks
python ask.py "What relay settings need approval?" --role engineering --no-answer
```

<details><summary>One possible answer</summary>

```sql
WHERE %(role)s = 'engineering' OR access = 'field'
```

This filters at retrieval time, so restricted text never reaches the model. Hybrid mode would need the same filter in both CTEs.
</details>

## Things to notice (these are your interview talking points)

- **Distances.** Look at the cosine distance numbers `ask.py` prints. How far apart are a good match and a bad one?
- **Part numbers.** `PX-4471-B`, `-C` and `-D` and `PX-4417-B` live in different manuals. Does vector search keep them apart? Does hybrid? Postgres's keyword parser splits `PX-4471-B` into `px`, `-4471`, and `b`, so keyword search ranks the exact match highest because only it matches all three.
- **Recall.** `eval.py` gives you a number: "vector got X/10, hybrid got Y/10 at k=3." Try `--k 1` too.
- **Grounding.** Ask something the docs don't cover ("What's the pension policy?"). Does it say it doesn't know?
- **Filtered HNSW.** With a `WHERE` filter, an HNSW index can return fewer than k rows, because it filters after the approximate search. pgvector 0.8 added iterative index scans to help with this. That's a good detail for an interview.

## Write down (10 minutes, after you're done)

1. Three things you learned.
2. One thing that broke or surprised you.
3. What you'd do next for a real client: a larger golden set, reranking, access control on hybrid mode, tracing, cost per query.
