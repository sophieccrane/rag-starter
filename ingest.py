"""INDEX TIME: read docs -> chunk by section -> embed -> store in Postgres.

Run:  python ingest.py
"""
from pathlib import Path

from rag_common import connect, embed

HERE = Path(__file__).parent
DOCS = HERE / "docs"
MAX_CHARS = 1500  # roughly 350 tokens; longer sections get split on paragraph breaks


def parse(path):
    """Split a markdown file into (front-matter dict, body text)."""
    text = path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        _, header, text = text.split("---", 2)
        for line in header.strip().splitlines():
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text


def chunk(text):
    """Structure-aware chunking: one chunk per '## ' section, split further only if too long."""
    sections, heading, lines = [], "Overview", []
    for line in text.splitlines():
        if line.startswith("## "):
            if "\n".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):  # skip the document title line
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))

    for heading, body in sections:
        buf = ""
        for para in body.split("\n\n"):
            if buf and len(buf) + len(para) > MAX_CHARS:
                yield heading, buf.strip()
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            yield heading, buf.strip()


def main():
    conn = connect()
    conn.execute((HERE / "schema.sql").read_text())

    files = sorted(DOCS.glob("*.md"))
    rows = []
    for path in files:
        meta, text = parse(path)
        for section, body in chunk(text):
            rows.append((path.stem, meta.get("title", path.stem), section, meta.get("access", "field"), body))

    # Embed the title + heading together with the body, so a short chunk keeps its context.
    vectors = embed([f"{title} - {section}\n{body}" for _, title, section, _, body in rows])

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, title, section, access, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(*row, vec) for row, vec in zip(rows, vectors)],
        )
    print(f"Indexed {len(rows)} chunks from {len(files)} documents.")


if __name__ == "__main__":
    main()
