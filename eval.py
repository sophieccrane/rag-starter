"""Measure retrieval quality with recall@k on a small golden set.

recall@k = share of questions where the chunk that holds the answer shows up in the top k.
No API key needed: this only tests retrieval, not generation.

Run:  python eval.py            (compares vector vs hybrid at k=3)
      python eval.py --k 1
"""
import argparse

from ask import retrieve
from rag_common import connect

# (question, doc that should come back, section that holds the answer)
GOLDEN = [
    ("How often do we sample transformer oil?", "transformer-maintenance", "Oil sampling"),
    ("What torque for the bushing clamp bolts?", "transformer-maintenance", "Bushing gasket replacement"),
    ("Recloser keeps locking out and won't reset", "recloser-troubleshooting", "Recloser will not reset"),
    ("PX-4471-C", "recloser-troubleshooting", "Replacing the control board"),
    ("PX-4417-B", "meter-replacement", "Meter base adapter"),
    ("PX-4471-D", "substation-battery", "Charger module replacement"),
    ("Which customers get power back first after a storm?", "storm-response", "Restoration priorities"),
    ("How long can a lineman work during a storm before resting?", "storm-response", "Crew rest"),
    ("What arc rating do I need for 480 volt work?", "lockout-tagout", "Arc-flash PPE"),
    ("How far should branches be from the primary lines?", "vegetation-clearance", "Clearance distances"),
]


def recall_at_k(conn, mode, k):
    found, misses = 0, []
    for question, doc, section in GOLDEN:
        hits = retrieve(conn, question, mode=mode, k=k)
        if any(h["doc_id"] == doc and h["section"] == section for h in hits):
            found += 1
        else:
            top = f"{hits[0]['doc_id']} > {hits[0]['section']}" if hits else "nothing"
            misses.append(f"    missed: {question!r}  (top hit: {top})")
    return found, misses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    conn = connect()
    for mode in ("vector", "hybrid"):
        found, misses = recall_at_k(conn, mode, args.k)
        print(f"{mode:>6}: recall@{args.k} = {found}/{len(GOLDEN)} = {found / len(GOLDEN):.0%}")
        for line in misses:
            print(line)


if __name__ == "__main__":
    main()
