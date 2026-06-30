"""Candidate generation (recall layer) for the Recommendation Service.

Public boundary for future ranking engines:

    from app.candidates import generate_candidates

CODE ONLY (Batch 7): not activated — not called from any API route, not started
by any background job, not wired into the gateway. No ranking / reranking / ML
lives here.
"""

from app.candidates.generator import generate_candidates

__all__ = ["generate_candidates"]
