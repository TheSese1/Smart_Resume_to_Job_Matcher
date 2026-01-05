from typing import List, Dict, Any
import numpy as np

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def score_resumes_for_job(
    job_embedding: List[float],
    resume_embeddings: Dict[str, List[float]]
) -> Dict[str, float]:
    scores = {}

    for resume_id, resume_vector in resume_embeddings.items():
        scores[resume_id] = cosine_similarity(job_embedding, resume_vector)

    return scores

def rank_matches(scores: Dict[str, float], top_k: int = 5):
    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

def match_job_to_resumes(
    job_id: str,
    job_embedding: List[float],
    resume_embeddings: Dict[str, List[float]],
    top_k: int = 5
):
    scores = score_resumes_for_job(job_embedding, resume_embeddings)
    ranked = rank_matches(scores, top_k=top_k)

    results = []

    for resume_id, score in ranked:
        results.append({
            "job_id": job_id,
            "resume_id": resume_id,
            "score": score
        })

    return results
