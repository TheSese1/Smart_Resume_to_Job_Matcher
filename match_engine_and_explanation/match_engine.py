from typing import List, Dict, Any
import numpy as np

#First, we define the different similarity functions
## 1: Cosine similarity
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
## 2: ...


# Then , we define function to apply scoring functions to all files
## Resumes score for one job
def score_resumes_for_job(
    job_vector: List[float],
    resume_embeddings: Dict[str, List[float]],
    
) -> Dict[str, float]:
    scores = {}

    for resume_id, resume_vector in resume_embeddings.items():
        scores[resume_id] = cosine_similarity(job_vector, resume_vector)

    return scores
## Jobs score for one resume
def score_jobs_for_resume(
    resume_vector: List[float],
    jobs_embeddings: Dict[str, List[float]]
) -> Dict[str, float]:
    scores = {}

    for job_id, job_vector in jobs_embeddings.items():
        scores[job_id] = cosine_similarity(resume_vector, job_vector)

    return scores

# We can eventually rank the different files and giv the best
## Ranking function
def rank_matches(scores: Dict[str, float], top_k: int = 5):
    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
## Best resumes for one job
def match_resumes_to_job(
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
## Best jobs for one resume
def match_jobs_to_resume(
    resume_id: str,
    resume_embedding: List[float],
    job_embeddings: Dict[str, List[float]],
    top_k: int = 5
):
    scores = score_jobs_for_resume(resume_embedding, job_embeddings)
    ranked = rank_matches(scores, top_k=top_k)

    results = []

    for job_id, score in ranked:
        results.append({
            "job_id": job_id,
            "resume_id": resume_id,
            "score": score
        })

    return results