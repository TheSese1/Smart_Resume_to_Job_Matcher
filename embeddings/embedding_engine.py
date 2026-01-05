from typing import Dict, Any, List
from langchain_community.embeddings import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Build the Resume embedding 
def build_resume_embedding_text(normalized_resume: Dict[str, Any]) -> str:
    return (
        f"Skills: {', '.join(normalized_resume.get('skills', []))}\n"
        f"Experience: {', '.join(normalized_resume.get('experience', []))}\n"
        f"Education: {', '.join(normalized_resume.get('education', []))}\n"
        f"Certifications: {', '.join(normalized_resume.get('certifications', []))}\n"
        f"Industries: {', '.join(normalized_resume.get('industries', []))}"
    )

# Build the Job embedding
def build_job_embedding_text(normalized_job: Dict[str, Any]) -> str:
    return (
        f"Job title: {normalized_job.get('job_title', '')}\n"
        f"Required skills: {', '.join(normalized_job.get('required_skills', []))}\n"
        f"Required experience: {normalized_job.get('required_experience', '')}\n"
        f"Required education: {normalized_job.get('required_education', '')}\n"
        f"Industry: {normalized_job.get('industry', '')}"
    )

# Resume embedding
def embed_resume(normalized_resume: Dict[str, Any]) -> List[float]:
    text = build_resume_embedding_text(normalized_resume)
    return embeddings_model.embed_query(text)

# Job embedding
def embed_job(normalized_job: Dict[str, Any]) -> List[float]:
    text = build_job_embedding_text(normalized_job)
    return embeddings_model.embed_query(text)
