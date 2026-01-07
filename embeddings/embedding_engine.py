from typing import Dict, Any, List
from langchain_community.embeddings import OllamaEmbeddings
import torch
from transformers import AutoTokenizer, AutoModel

# Build the Resume embedding text -> sorting the lists helps the model with consistency
def build_resume_embedding_text(normalized_resume: Dict[str, Any]) -> str:
    return (
        f"Skills: {', '.join(sorted(set(normalized_resume.get('skills', []))))}\n"
        f"Experience: {', '.join(sorted(set(normalized_resume.get('experience', []))))}\n"
        f"Education: {', '.join(sorted(set(normalized_resume.get('education', []))))}\n"
        f"Certifications: {', '.join(sorted(set(normalized_resume.get('certifications', []))))}\n"
        f"Industries: {', '.join(sorted(set(normalized_resume.get('industries', []))))}"
    )
# Build the Job embedding text
def build_job_embedding_text(normalized_job: Dict[str, Any]) -> str:
    return (
        f"Job title: {normalized_job.get('job_title', '')}\n"
        f"Required skills: {', '.join(sorted(set(normalized_job.get('required_skills', []))))}\n"
        f"Required experience: {normalized_job.get('required_experience', '')}\n"
        f"Required education: {normalized_job.get('required_education', '')}\n"
        f"Industry: {normalized_job.get('industry', '')}"
    )

## We will test two embeddings
# nomic-embed-text (mainly english): low latency or strong semantic similarity, for long documents, multilingual
# BAAI/bge-base-en-v1.5 — English embedding model with 768‑dim vectors.

## With Nomic embedding
nomic_embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)
# Resume embedding
def embed_resume_nomic(normalized_resume: Dict[str, Any]) -> List[float]:
    text = build_resume_embedding_text(normalized_resume)
    return nomic_embedding_model.embed_query(text)
# Job embedding
def embed_job_nomic(normalized_job: Dict[str, Any]) -> List[float]:
    text = build_job_embedding_text(normalized_job)
    return nomic_embedding_model.embed_query(text)


## With BAAI/bge-base-en-v1.5 : Hugging face embedding model
# Initialize BGE model
model_name = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
BGE_embedding_model = AutoModel.from_pretrained(
    model_name
)
BGE_embedding_model.eval()  # evaluation mode, disables dropout
# Simple function to embed the text
def embed_text_BGE(text: str) -> List[float]:
    # Tokenize and truncate to max length
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = BGE_embedding_model(**inputs)

    # Use the CLS token representation as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :]

    # Normalize to unit length for cosine similarity
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding[0].cpu().tolist()
# Resume embedding
def embed_resume_BGE(normalized_resume: Dict[str, Any]) -> List[float]:
    text = build_resume_embedding_text(normalized_resume)
    return embed_text_BGE(text)

# Job embedding
def embed_job_BGE(normalized_job: Dict[str, Any]) -> List[float]:
    text = build_job_embedding_text(normalized_job)
    return embed_text_BGE(text)