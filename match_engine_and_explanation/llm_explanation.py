from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(
    model="llama3",
    temperature=0.0,  # critical for deterministic output
    num_predict=512
)

def generate_match_explanation(
    job_description: Dict[str, Any],
    normalized_resume: Dict[str, Any],
    similarity_score: float
) -> str:
    prompt = f"""
You are an AI recruitment assistant.

Explain why the following resume is a good or poor match for the job.
Be concise, factual, and explicit.

Job description:
- Job title: {.join(job_description.get('job_title', []))}
- Skills: {', '.join(job_description.get('required_skills', []))}
- Experience: {job_description.get('required_experience', '')}
- Education: {job_description.get('required_education', '')}
- Industry: {job_description.get('industry', '')}

Candidate profile:
- Skills: {', '.join(normalized_resume.get('skills', []))}
- Experience: {normalized_resume.get('experience', '')}
- Education: {normalized_resume.get('education', '')}
- Certifications : {normalized_resume.get('certifications', '')}
- Industries: {', '.join(normalized_resume.get('industries', []))}

Similarity score: {round(similarity_score, 3)}

Explain:
- Key strengths
- Any gaps or risks
- Overall suitability
"""
    return llm.invoke(prompt)