import streamlit as st
from pathlib import Path
import tempfile
import sys
import json

sys.path.append(str(Path("..").resolve()))

# ---- PIELINE MODULES ----
from ingestion.preprocess import resumes_to_raw_text, jobs_to_raw_text, any_resume_to_raw_text, any_job_to_raw_text
from agents.normalization_agent import normalize_resume, normalize_job
from agents.fixing_agent import fix_error_experience, fix_error_education
from embeddings.embedding_engine import embed_resume_BGE, embed_job_BGE, embed_resume_nomic, embed_job_nomic
from embeddings.embedding_format_conversion import lists_to_id_vector_dicts
from match_engine_and_explanation.match_engine import cosine_similarity, eucl_distance, match_jobs_to_resume, match_resumes_to_job
from match_engine_and_explanation.llm_explanation import generate_match_explanation

# APP
st.set_page_config(page_title="AI Resume ↔ Job Matcher", layout="wide")

st.title("AI-Powered Resume ↔ Job Matching System")

st.markdown("""
Upload resumes and job descriptions to compute semantic matches and generate LLM explanations.
""")

match_mode = st.radio(
    "Select matching direction",
    ["Upload Resume → Match Jobs", "Upload Job → Match Resumes"]
)

embedding_selection = st.radio(
    "Select embedding",
    ["BGE embedding", "Nomic embedding"]
)

sim_function = st.radio(
    "Select similarity function",
    ["Cosine similarity", "Euclidian distance"],
    index=0
)

uploaded_file = st.file_uploader(
                    "Upload File (PDF/DOCX/TXT)"
                    , type=["pdf", "docx", "txt"]
                    )

top_k = st.slider(
                "Top K Matches", 
                min_value=1, 
                max_value=10, 
                value=5
                )

if st.button("Run Matching"):

    if not uploaded_file:
        st.warning("Please upload at least one resume or one job description.")
        st.stop()

    # ---- LOAD CHECKPOINT EMBEDDINGS ----
    if embedding_selection == "BGE embedding":
        resume_emb_path = Path("../notebooks/checkpoints/resume_embeddings_bge.json")
        job_emb_path = Path("../notebooks/checkpoints/jobs_embeddings_bge.json")
    else:
        resume_emb_path = Path("../notebooks/checkpoints/resume_embeddings_nomic.json")
        job_emb_path = Path("../notebooks/checkpoints/jobs_embeddings_nomic.json")

    with open(resume_emb_path, "r") as f:
        emb_resume = json.load(f)

    with open(job_emb_path, "r") as f:
        emb_job = json.load(f)
    
    emb_resume_dict, emb_job_dic = lists_to_id_vector_dicts(emb_resume, emb_job)

    # ---- LOAD NORMALIZED TEXT FOR LLM EXPLANATION ----
    with open(Path("../notebooks/checkpoints/normalized_resumes.json"), "r") as f:
        normalized_resumes = json.load(f)

    with open(Path("../notebooks/checkpoints/normalized_jobs.json"), "r") as f:
        normalized_jobs = json.load(f)
    
    st.write("Processing documents...")
    # --- Step 1: Save uploaded file temporarily ---
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".pdf", ".docx", ".txt"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
    else:
        st.error("Unsupported file format")
        st.stop()

    # --- Step 2: Ingestion to raw text + Normalization --- but file_path -> (PDF/DOCX/TXT)
    if match_mode == "Upload Resume → Match Jobs":
        raw_text = any_resume_to_raw_text(file_path)[0]
        norm_text = normalize_resume(raw_text)
        # Fixing potential normalization errors
        norm_text = fix_error_experience( fix_error_education(norm_text))
        # --- Step 3: Embedding ---
        st.write("Generating embeddings...")
        if embedding_selection == "BGE embedding":
            vec = embed_resume_BGE(norm_text)
        else:
            vec = embed_resume_nomic(norm_text)
    else:
        raw_text = any_job_to_raw_text(file_path)[0]
        norm_text = normalize_job(raw_text)
        # --- Step 3: Embedding ---
        st.write("Generating embeddings...")
        if embedding_selection == "BGE embedding":
            vec = embed_job_BGE(norm_text)
        else:
            vec = embed_job_nomic(norm_text)
    
    # --- Step 4: Matching (Similarity and ranking) ---
    st.subheader("Match Results")
    if sim_function == "Cosine similarity":
        similarity_func = cosine_similarity
    else:
        similarity_func = eucl_distance

    if match_mode == "Upload Resume → Match Jobs":
        matches = match_jobs_to_resume(0, 
                                        vec, 
                                        emb_job_dic, 
                                        similarity_func, 
                                        top_k)
        st.markdown(f"**Top {top_k} job matches:**")
        
        for match in matches:
            job_id, score = match["job_id"], match["score"]
            st.write(f"Job {job_id} (similarity: {score:.4f})")
            
            for job in normalized_jobs:
                if job.get('job_id') == job_id:
                    job_description = job['job_description']
                    break
            
            st.markdown(f"**Job description** :")
            for key, value in job_description.items():
                st.markdown(f"  - **{key.upper().replace('_', ' ')}** : {value}")
            
            explanation = generate_match_explanation(
                job_description,
                norm_text,
                score
            )
            with st.expander("LLM Explanation"):
                st.write(explanation)

    else:  # Match Job to Resumes
        matches = match_resumes_to_job(0, 
                                        vec, 
                                        emb_resume_dict, 
                                        similarity_func, 
                                        top_k)
        st.markdown(f"**Top {top_k} resume matches:**")
        
        for match in matches:
            resume_id, score = match["resume_id"], match["score"]
            st.write(f"Resume {resume_id} (similarity: {score:.4f})")
            
            for resume in normalized_resumes:
                if resume.get('resume_id') == resume_id:
                    resume_text = resume['norm_text']
                    break
            
            st.markdown(f"**Resume** :")
            for key, value in resume_text.items():
                if type(value) == list:
                    st.markdown(f"**{key.upper().replace('_', ' ')}**:")
                    for ele in value:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;• {ele.replace('_', ' ')}", unsafe_allow_html=True)
                   # nested = " | ".join(f"{ele.replace('_', ' ')}" for ele in value)
                   # st.markdown(nested)
                else:
                    st.markdown(f"  - **{key.upper().replace('_', ' ')}** : {value.replace('_', ' ')}")

            explanation = generate_match_explanation(
                norm_text,
                resume_text,
                score
            )
            with st.expander("LLM Explanation"):
                st.write(explanation)
