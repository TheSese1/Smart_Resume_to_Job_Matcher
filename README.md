# Smart Resume & Job Matcher

An AI-powered Resume and Job Matching application built in **Jupyter Notebook** and deployed through a **Streamlit** interface. The system leverages **Ollama**, **LangChain**, **LangGraph**, and **FastAPI** to enable semantic resume parsing, job-description analysis, and intelligent candidate–job matching.

## 🚀 Project Overview

Traditional resume screening often relies on keyword matching, missing the true context, skills, and experience behind a candidate’s profile.  
This project goes beyond simple keyword search by using **embeddings**, **semantic similarity**, and **Generative AI reasoning** to evaluate how well a candidate matches a job posting.

### ✨ Key Features

- **Resume Parsing**  
  - Supports **PDF**, **DOCX**, and **TXT** files  
  - Extracts structured fields: *skills, experience, education, certifications, interests*

- **Job Description Processing**  
  - Upload job description files or fetch descriptions from online sources  
  - Converts job requirements into structured representation

- **Semantic Embedding & Matching**  
  - Uses **Ollama embeddings** (or alternative embedding models)  
  - Generates vector embeddings for both resumes and job descriptions  
  - Computes **semantic similarity scores**  
  - Ranks job matches based on contextual relevance

- **Explainable AI Reasoning**  
  - Generates natural-language explanations for why a resume matches a job  
  - Example:  
    > “This candidate’s experience in data analytics aligns with the Python and SQL requirements of this role.”

- **Streamlit Application**  
  - Intuitive UI for uploading resumes and job descriptions  
  - Displays match scores and explanations  
  - Interactive exploration of structured resume and job data

- **FastAPI Backend (Optional)**  
  - Serves embedding endpoints  
  - Powers job-resume matching as an API for future scalability

### 🏗️ Architecture

```
User → Streamlit UI → (FastAPI backend) → LangChain + LangGraph pipeline
↓
Ollama Models (LLM + embeddings)
↓
Resume & Job Embeddings → Semantic Matching → Ranking + Explanation
```

### 📁 Project Structure

```
project/
│
├── app/
│ ├── streamlit_app.py
│ ├── api.py # FastAPI backend
│ ├── parsers.py
│ ├── embeddings.py
│ ├── match_engine.py
│ └── graph.py # LangGraph agent flow
│
├── agents/               # only if we build agentic workflows
│ ├── resume_agent.py
│ ├── job_matching_agent.py
│
├── ingestion/
│ ├── resume_loader.py # PDF/DOCX parsing
│ ├── job_loader.py # ingest job descriptions
│ ├── preprocess.py # normalize, clean text and convert raw text into structured schema
│
├── ui/                   # reusable UI components
│ ├── components.py
│ └── style.css
|
├── notebooks/            # experiments
│ └── smart_resume_matcher.ipynb
│ └── demo.ipynb
│
├── data/
│ ├── resumes/
│ └── jobs/
│
├── README.md
└── requirements.txt
```

### ▶️ How to Run

#### **1. Install Dependencies**
```
pip install -r requirements.txt
```

#### **2. Make sure Ollama is installed**
Download Ollama from: https://ollama.com  
Start the Ollama service:
``` ollama serve ```

Pull your desired model:
```
ollama pull llama3
ollama pull nomic-embed-text
```

#### **3. Run the Streamlit App**
```
streamlit run app/streamlit_app.py
```

#### **4. (Optional) Run FastAPI backend**
```
uvicorn app.api:app --reload --port 8000
```

### 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Front-end UI |
| **FastAPI** | Backend API for model inference |
| **Ollama** | Local LLM + embedding models |
| **LangChain** | Orchestration, retrieval, embedding pipelines |
| **LangGraph** | Graph-based agent workflow |
| **Python** | Core logic |
| **Jupyter Notebook** | Development & experimentation |

### 📌 Future Enhancements

- Integration with LinkedIn job scraping  
- Multi-resume batch processing  
- Recruiter dashboard  
- Fine-tuned domain-specific embedding models  
- Support for additional file formats  

### 🤝 Contributors

- Sébastien LEVESQUE 

---
