# Smart Resume & Job Matcher

An AI-powered Resume and Job Matching application built in **Jupyter Notebook** and deployed through a **Streamlit** interface. The system leverages **Ollama** and **Streamlit** to enable semantic resume parsing, job-description analysis, and intelligent candidate–job matching.

## 🚀 Project Overview

Traditional resume screening often relies on keyword matching, missing the true context, skills, and experience behind a candidate’s profile.  
This project goes beyond simple keyword search by using **embeddings**, **semantic similarity**, and **Generative AI reasoning** to evaluate how well a candidate matches a job posting.

### ✨ Key Features

- **Resume Parsing**  
  - Supports **PDF**, **DOCX**, and **TXT** files  
  - Extracts structured fields: *skills, experience, education, certifications and industries*
  - Initial training being done via strings in a csv file

- **Job Description & Resume Ingestion**  
  - Uploads job description files or a resume  
  - Converts job description and resume to rax text

- **Extraction & normalization**  
  - Extracts the main information from a resume or job description via llama3
  - Normalizes the resume thanks to predefined function to fix LLM mistakes

- **Semantic Embedding & Matching**  
  - Uses **Ollama embeddings** (Nomic or BGE)  
  - Predefines texts for future embedding
  - Generates vector embeddings for both resumes and job descriptions  

- **Similarity & Ranking**
  - Computes **semantic similarity scores** (Cosine similarity or Euclidian Distance)
  - Ranks job matches based on contextual relevance

- **Explainable AI Reasoning**  
  - Generates **natural-language explanations** for why a resume matches a job, showing key stranghts and gaps
  - Example:  
    > “This candidate’s experience in data analytics aligns with the Python and SQL requirements of this role.”

- **Streamlit Application**  
  - Intuitive UI for uploading resumes and job descriptions  
  - Options to choose embedding or similarity score
  - Displays match scores and explanations

### 🏗️ Architecture

```
User → Streamlit UI → (FastAPI backend) → Agentic workflow
↓
Ollama Models (LLM + embeddings)
↓
Resume & Job Embeddings → Matching → Ranking + Explanation
```

### 📁 Project Structure

```
project/
│
├── app/# not created yet
│ └── streamlit_app.py
│
├── agents/
│ ├── normalization_agent.py           # agentic functions to create normalized prompts
│ └── fixing_agent.py                  # fiwes LLM output to fit a correct JSON schema
│
├── ingestion/
│ └── preprocess.py                    # normalize, clean text and convert raw text into structured schema
│
├── embeddings/
│ ├── embedding_engine.py              # Building and using embedding prompts
│ └── embedding_format_conversion.py   # Converting embedding results into ranking agent input format
│
├── match_engine_and_explanation/
│ ├── match_engine.py                  # Generating matching scores resume-job
│ └── llm_explanation.py               # Explanation prompt for matching
|
├── notebooks/                         # experiments
│ ├── smart_resume_matcher.ipynb       # Main notebook for transformation
│ └── eval.ipynb                       # evaluation notebook
│
├── data/
│ ├── resumes/
│ │ └── resumes.csv                    # resume texts for training
│ ├── jobs/
│ │ └── job_postings.csv               # job posting information for training
│ └── test/                            # resume and job posting examples (txt, docx, csv)
│
├── README.md
└── requirements.txt
```

### ▶️ How to Run

#### **0. Clone the project**
```
git clone https://github.com/TheSese1/Smart_Resume_to_Job_Matcher.git
cd Smart_Resume_to_Job_Matcher

# Create a python environment with correct dependencies (optional)
python -m venv .venv		
source .venv/bin/activate   # or Windows equivalent
```

#### **1. Install Dependencies**
```
pip install -r requirements.txt
```

#### **2. Make sure Ollama is installed**
Download Ollama from: https://ollama.com  
Start the Ollama service:
``` ollama serve ``` or open ollama application

Pull your desired llm and embedding (actually done inside the notebook):
```
ollama pull llama3
ollama pull nomic-embed-text
```

#### **3. Run the Streamlit App**
```
streamlit run app/streamlit_app.py
```

### 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Front-end UI |
| **Ollama** | Local LLM + embedding models |
| **Python** | Core logic |
| **Jupyter Notebook** | Development & experimentation |
| **Scikit-learn** | Metrics and representation |

### 📌 Future Enhancements

- Integration with LinkedIn job scraping
- Recruiter dashboard  
- Fine-tuned domain-specific embedding models  
- Support for additional file formats (pictures)

### 🤝 Contributors

- Sébastien LEVESQUE 

---
