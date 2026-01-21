import pandas as pd
import re
from pathlib import Path
from docx import Document
from pypdf import PdfReader

# ----------------------
# Load Resumes CSV
# ----------------------
def resumes_to_raw_text(csv_path: str):
    df = pd.read_csv(csv_path)
    raw_texts  = []

    for idx, row in df.iterrows():
        # Get the data from the csv file
        text = clean_text(row.get("Text", ""))
        category = clean_text(row.get("Category", ""))
        # Then we combine into predefined format
        combined_text = f"Category: {category}. Resume text: {text}"
        raw_texts.append({
            "resume_id": idx + 1,
            "text": combined_text,
        })
    return raw_texts

# We rewrite the function for file uploading
def any_resume_to_raw_text(path: str):
    """
    Convert a list of PDF/DOCX/TXT files into raw text strings.
    
    Parameters
    ----------
    path : Path or str
        List of file paths
    
    Returns
    -------
    List[str]
        List of raw extracted text, one per file
    """
    raw_texts = []

    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext == ".docx":
        text = _extract_docx(path)
    elif ext == ".txt":
        text = _extract_txt(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    text = clean_text(text)
    raw_texts.append(text)

    return raw_texts

# ----------------------
# Load Job Descriptions CSV
# ----------------------
def jobs_to_raw_text(csv_path):
    df = pd.read_csv(csv_path)
    raw_texts = []
    fields_to_combine = [
        "title",
        "location",
        "description",
        "requirements",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function"# Sales, Engeneering, ...
    ]

    for idx, row in df.iterrows():
        # Get the title first
        combined_text = ". ".join([f + ": " + clean_text(row.get(f, "")) for f in fields_to_combine])
        raw_texts.append({
            "job_id": row.get("job_id", idx + 1),
            "text": combined_text,
        })
    return raw_texts


def any_job_to_raw_text(path):
    """
    Convert a list of PDF/DOCX/TXT files into raw text strings.
    
    Parameters
    ----------
    path : Path or str
        List of file paths
    
    Returns
    -------
    List[str]
        List of raw extracted text, one per file
    """
    return any_resume_to_raw_text(path)


# ----------------------
# Helper functions
# ----------------------
def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text += extracted + "\n"
    return text

def _extract_docx(path: Path) -> str:
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def _extract_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    # remove control chars and normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text