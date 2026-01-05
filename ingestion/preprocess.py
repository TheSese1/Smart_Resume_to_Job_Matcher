import pandas as pd
import re
from pathlib import Path

# ----------------------
# Helper function to clean text
# ----------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace and line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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