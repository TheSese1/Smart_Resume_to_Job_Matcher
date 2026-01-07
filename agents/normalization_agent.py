import json
import re
from typing import Dict, Any, List
from langchain_community.llms import Ollama

# Prompt Builders
## Resume JSON Prompt Builder
def build_resume_norm_prompt(resume_text: str) -> str:
    return f"""
RETURN ONLY VALID JSON.

You are an expert resume parser.

Extract the following fields and produce exactly one JSON object with keys:

- skills: list of professional skills (strings)
- experience: summary of work experience including roles and years (string)
- education: degrees and fields of study (string)
- certifications: list of professional certifications (strings)
- industries: list of industries the candidate has worked in (strings)

Output format rules:

Skills:
- Output a JSON array
- Normalize skill names (e.g. "MS Office" -> "Microsoft Office")
- Max 8 skills

Experience:
- Output a JSON array
- Each item represents ONE job role
- Split roles by employer or job title
- Format: "<Title> – <Company> (<Years>): <Main responsibilities>"
- Summarize each role in ≤ 25 words
- Maximum 4 roles

Education:
- Output a JSON array
- One item per degree or certification
- Format: "<Degree> – <Field>, <Institution> (<Year or year range>)"
- Ignore grades, exam scores, and remarks
- Maximum 3 items

Industries:
- Maximum 3 industries

STRICT RULES:
- Do NOT print any text before or after the JSON
- Do NOT include explanations, comments, or greetings
- Do NOT invent information
- Ensure the JSON is syntactically valid and complete
- Output exactly one JSON object

Resume:
{resume_text}
"""
## Job JSON Prompt Builder
def build_job_norm_prompt(job_text: str) -> str:
    return f"""
RETURN ONLY VALID JSON.

You are an expert job description parser.

Extract the following fields and produce exactly one JSON object with keys:

- job_title: specific and normalized title of the job (string)
- required_skills: list of skills explicitly or implicitly required (list of strings)
- required_experience: required experience level or years (string)
- required_education: minimum education requirement (string)
- industry: primary industry of the role (string)

Output format rules:

job_title:
- Output a single string
- Normalize the title (e.g. "Backend Ninja" -> "Backend Software Engineer")
- Exclude seniority words only if unclear; otherwise keep them (e.g. Senior, Junior)

required_skills:
- Output a JSON array
- Include only skills relevant to performing the job
- Normalize skill names (e.g. "MS Office" → "Microsoft Office")
- Max 8 skills

required_experience:
- Output a single string
- Prefer explicit years if stated (e.g. "3+ years of experience")
- Otherwise infer a reasonable level (e.g. "Entry-level", "Mid-level", "Senior")
- Do NOT invent exact years if not implied

required_education:
- Output a single string
- Use the minimum required level (e.g. "Bachelor's degree in Computer Science")
- If education is not specified, output "Not specified"

industry:
- Output a single string
- Choose the most relevant industry based on the job context
- Do NOT list multiple industries

STRICT RULES:
- Do NOT print any text before or after the JSON
- Do NOT include explanations, comments, or greetings
- Do NOT invent information
- Ensure the JSON is syntactically valid and complete
- Output exactly one JSON object

Job description:
{job_text}
"""

# Initialize Ollama LLM
llm = Ollama(
    model="llama3",
    temperature=0.0,  # critical for deterministic output
    stop=["\n}"],   # We add explicit stop tokens to force the model into our schema
    num_predict=512 # We augment prediction limit to get complete answers
)

# Helper JSON Parser for output safety
def force_close_json(text: str) -> str:
    """
    Force closes a string with a JSON like format
    to answer a problem I had with non-valid JSON outputs from LLMs.
    """
    diff = text.count("{") - text.count("}")
    return text + ("}" * max(diff, 0))

def parse_json_or_raise(llm_output: str) -> dict[str, Any]:
    """
    Attempt to parse valid JSON from the LLM response.
    Raises a ValueError if parsing fails.
    """
    # First, we force close the JSON object
    closed_llm_output = force_close_json(llm_output)
    # Then we extract the content
    match = re.search(r"\{.*\}", closed_llm_output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output. LLM output : {llm_output}")
    json_str = match.group(0)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Optionally include debugging info
        raise ValueError(f"Invalid JSON extracted from LLM response:\n{json_str}") from e

# Normalization Functions
## Resume
def normalize_resume(resume_text: str) -> Dict[str, Any]:
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        prompt = build_resume_norm_prompt(resume_text)
        raw_output = llm.invoke(prompt)
        try:
            return parse_json_or_raise(raw_output)
        except ValueError as e:
            print(f"⚠ Attempt {attempt} failed for resume. Retrying...")
            if attempt == MAX_RETRIES:
                print("❌ Failed to normalize resume after 3 attempts.")
                return None
## Job
def normalize_job(job_text: str) -> Dict[str, Any]:
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        prompt = build_job_norm_prompt(job_text)
        raw_output = llm.invoke(prompt)
        try:
            return parse_json_or_raise(raw_output)
        except ValueError as e:
            print(f"⚠ Attempt {attempt} failed for job. Retrying...")
            if attempt == MAX_RETRIES:
                print("❌ Failed to normalize job after 3 attempts.")
                return None

# Normalization fixing
## Some of the answers from the LLM are not formalized correctly.
## Here, we build a check function to verify that, and normalize if necessary

def coerce_to_strings_experience(items: List[Any]) -> List[str]:
    out: List[str] = []

    for item in items or []:
        # Case 1: already a string
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue

        # Case 2: structured experience dict
        if isinstance(item, dict):
            parts = []
            # Use canonical keys if present
            title = item.get("title") or item.get("job_title") or item.get("role")
            company = item.get("Company") or item.get("company")
            years = item.get("Years") or item.get("years")
            summary = item.get("Summary") or item.get("summary") or item.get("responsibilities")  or item.get("Main responsibilities") or item.get("description")

            if title:
                parts.append(str(title))
            if company:
                parts.append(f"at {company}")
            if years:
                parts.append(f"({years})")
            if summary:
                parts.append(f": {str(summary)}")
            
            if parts:
                out.append(" ".join(parts))

            continue

        # Case 3: experience list
        if isinstance(item, list):
            out.append(": ".join(item))
            
            continue

    return out


def coerce_to_strings_education(items: List[Any]) -> List[str]:
    out: List[str] = []

    for item in items or []:
        # Case 1: already a string (OK)
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue
        
        # Case 2: structured dict
        if isinstance(item, dict):
            parts = []

            degree = (
                item.get("Degree")
                or item.get("degree")
                or item.get("credential")
            )
            field = (
                item.get("Field")
                or item.get("field")
                or item.get("Certificate Program")
            )
            institution = (
                item.get("Institution")
                or item.get("university")
                or item.get("institution")
            )
            years_range = (
                item.get("Year")
                or item.get("year")
                or item.get("Years")
                or item.get("years")
                or item.get("Year or Year Range")
                or item.get("Year or year range")
            )
            certification = (
                item.get("Certification")
                or item.get("Certificate Completion")
                or item.get("Course")
                or item.get("Certificate")
                or item.get("certificate")
            )

            if degree:
                parts.append(str(degree))
            if field:
                parts.append(f"in {field}")
            if institution:
                parts.append(f"from {institution}")
            if years_range:
                parts.append(f"({years_range})")
            if certification:
                parts.append(f"— Certification: {certification}")

            if parts:
                out.append(" ".join(parts))

            continue

        # Case 3: list
        if isinstance(item, list):
            out.append(": ".join(item))
            
            continue

    return out