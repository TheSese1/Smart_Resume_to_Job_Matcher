from typing import Dict, Any, List

# Normalization error detection

def detailed_error(norm_text:dict, field:str):
    """
    We will try seeing why some errors pop and 
    count the number of occurance for each type of errors
    """
    record = norm_text.get(field)
    error = ""

    if not record:
        error = "no field"
    elif all(isinstance(item, str) for item in record):
        error = "no error"
    elif any(isinstance(item, dict) for item in record):
        error = "dictionnaries in list"
    elif any(isinstance(item, list) for item in record):
        error = "lists in list"
    else:
        error = "other"
    
    return error


# Normalization fixing for one field
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


# Global fixing of the normalization errors
## For the experience field
def fix_error_experience(norm_text:dict):
    """
    Now, we will use the functions defined in the normalization agent to fix 
    the issues made by the LLM.

    Depending on the error, the fix is different, 
    so we will also use the previously define detailed_error function.
    """
    error = detailed_error(norm_text, "experience")
    if error == "no field":# We add an empty field
        norm_text["experience"] = []
    elif error == "dictionnaries in list" or error == "lists in list":
        field_value = norm_text.get("experience")
        norm_text["experience"] = coerce_to_strings_experience(field_value)
    return norm_text# Return the fixed dictionnary

## For the education field
def fix_error_education(norm_text:dict):
    """
    Now, we will use the functions defined in the normalization agent to fix 
    the issues made by the LLM.

    Depending on the error, the fix is different, 
    so we will also use the previously define detailed_error function.
    """
    error = detailed_error(norm_text, "education")
    if error == "no field":# We add an empty field
        norm_text["education"] = []
    elif error == "dictionnaries in list" or error == "lists in list":
        field_value = norm_text.get("education")
        norm_text["education"] = coerce_to_strings_education(field_value)
    return norm_text# Return the fixed dictionnary