from typing import List, Dict, Any

def lists_to_id_vector_dicts(
    embedded_resumes_list: List[Dict[str, Any]],
    embedded_jobs_list: List[Dict[str, Any]]
) -> (Dict[int, List[float]], Dict[int, List[float]]):
    """
    Convert list-of-dictionaries embeddings into id->vector dictionaries.

    Args:
        embedded_resumes_list: List of dicts with 'resume_id' and 'resume_vector'.
        embedded_jobs_list: List of dicts with 'job_id' and 'job_vector'.

    Returns:
        embedded_resumes_dict: {resume_id: resume_vector}
        embedded_jobs_dict: {job_id: job_vector}
    """

    # Convert resumes
    embedded_resumes_dict = {
        r['resume_id']: r['resume_vector']
        for r in embedded_resumes_list
        if r.get('resume_vector') is not None  # skip None embeddings
    }

    # Convert jobs
    embedded_jobs_dict = {
        j['job_id']: j['job_vector']
        for j in embedded_jobs_list
        if j.get('job_vector') is not None  # skip None embeddings
    }

    return embedded_resumes_dict, embedded_jobs_dict
