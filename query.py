# pip install fastapi uvicorn pymongo transformers sentence-transformers torch pandas openpyxl accelerate

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from dotenv import get_key
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from bson.errors import InvalidId
import json
from typing import List

# Helper to convert ObjectId to string for JSON serialization
def oid_to_str(v):
    if isinstance(v, ObjectId):
        return str(v)
    raise TypeError(f"Object of type {type(v).__name__} is not JSON serializable")

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI(title="Skills-to-Course Recommender API")

# Input models
class RecommendationRequest(BaseModel):
    user_id: str
    job_id: str

class JobRecommendationRequest(BaseModel):
    user_id: str

# -----------------------------
# MONGO SETUP
# -----------------------------
username = "ashik1234d"
password = "1234567812345678"
uri = f"mongodb+srv://{username}:{password}@cluster0.nyaadry.mongodb.net/?retryWrites=false&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["hirelink"]
users_collection = db["users"]
jobs_collection = db["job_postings"]

# -----------------------------
# LOAD COURSES & MODELS
# -----------------------------
df = pd.read_excel("udemy_software_courses_100_new.xlsx")
df['text'] = df['title'].astype(str) + " - " + df['description'].astype(str)
course_texts = df['text'].tolist()
course_titles = df['title'].tolist()
course_links = df['link'].tolist()

retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
selector_llm = pipeline("text2text-generation", model="google/flan-t5-base")
corpus_embeddings = retriever_model.encode(course_texts, convert_to_tensor=True)

# -----------------------------
# HELPER FUNCTIONS (Existing)
# -----------------------------
def get_user_skills(user_id: str):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return [s['skill_id'] for s in user.get("skills", [])]

def get_job_skills(job_id: str):
    try:
        job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return [s['skill'] for s in job.get("required_skills", [])]

# ... (keep your existing course recommendation functions) ...

def recommend_courses_per_skill(missing_skills: list, top_k: int = 5) -> dict:
    recommendations = {}
    for skill in missing_skills:
        prompt_text = f"User is missing the skill: {skill}"
        prompt_embedding = retriever_model.encode(prompt_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(prompt_embedding, corpus_embeddings)
        top_results = torch.topk(cosine_scores, k=top_k)
        candidate_indices = top_results.indices[0].tolist()
        top_course_index = candidate_indices[0]
        course_link = course_links[top_course_index]

        candidate_courses = [course_titles[i] for i in candidate_indices]
        candidate_texts = [course_texts[i] for i in candidate_indices]

        llm_prompt = f"""
The user is missing the skill: {skill}.
From the following list of courses, choose the most appropriate course(s) to help the user learn this skill. 
Provide a brief explanation for your choice.
Courses:
- {candidate_courses[0]}: {candidate_texts[0]}
- {candidate_courses[1]}: {candidate_texts[1]}
- {candidate_courses[2]}: {candidate_texts[2]}
- {candidate_courses[3]}: {candidate_texts[3]}
- {candidate_courses[4]}: {candidate_texts[4]}
Format: Course: <exact title>\nReason: <brief explanation>
"""
        llm_response = selector_llm(llm_prompt, max_length=512, clean_up_tokenization_spaces=True)
        recommendations[skill] = {
            "course": llm_response[0]['generated_text'].strip(),
            "link": course_link
        }
    return recommendations
    
# -----------------------------
# NEW JOB RECOMMENDATION FUNCTIONS
# -----------------------------
def filter_jobs_by_skills(user_skills: list) -> List[dict]:
    """Filters jobs from MongoDB where the user has at least one of the required skills."""
    if not user_skills:
        return []
    query = {"required_skills.skill": {"$in": user_skills}}
    # Convert cursor to list and handle ObjectId serialization
    jobs = [json.loads(json.dumps(job, default=oid_to_str)) for job in jobs_collection.find(query)]
    return jobs

def rank_jobs_with_llm(user_skills: list, candidate_jobs: List[dict], top_n: int = 5) -> List[dict]:
    """Uses an LLM to rank candidate jobs based on user skills."""
    if not candidate_jobs:
        return []

    job_summaries = []
    for job in candidate_jobs:
        skills_str = ", ".join([s['skill'] for s in job['required_skills']])
        job_summaries.append(f"Job ID {job['_id']}: {job['title']} in {job['location']} requiring {skills_str}.")

    llm_prompt = f"""
A user has the following skills: {', '.join(user_skills)}.
Based on these skills, rank the following jobs from most to least suitable.
Jobs:
{chr(10).join(f"- {summary}" for summary in job_summaries)}

Your output must be only a ranked list of Job IDs, separated by commas. For example:
68aa91ad16e4c4a61a831e92,68aa91ad16e4c4a61a831e93,68aa91ad16e4c4a61a831e94
"""
    
    llm_response = selector_llm(llm_prompt, max_length=1024, clean_up_tokenization_spaces=True)
    generated_text = llm_response[0]['generated_text']

    # Parse the LLM output to get ranked IDs
    ranked_ids = [job_id.strip() for job_id in generated_text.split(',')]

    # Create a final, ranked list of job objects
    ranked_jobs = []
    job_map = {job['_id']: job for job in candidate_jobs}
    for job_id in ranked_ids:
        if job_id in job_map:
            ranked_jobs.append(job_map[job_id])
            
    # Add any unranked jobs to the end
    for job in candidate_jobs:
        if job['_id'] not in ranked_ids:
            ranked_jobs.append(job)

    return ranked_jobs[:top_n]

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/recommend_multiple_courses")
def recommend_multiple_courses_endpoint(request: RecommendationRequest):
    user_skills = get_user_skills(request.user_id)
    job_skills = get_job_skills(request.job_id)
    missing_skills = list(set(job_skills) - set(user_skills))
    if not missing_skills:
        return {"message": "User already has all required skills for this job."}
    recommendations = recommend_courses_per_skill(missing_skills)
    return {"recommendations": recommendations, "missing_skills": missing_skills}

@app.post("/recommend_jobs")
def recommend_jobs_endpoint(request: JobRecommendationRequest):
    """
    Recommends jobs for a user based on their skills.
    """
    user_skills = get_user_skills(request.user_id)
    if not user_skills:
        raise HTTPException(status_code=404, detail="User has no skills to base recommendations on.")

    # Stage 1: Simple filtering
    candidate_jobs = filter_jobs_by_skills(user_skills)
    if not candidate_jobs:
        return {"recommended_jobs": []}

    # Stage 2: LLM-based ranking
    ranked_jobs = rank_jobs_with_llm(user_skills, candidate_jobs)

    return {"recommended_jobs": ranked_jobs}