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
from bson.objectid import ObjectId # <-- IMPORT THIS
from bson.errors import InvalidId   # <-- IMPORT THIS FOR ERROR HANDLING

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI(title="Skills-to-Course Recommender API")

# Input model
class RecommendationRequest(BaseModel):
    user_id: str
    job_id: str

# -----------------------------
# MONGO SETUP
# -----------------------------

username = "ashik1234d"
password = "1234567812345678"

uri = f"mongodb+srv://{username}:{password}@cluster0.nyaadry.mongodb.net/?retryWrites=false&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi("1"))
db = client["hirelink"]  # replace with your DB name
users_collection = db["users"]
jobs_collection = db["job_postings"]

# -----------------------------
# LOAD COURSES
# -----------------------------
df = pd.read_excel("udemy_software_courses_100_new.xlsx")
df['text'] = df['title'].astype(str) + " - " + df['description'].astype(str)
course_texts = df['text'].tolist()
course_titles = df['title'].tolist()

# -----------------------------
# LOAD MODELS
# -----------------------------
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
selector_llm = pipeline("text2text-generation", model="google/flan-t5-base")
corpus_embeddings = retriever_model.encode(course_texts, convert_to_tensor=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_user_skills(user_id: str):
    try:
        # Convert string to ObjectId before querying
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    except InvalidId:
        # Handle cases where the ID format is wrong
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return [s['skill_id'] for s in user.get("skills", [])]

def get_job_skills(job_id: str):
    try:
        # Convert string to ObjectId before querying
        job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    except InvalidId:
        # Handle cases where the ID format is wrong
        raise HTTPException(status_code=400, detail="Invalid job ID format.")
        
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return [s['skill'] for s in job.get("required_skills", [])]

def recommend_course_for_skills(missing_skills: list, top_k: int = 5) -> str:
    prompt_text = "User is missing these skills: " + ", ".join(missing_skills)

    # Stage 1: semantic search
    prompt_embedding = retriever_model.encode(prompt_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(prompt_embedding, corpus_embeddings)
    top_results = torch.topk(cosine_scores, k=top_k)

    candidate_indices = top_results.indices[0].tolist()
    candidate_courses = [course_titles[i] for i in candidate_indices]
    candidate_texts = [course_texts[i] for i in candidate_indices]

    # Stage 2: LLM selection + explanation
    llm_prompt = f"""
The user is missing these skills: {', '.join(missing_skills)}.

From the following list of courses, choose the most appropriate courses for each missing skill to help the user learn these skills.


Courses:
- {candidate_courses[0]}: {candidate_texts[0]}
- {candidate_courses[1]}: {candidate_texts[1]}
- {candidate_courses[2]}: {candidate_texts[2]}
- {candidate_courses[3]}: {candidate_texts[3]}
- {candidate_courses[4]}: {candidate_texts[4]}

Your output should be in the format:
Course: <exact title>
"""
    llm_response = selector_llm(llm_prompt, max_length=512, clean_up_tokenization_spaces=True)
    return llm_response[0]['generated_text'].strip()


# -----------------------------
# UPGRADED LLM RECOMMENDATION FUNCTION
# -----------------------------
def recommend_courses_per_skill(missing_skills: list, top_k: int = 5) -> dict:
    """
    For each missing skill, recommend one or more courses using semantic search + LLM.
    Returns a dictionary: {skill: recommendation_text}
    """
    recommendations = {}

    for skill in missing_skills:
        prompt_text = f"User is missing the skill: {skill}"

        # Stage 1: Semantic search
        prompt_embedding = retriever_model.encode(prompt_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(prompt_embedding, corpus_embeddings)
        top_results = torch.topk(cosine_scores, k=top_k)

        candidate_indices = top_results.indices[0].tolist()
        candidate_courses = [course_titles[i] for i in candidate_indices]
        candidate_texts = [course_texts[i] for i in candidate_indices]

        # Stage 2: LLM selection
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

Format:
Course: <exact title>
Reason: <brief explanation>
"""
        llm_response = selector_llm(llm_prompt, max_length=512, clean_up_tokenization_spaces=True)
        recommendations[skill] = llm_response[0]['generated_text'].strip()

    return recommendations


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/recommend_single_course")
def recommend_endpoint(request: RecommendationRequest):
    user_skills = get_user_skills(request.user_id)
    job_skills = get_job_skills(request.job_id)

    # Compute missing skills
    missing_skills = list(set(job_skills) - set(user_skills))
    if not missing_skills:
        return {"message": "User already has all required skills for this job."}

    # Recommend course via LLM
    recommendation = recommend_course_for_skills(missing_skills)
    return {"recommendation": recommendation, "missing_skills": missing_skills}



@app.post("/recommend_multiple_courses")
def recommend_endpoint(request: RecommendationRequest):
    user_skills = get_user_skills(request.user_id)
    job_skills = get_job_skills(request.job_id)

    # Compute missing skills
    missing_skills = list(set(job_skills) - set(user_skills))
    if not missing_skills:
        return {"message": "User already has all required skills for this job."}

    # Recommend courses per missing skill
    recommendations = recommend_courses_per_skill(missing_skills)
    return {"recommendations": recommendations, "missing_skills": missing_skills}

