from fastapi import FastAPI, HTTPException
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from bson import ObjectId
import datetime

# --- MongoDB Connection ---
username = "ashik1234d"
password = quote_plus("1234567812345678")

uri = f"mongodb+srv://{username}:{password}@cluster0.nyaadry.mongodb.net/?retryWrites=false&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi("1"))
db = client["hirelink"]

app = FastAPI(title="Hirelink API")

# --- Recursive Serializer ---
def clean_doc(value):
    if isinstance(value, ObjectId):
        return str(value)
    elif isinstance(value, datetime.datetime):
        return value.isoformat()
    elif isinstance(value, list):
        return [clean_doc(v) for v in value]
    elif isinstance(value, dict):
        return {k: clean_doc(v) for k, v in value.items()}
    else:
        return value

def serialize_doc(doc):
    return clean_doc(doc)


@app.get("/")
def root():
    return {"message": "Hirelink API is running 🚀"}


@app.get("/employees")
def get_employees():
    employees = list(db["employees"].find())
    return [serialize_doc(emp) for emp in employees]


@app.get("/job_postings")
def get_job_postings():
    postings = list(db["job_postings"].find())
    return [serialize_doc(job) for job in postings]


@app.get("/skills")
def get_skills():
    skills = list(db["skills"].find())
    return [serialize_doc(skill) for skill in skills]
