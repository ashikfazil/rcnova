import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import pandas as pd
import dotenv

# -----------------------------
# DB CONNECTION
# -----------------------------

uri = f"mongodb+srv://{dotenv.get('MONGO_USERNAME')}:{dotenv.get('MONGO_PASSWORD')}@cluster0.nyaadry.mongodb.net/?retryWrites=false&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi("1"))
db = client["hirelink"]
users_collection = db["users"]
jobs_collection = db["job_postings"]
applications_collection = db["jobapplications"]

# -----------------------------
# MODELS
# -----------------------------
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
selector_llm = pipeline("text2text-generation", model="google/flan-t5-base")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_job(job_id):
    return jobs_collection.find_one({"_id": ObjectId(job_id)})

def get_user(user_id):
    return users_collection.find_one({"_id": ObjectId(user_id)})

def fetch_applicants(job_id):
    """Get all applicants for a given job"""
    apps = applications_collection.find({"jobId": ObjectId(job_id)})
    return [get_user(app["userId"]) for app in apps]

def score_candidate(user, job):
    """Simple skill + exp matching score"""
    user_skills = {s["skill_id"].lower(): s["experience"] for s in user.get("skills", [])}
    score, matched, missing = 0, [], []

    for req in job["required_skills"]:
        skill = req["skill"].lower()
        req_exp = req["experience"]

        if skill in user_skills:
            score += min(user_skills[skill], req_exp)  # partial credit if less exp
            matched.append(f"{skill} ({user_skills[skill]}y)")
        else:
            missing.append(f"{skill} ({req_exp}y)")

    return score, matched, missing

def rank_candidates(job, candidates, top_n=5):
    scored = []
    for user in candidates:
        score, matched, missing = score_candidate(user, job)
        scored.append({
            "name": user["name"],
            "email": user["email"],
            "age": user.get("age"),
            "year_of_exp": user.get("year_of_exp"),
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing
        })
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

def explain_ranking(job, top_candidates):
    """Use LLM to explain why these candidates fit"""
    job_title = job["title"]
    job_skills = ", ".join([f"{s['skill']} ({s['experience']}y)" for s in job["required_skills"]])

    candidate_text = "\n".join(
        [f"- {c['name']} ({c['year_of_exp']}y exp) | Matched: {', '.join(c['matched_skills'])} | Missing: {', '.join(c['missing_skills'])}"
         for c in top_candidates]
    )

    prompt = f"""
    As an HR expert, provide a concise explanation (2-3 sentences) for why the top candidate should be selected for the {job_title} role.
    Focus on the skills match and experience.
    
    Job Requirements: {job_skills}
    
    Top Candidates:
    {candidate_text}
    """

    resp = selector_llm(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Clean up the response to remove any prompt remnants
    generated_text = resp[0]["generated_text"].strip()
    
    # Remove any parts that seem to be repeating the instruction
    if "As an HR expert" in generated_text:
        parts = generated_text.split("As an HR expert")
        if len(parts) > 1:
            generated_text = parts[1].strip()
    
    if "provide a concise explanation" in generated_text:
        parts = generated_text.split("provide a concise explanation")
        if len(parts) > 1:
            generated_text = parts[1].strip()
            
    return generated_text
# -----------------------------
# STREAMLIT ADMIN PAGE WITH HSBC STYLING
# -----------------------------
st.set_page_config(page_title="HSBC Hiring Dashboard", layout="wide")

# HSBC-inspired CSS styling
st.markdown("""
<style>
    .main-header {
        color: #DB0011;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1rem 0;
        border-bottom: 2px solid #DB0011;
        margin-bottom: 2rem;
    }
    .hsbc-red {
        color: #DB0011;
    }
    .hsbc-dark {
        color: #0C1E3E;
    }
    .hsbc-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #DB0011;
    }
    .stButton button {
        background-color: #DB0011;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #B0000E;
        color: white;
    }
    .candidate-table {
        width: 100%;
        border-collapse: collapse;
    }
    .candidate-table th {
        background-color: #0C1E3E;
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    .candidate-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }
    .candidate-table tr:hover {
        background-color: #f5f5f5;
    }
    .explanation-box {
        background-color: #F8F9FA;
        border-left: 4px solid #0C1E3E;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="main-header">HSBC Hiring Intelligence Dashboard</h1>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown('<h3 class="hsbc-red">Job Selection</h3>', unsafe_allow_html=True)
    
    # Job Selector
    jobs = list(jobs_collection.find())
    job_map = {job["title"]: str(job["_id"]) for job in jobs}
    selected_job_title = st.selectbox("Select Job Posting:", list(job_map.keys()))
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected_job_title:
        job_id = job_map[selected_job_title]
        job = get_job(job_id)
        
        st.markdown(f'<h3 class="hsbc-red">{job["title"]}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p><strong>Location:</strong> {job["location"]}</p>', unsafe_allow_html=True)
        
        skills_list = [f"{s['skill']} ({s['experience']}y)" for s in job["required_skills"]]
        st.markdown("<strong>Required Skills:</strong>", unsafe_allow_html=True)
        for skill in skills_list:
            st.markdown(f"- {skill}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if selected_job_title:
        job_id = job_map[selected_job_title]
        job = get_job(job_id)
        
        st.markdown('<h3 class="hsbc-red">Candidate Ranking</h3>', unsafe_allow_html=True)
        
        # Fetch candidates
        applicants = fetch_applicants(job_id)
        if not applicants:
            st.warning("No applicants yet for this job.")
        else:
            top_n = st.slider("Number of Top Candidates to Show", 1, 10, 5, key="top_n_slider")
            ranked = rank_candidates(job, applicants, top_n)
            
            # Display as a formatted table
            st.markdown("<table class='candidate-table'><tr><th>Rank</th><th>Name</th><th>Email</th><th>Experience</th><th>Score</th><th>Matched Skills</th><th>Missing Skills</th></tr>", unsafe_allow_html=True)
            
            for i, candidate in enumerate(ranked, 1):
                st.markdown(f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{candidate['name']}</strong></td>
                    <td>{candidate['email']}</td>
                    <td>{candidate['year_of_exp']} years</td>
                    <td><strong>{candidate['score']}</strong></td>
                    <td>{', '.join(candidate['matched_skills'])}</td>
                    <td>{', '.join(candidate['missing_skills'])}</td>
                </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</table>", unsafe_allow_html=True)
            
            if st.button("Generate AI Ranking Explanation", key="explain_btn"):
                with st.spinner("Generating explanation..."):
                    explanation = explain_ranking(job, ranked).split(".")[0] + "."
                    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                    st.markdown(f'<h4 class="hsbc-dark">AI Ranking Analysis</h4><p>{explanation}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        

# If this is the main file being run
if __name__ == "__main__":
    # This block only runs when you execute the file directly
    # It won't run when Render runs the app using the startCommand
    import streamlit.web.cli as stcli
    import sys
    
    port = int(os.environ.get("PORT", 8501))
    sys.argv = ["streamlit", "run", sys.argv[0], "--server.port", str(port), "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())
