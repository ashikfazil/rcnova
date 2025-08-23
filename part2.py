# First, ensure you have all the necessary libraries:
# pip install -U transformers sentence-transformers torch pandas openpyxl accelerate

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------
# SETUP: Load Models and Data
# -----------------------------
print("Loading models and data...")

# 1. Load the model for semantic search (the retriever)
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load the LLM for reasoning and selection (the reranker/selector)
# We use a smaller, instruction-tuned model which is great for this kind of task.
# Flan-T5 is excellent at following instructions like "choose the best option".
selector_llm = pipeline("text2text-generation", model="google/flan-t5-base")

# 3. Load your Excel data
try:
    df = pd.read_excel("udemy_software_courses_100_new.xlsx")
    course_titles = df["title"].astype(str).tolist()
    print(f"Successfully loaded {len(course_titles)} course titles.")
except FileNotFoundError:
    print("Error: The file 'udemy_software_courses_100_new.xlsx' was not found.")
    exit()

# 4. Pre-compute embeddings for all course titles (this only happens once)
print("Creating embeddings for course titles...")
corpus_embeddings = retriever_model.encode(course_titles, convert_to_tensor=True)
print("Setup complete.")

# -----------------------------
# The Main Recommendation Function
# -----------------------------
def recommend_course_with_llm(prompt: str, top_k: int = 5):
    """
    Finds the best course using a two-stage retrieval and LLM selection process.
    """
    print(f"\n----- New Request -----")
    print(f"User prompt: '{prompt}'")

    # --- STAGE 1: RETRIEVAL ---
    # Find the top_k most similar courses using semantic search.
    prompt_embedding = retriever_model.encode(prompt, convert_to_tensor=True)
    cosine_scores = util.cos_sim(prompt_embedding, corpus_embeddings)

    # Get the top_k scores and indices
    top_results = torch.topk(cosine_scores, k=top_k)
    
    candidate_indices = top_results.indices[0].tolist()
    candidate_courses = [course_titles[i] for i in candidate_indices]

    print(f"\n[Stage 1] Top {top_k} candidates found via semantic search:")
    for i, course in enumerate(candidate_courses):
        print(f"  {i+1}. {course}")

    # --- STAGE 2: LLM SELECTION ---
    # Prepare the prompt for the Large Language Model.
    # We give it clear instructions and the list of candidates.
    llm_prompt = f"""
    The user wants a course about: "{prompt}"

    From the following list of course titles, which is the single best match?
    
    Choices:
    - {candidate_courses[0]}
    - {candidate_courses[1]}
    - {candidate_courses[2]}
    - {candidate_courses[3]}
    - {candidate_courses[4]}

    Your answer must be ONLY the full, exact title of the best course.
    """

    print("\n[Stage 2] Asking LLM to select the best option...")
    
    # Get the LLM's response
    llm_response = selector_llm(llm_prompt, max_length=100, clean_up_tokenization_spaces=True)
    
    # The output is a list of dictionaries, we extract the generated text
    final_choice = llm_response[0]['generated_text'].strip()

    print("\n----- Recommendation -----")
    print(f"Final Recommended Course: {final_choice}")
    print("--------------------------")
    return final_choice


# --- EXAMPLES ---
recommend_course_with_llm("I want a complete guide to web development with React")
