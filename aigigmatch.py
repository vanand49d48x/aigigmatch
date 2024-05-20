import os
import gradio as gr
import google.generativeai as genai
import psycopg2
import json
import logging
import time
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Google AI model
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model with predefined configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# In-memory cache for profiles
profiles_cache = {
    "data": None,
    "timestamp": None,
    "cache_duration": 60 * 5  # Cache duration in seconds (5 minutes)
}

def get_database_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL is not set in environment variables.")
        raise ValueError("DATABASE_URL is not set in environment variables.")
    try:
        return psycopg2.connect(database_url)
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        raise

def fetch_profiles():
    global profiles_cache
    current_time = time.time()
    
    # Check if cache is valid
    if profiles_cache["data"] and (current_time - profiles_cache["timestamp"] < profiles_cache["cache_duration"]):
        logging.info("Using cached profiles data.")
        return profiles_cache["data"]

    logging.info("Fetching profiles from database.")
    conn = get_database_connection()
    profiles = []
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT name, about, skills, rating, trust_score, ninja_level, task_experience, online_status FROM gig_workers")
            for row in curs.fetchall():
                skills = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                profile = {
                    "Name": row[0],
                    "About": row[1],
                    "Skills": skills,
                    "Rating": row[3],
                    "Trust Score": row[4],
                    "Ninja Level": row[5],
                    "Task Experience": row[6],
                    "Online Status": row[7]
                }
                profiles.append(profile)
    except Exception as e:
        logging.error(f"Error fetching profiles: {e}")
        raise
    finally:
        conn.close()
    
    # Update cache
    profiles_cache["data"] = profiles
    profiles_cache["timestamp"] = current_time
    
    return profiles

def remove_markdown_bold(text):
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def ask_model_questions(task_description, history=None):
    if history is None:
        history = []

    chat_session = model.start_chat(history=history)

    prompt = (f"As an expert in the field related to the task description: '{task_description}', "
              "what additional information do you need to better understand the task requirements? "
              "Please provide your questions in the following format:\n"
              "Question 1: [Your question here]?\n"
              "Question 2: [Your question here]?\n"
              "...\n"
              "If you have enough information, please type 'Enough information'.")

    response = chat_session.send_message(prompt)
    history.append({"prompt": prompt, "response": response.text})

    if "enough information" in response.text.lower():
        return None, history

    return remove_markdown_bold(response.text), history

def refine_task_description(task_description, question_answers, history=None):
    if history is None:
        history = []

    chat_session = model.start_chat(history=history)

    prompt = (f"Original task description: {task_description}\n\n"
              f"Additional information provided:\n{question_answers}\n\n"
              "Based on the original task description and the additional information provided, "
              "please provide a complete and refined task description.")

    response = chat_session.send_message(prompt)
    history.append({"prompt": prompt, "response": response.text})

    return remove_markdown_bold(response.text), history

def find_best_profile(task_description):
    profiles = fetch_profiles()
    chat_session = model.start_chat()
    
    prompt = (f"Based on the complete task description: {task_description}\n\n"
              "Evaluate the suitability of the following gig worker profiles "
              "and provide the name of the best-matched profile along with an explanation in the following format:\n"
              "Name: [Best-matched profile name]\n"
              "Why [Best-matched profile name]? [Explanation]\n\nProfiles:\n")
    for i, profile in enumerate(profiles, start=1): 
        prompt += (f"{i}. Name: {profile['Name']}, Skills: {', '.join(profile['Skills'])}, Experience: {profile['Task Experience']} hours, "
                   f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Online Status: {profile['Online Status']}\n")

    response = chat_session.send_message(prompt)
    return remove_markdown_bold(response.text)

def gradio_interface():
    with gr.Blocks(title="Interactive AI for Gig Worker Matching", css=".gradio-container {background-color: #f4f4f4}") as iface:
        gr.Markdown("Enter a task description and interact with the AI. Provide additional details if prompted by the AI for more information.")
        
        task_description = gr.Textbox(label="Enter your task description")
        
        with gr.Column(visible=False) as question_column:
            model_questions = gr.Textbox(label="Model Questions", interactive=False)
            question_answers = gr.Textbox(label="Your Answer", placeholder="Type your answer here...")
            question_submit_button = gr.Button(value="Submit", variant="primary")
        
        refined_task_description = gr.Textbox(label="Refined Task Description", visible=False)
        find_button = gr.Button(value="Find", visible=False)
        best_matched_profile = gr.Textbox(label="Best Matched Profile", visible=False)
        reset_button = gr.Button(value="Reset", variant="secondary")
    
        question_history = gr.State([])
        refine_history = gr.State([])
         
        def process_task_questions(task_description, question_history):
            questions, updated_history = ask_model_questions(task_description, question_history)
            
            if questions:
                return gr.update(value=questions, visible=True), gr.update(visible=True), gr.update(visible=True), updated_history
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), updated_history
                
        def process_task_refinement(task_description, question_answers, refine_history):
            refined_description, updated_history = refine_task_description(task_description, question_answers, refine_history)
            return gr.update(value=refined_description, visible=True), gr.update(visible=True), updated_history
                
        def find_best_profile_event(complete_task_description):
            best_profile = find_best_profile(complete_task_description)
            return gr.update(value=best_profile, visible=True)
                
        def reset_interface():
            return (
                "",
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                [],
                []
            )

        task_description.submit(process_task_questions, [task_description, question_history], [model_questions, question_column, question_history])
        question_submit_button.click(process_task_refinement, [task_description, question_answers, refine_history], [refined_task_description, find_button, refine_history])
        find_button.click(find_best_profile_event, inputs=refined_task_description, outputs=best_matched_profile)
        reset_button.click(reset_interface, None, [task_description, model_questions, question_column, refined_task_description, find_button, best_matched_profile, question_history, refine_history])
    
    # Add custom CSS styles to remove bold text in text fields
    iface.css += """
        .gradio-interface .gradio-textbox {
            font-weight: normal;
        }
    """
    
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=True)


if __name__ == "__main__":
    gradio_interface()
