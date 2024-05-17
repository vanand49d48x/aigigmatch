import os
import gradio as gr
import google.generativeai as genai
import psycopg2
import json
import logging
import uuid

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

# Dictionary to hold chat sessions
chat_sessions = {}

def get_database_connection():
    try:
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            raise ValueError("DATABASE_URL is not set in environment variables.")
        return psycopg2.connect(database_url)
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        raise

def fetch_profiles():
    conn = get_database_connection()
    profiles = []
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT name, about, skills, rating, trust_score, ninja_level, task_experience, online_status FROM gig_workers")
            for row in curs.fetchall():
                try:
                    skills = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON for skills data: {e}")
                    skills = []
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
    finally:
        conn.close()
    return profiles

def ask_model(task_description, session_id):
    profiles = fetch_profiles()
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])

    chat_session = chat_sessions[session_id]
    prompt = (f"Here are the profiles of gig workers. Based on the task description '{task_description}', " 
              "evaluate their fit. Consider their skills, experience, online status, and rating. If the task description " 
              "is unclear or more details are needed for a precise match, please clearly state 'Need more information' and " 
              "ask specific questions to clarify. \n\nProfiles:\n")

    for i, profile in enumerate(profiles, start=1):
        prompt += (f"{i}. Name: {profile['Name']}, Skills: {', '.join(profile['Skills'])}, Experience: {profile['Task Experience']} hours, "
                   f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Online Status: {profile['Online Status']}\n")
    prompt += "\nList the top profile name with the best match. If unsure, state 'Need more information' followed by your questions."

    response = chat_session.send_message(prompt)
    if "need more information" in response.text.lower():
        output_text = f"{response.text}\n\nPlease provide more information on the task below and resubmit:"
    else:
        return response.text

    return output_text


def gradio_interface(task_description):
    session_id = gr.get_state("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        gr.set_state(session_id=session_id)
    
    return ask_model(task_description, session_id)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your task description or additional information here"),
    outputs=gr.Textbox(label="Model Response"),
    title="AI Gig Worker Matcher",
    description="Describe the task you need help with, and let the AI recommend the best gig worker for you. If the AI asks for more information, please provide it in the same input box.",
    theme="huggingface",
    css="""
        body { font-family: Arial, sans-serif; }
        label { font-weight: bold; color: #303F9F; }
        textarea { font-family: Courier, monospace; }
    """
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=port, share=True)
