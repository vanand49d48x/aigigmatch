import os
import gradio as gr
import google.generativeai as genai
import psycopg2
import json
import logging

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
    conn = get_database_connection()
    profiles = []
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
    conn.close()
    return profiles

def ask_model(task_description):
    profiles = fetch_profiles()
    prompt = (f"Here are the profiles of gig workers. Based on the task description '{task_description}', "
              "evaluate their fit. Consider their skills, experience, online status, and rating. "
              "If the description is insufficient, please respond with 'Need more information' followed by specific questions. \n\nProfiles:\n")
    for i, profile in enumerate(profiles, start=1):
        prompt += (f"{i}. Name: {profile['Name']}, Skills: {', '.join(profile['Skills'])}, Experience: {profile['Task Experience']} hours, "
                   f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Online Status: {profile['Online Status']}\n")
    prompt += "\nList the top profile name with best match. If unclear, respond with 'Need more information' and specify what is needed."

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

iface = gr.Interface(
    fn=ask_model,
    inputs=gr.Textbox(label="Enter your task description"),
    outputs=gr.Textbox(label="Model Response"),
    title="Google Generative AI Chat Model",
    description="Enter a task description to interact with the Google AI model. If more details are needed, the AI will clearly ask for them."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port, share=True)
