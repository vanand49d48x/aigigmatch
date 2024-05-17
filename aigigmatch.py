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


def ask_model(task_description, additional_info="", history=None):
    if history is None:
        history = []

    profiles = fetch_profiles()
    chat_session = model.start_chat(history=history)

    while True:
        prompt = (f"Based on the task description '{task_description}', what additional information do you need to better understand the task requirements? "
                  "If you have enough information, please provide a complete task description.")

        response = chat_session.send_message(prompt)
        history.append({"prompt": prompt, "response": response.text})

        if "complete task description" in response.text.lower():
            return None, response.text, history

        return response.text, None, history

def find_best_profile(task_description):
    profiles = fetch_profiles()
    chat_session = model.start_chat()

    prompt = (f"Based on the complete task description '{task_description}', evaluate the suitability of the following gig worker profiles "
              "and provide the name of the best-matched profile.\n\nProfiles:\n")
    for i, profile in enumerate(profiles, start=1):
        prompt += (f"{i}. Name: {profile['Name']}, Skills: {', '.join(profile['Skills'])}, Experience: {profile['Task Experience']} hours, "
                   f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Online Status: {profile['Online Status']}\n")

    response = chat_session.send_message(prompt)
    return response.text

def gradio_interface():
    with gr.Blocks(title="Interactive AI for Gig Worker Matching", css=".gradio-container {background-color: #f4f4f4}") as iface:
        gr.Markdown("Enter a task description and interact with the AI. Provide additional details if prompted by the AI for more information.")
        
        with gr.Row():
            task_description = gr.Textbox(label="Enter your task description")
            additional_info = gr.Textbox(label="Additional Information", visible=False)
        
        with gr.Row():
            model_question = gr.Textbox(label="Model Question", visible=False)
            refined_task_description = gr.Textbox(label="Refined Task Description", visible=False)
        
        find_button = gr.Button(value="Find", visible=False)
        best_matched_profile = gr.Textbox(label="Best Matched Profile", visible=False)
        
        history = gr.JSON(label="History", visible=False)
        
        def process_task(task_description, additional_info, history):
            if history is None:
                history = []
            
            if additional_info:
                task_description += " " + additional_info
            
            question, complete_task_description, history = ask_model(task_description, history=history)
            
            if question:
                return gr.update(value=question), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), history
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(value=complete_task_description, visible=True), gr.update(visible=True), history
        
        def find_best_profile_event(complete_task_description):
            best_profile = find_best_profile(complete_task_description)
            return gr.update(value=best_profile, visible=True)
        
        task_description.submit(process_task, [task_description, additional_info, history], [model_question, additional_info, refined_task_description, find_button, history])
        additional_info.submit(process_task, [task_description, additional_info, history], [model_question, additional_info, refined_task_description, find_button, history])
        find_button.click(find_best_profile_event, inputs=refined_task_description, outputs=best_matched_profile)
    
    iface.launch(server_name="0.0.0.0", server_port=port, share=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    gradio_interface()
