import os
import gradio as gr
import google.generativeai as genai
import psycopg2
import json

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
    """Establish a connection to the database using environment variables."""
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST')
    )

def fetch_profiles():
    """Fetch all worker profiles from the database."""
    conn = get_database_connection()
    profiles = []
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT name, about, skills, rating, trust_score, ninja_level, task_experience, availability FROM gig_workers")
            for row in curs.fetchall():
                # Check if the skills data is a string and convert it to a list if it is
                if isinstance(row[2], str):
                    skills = json.loads(row[2])
                else:
                    skills = row[2]  # Assuming it's already a list
                #printf(row[0])
                profile = {
                    "Name": row[0],
                    "About": row[1],
                    "Skills": skills,
                    "Rating": row[3],
                    "Trust Score": row[4],
                    "Ninja Level": row[5],
                    "Task Experience": row[6],
                    "Availability": row[7]
                }
                profiles.append(profile)
    finally:
        conn.close()
    return profiles



def ask_model(task_description):
    profiles = fetch_profiles()
    # Generate a prompt that includes all profiles and a task description
    prompt = f"Rank the following profiles based on their suitability for the task: '{task_description}'. Consider their skills, experience, availability, and rating.\n\n"
    for i, profile in enumerate(profiles, start=1):
        prompt += f"{i}. Name: {profile['Name']}, Skills: {', '.join(profile['Skills'])}, Experience: {profile['Task Experience']} hours, "
        prompt += f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Availability: {profile['Availability']}\n"
    prompt += "\nList the profile names in order of best fit to least fit for the task."

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text



# Create Gradio interface
iface = gr.Interface(
    fn=ask_model,
    inputs=gr.Textbox(label="Enter your task description"),
    outputs=gr.Textbox(label="Model Response"),
    title="Google Generative AI Chat Model",
    description="Enter a task description to interact with the Google AI model."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use Heroku's PORT environment variable or 7860 if local
    iface.launch(port=port, share=True)  # Set share=True to create a public URL
