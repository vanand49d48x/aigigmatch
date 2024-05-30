import os
import gradio as gr
import google.generativeai as genai
import psycopg2
import json
import logging
import time
#import uuid
from dotenv import load_dotenv
import re
import stripe

# Load environment variables from .env file
load_dotenv()

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Stripe API
stripe.api_key = os.getenv("STRIPE_API_KEY")

# Configure the Google AI model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
            curs.execute("SELECT name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic FROM gig_workers")
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
                    "Online Status": row[7],
                    "Profile Pic": row[8]
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

def add_person(name, about, profile_type, skills=None, rating=None, level=None):
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            if profile_type == 'User':
                curs.execute("""
                    INSERT INTO users (name, about, level) VALUES (%s, %s, %s) RETURNING user_id
                """, (name, about, level))
            else:
                curs.execute("""
                    INSERT INTO ninjas (name, skills, rating) VALUES (%s, %s, %s) RETURNING ninja_id
                """, (name, json.dumps(skills.split(', ')), rating))
            profile_id = curs.fetchone()[0]
            conn.commit()
        return f"{profile_type} added with ID: {profile_id}"
    finally:
        conn.close()

# Function to fetch User or Ninja details
def fetch_details(profile_id, profile_type):
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            if profile_type == 'User':
                curs.execute("SELECT name, about, level FROM users WHERE user_id = %s", (profile_id,))
            else:
                curs.execute("SELECT name, skills, rating FROM ninjas WHERE ninja_id = %s", (profile_id,))
            details = curs.fetchone()
            return details if details else f"No {profile_type} found with the given ID."
    finally:
        conn.close()


def fetch_users():
    conn = get_database_connection()
    users = []
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT user_id, name, profile_name, about, trust_score, online_status, profile_pic, level FROM users")
            for row in curs.fetchall():
                skills = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                user = {
                    "User ID": row[0],
                    "Name": row[1],
                    "Profile Name": row[2],
                    "About": row[3],
                    "Trust Score": row[4],
                    "Online Status": row[5],
                    "Profile Pic": row[6],
                    "Level": row[7],
                    
                }
                users.append(user)
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        raise
    finally:
        conn.close()
    
    return users

def add_user(name, profile_name, about, trust_score, online_status, profile_pic, level):

    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                INSERT INTO users (name, profile_name, about,  trust_score, online_status, profile_pic, level)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING user_id
            """, (name, profile_name, about, trust_score, online_status, profile_pic, level))
            user_id = curs.fetchone()[0]
            conn.commit()
        logging.info(f"User added with ID: {user_id}")
        return f"User added with ID: {user_id}"
    except Exception as e:
        logging.error(f"Error adding user: {e}")
        conn.rollback()
        return f"Error adding user: {e}"
    finally:
        conn.close()

def user_exists(user_id):
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            return curs.fetchone() is not None
    except Exception as e:
        logging.error(f"Error checking user existence: {e}")
        return False
    finally:
        conn.close()

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
                   f"Rating: {profile['Rating']}, Trust Score: {profile['Trust Score']}, Online Status: {profile['Online Status']}, Profile Pic: {profile['Profile Pic']}\n")

    response = chat_session.send_message(prompt)
    return remove_markdown_bold(response.text)

# Backend Functions for Task and User Profile Management

def post_task(user_id, task_description, additional_info):
    if not user_id or not task_description:
        logging.error("User ID and Task Description are required")
        return "User ID and Task Description are required"
    if not user_exists(user_id):
        logging.error(f"User ID {user_id} does not exist")
        return f"User ID {user_id} does not exist"
    
    #task_id = str(uuid.uuid4())
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                INSERT INTO tasks (user_id, task_description, additional_info)
                VALUES (%s, %s, %s) RETURNING task_id
            """, (user_id, task_description, additional_info))
            task_id = curs.fetchone()[0]
            conn.commit()
        logging.info(f"Task posted with ID: {task_id}")
        return f"Task posted with ID: {task_id}"
    except Exception as e:
        logging.error(f"Error posting task: {e}")
        conn.rollback()
        return f"Error posting task: {e}"
    finally:
        conn.close()

def update_task_status(task_id, status):
    if not task_id or not status:
        logging.error("Task ID and Status are required")
        return "Task ID and Status are required"
    
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                UPDATE tasks SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE task_id = %s
            """, (status, task_id))
            conn.commit()
        logging.info(f"Task {task_id} status updated to {status}")
        return f"Task {task_id} status updated to {status}"
    except Exception as e:
        logging.error(f"Error updating task status: {e}")
        conn.rollback()
        return f"Error updating task status: {e}"
    finally:
        conn.close()

def get_task_progress(task_id):
    if not task_id:
        logging.error("Task ID is required")
        return "Task ID is required"
    
    conn = get_database_connection()
    progress = []
    try:
        with conn.cursor() as curs:
            curs.execute("""
                SELECT update_text, updated_at FROM task_progress WHERE task_id = %s ORDER BY updated_at DESC
            """, (task_id,))
            for row in curs.fetchall():
                progress.append({"update_text": row[0], "updated_at": row[1]})
    except Exception as e:
        logging.error(f"Error fetching task progress: {e}")
        return f"Error fetching task progress: {e}"
    finally:
        conn.close()
    return progress

def update_task_progress(task_id, update_text):
    if not task_id or not update_text:
        logging.error("Task ID and Progress Update Text are required")
        return "Task ID and Progress Update Text are required"
    
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                INSERT INTO task_progress (task_id, update_text) VALUES (%s, %s)
            """, (task_id, update_text))
            conn.commit()
        logging.info(f"Progress for task {task_id} updated")
        return f"Progress for task {task_id} updated"
    except Exception as e:
        logging.error(f"Error updating task progress: {e}")
        conn.rollback()
        return f"Error updating task progress: {e}"
    finally:
        conn.close()

def upload_task_result(task_id, result_file):
    if not task_id or not result_file:
        logging.error("Task ID and Result File are required")
        return "Task ID and Result File are required"

    # Save the uploaded file to a location and get its URL
    result_url = f"task_results/{task_id}.pdf"
    result_file.save(result_url)
    
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                UPDATE tasks SET result_url = %s WHERE task_id = %s
            """, (result_url, task_id))
            conn.commit()
        logging.info(f"Result for task {task_id} uploaded")
        return f"Result for task {task_id} uploaded"
    except Exception as e:
        logging.error(f"Error uploading task result: {e}")
        conn.rollback()
        return f"Error uploading task result: {e}"
    finally:
        conn.close()

def update_user_online_status(user_id, online_status):
    if not user_id:
        logging.error("User ID is required")
        return "User ID is required"
    
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            # Check if user exists
            curs.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            if curs.fetchone() is None:
                logging.error(f"User ID {user_id} does not exist")
                return f"User ID {user_id} does not exist"
            
            # Update the online status if user exists
            curs.execute("""
                UPDATE users SET online_status = %s WHERE user_id = %s
            """, (online_status, user_id))
            conn.commit()
        logging.info(f"User {user_id} online status updated to {online_status}")
        return f"User {user_id} online status updated to {online_status}"
    except Exception as e:
        logging.error(f"Error updating user online status: {e}")
        conn.rollback()
        return f"Error updating user online status: {e}"
    finally:
        conn.close()

def update_user_profile_pic(user_id, profile_pic_file):
    if not user_id or not profile_pic_file:
        logging.error("User ID and Profile Picture File are required")
        return "User ID and Profile Picture File are required"
    
    # Save the uploaded file to a location and get its URL
    profile_pic_url = f"profile_pictures/{user_id}.png"
    profile_pic_file.save(profile_pic_url)
    
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                UPDATE users SET profile_pic = %s WHERE user_id = %s
            """, (profile_pic_url, user_id))
            conn.commit()
        logging.info(f"User {user_id} profile picture updated")
        return f"User {user_id} profile picture updated"
    except Exception as e:
        logging.error(f"Error updating user profile picture: {e}")
        conn.rollback()
        return f"Error updating user profile picture: {e}"
    finally:
        conn.close()

def add_gig_worker(name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_file):
    if not name:
        logging.error("Name is required")
        return "Name is required"

    #worker_id = str(uuid.uuid4())
    #skills_array = skills.split(", ")
    profile_pic_url = f"profile_pictures/{name}.png"
    #profile_pic_file.save(profile_pic_url)

    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                INSERT INTO gig_workers (name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id 
            """, (name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_url))
            id = curs.fetchone()[0]
            conn.commit()
        logging.info(f"Gig worker added with ID: {id}")
        return f"Gig worker added with ID: {id}"
    except Exception as e:
        logging.error(f"Error adding gig worker: {e}")
        conn.rollback()
        return f"Error adding gig worker: {e}"
    finally:
        conn.close()

def update_gig_worker(worker_id, name, about, skills, rating, trust_score, ninja_level, task_experience, online_status):
    if not worker_id:
        logging.error("Worker ID is required")
        return "Worker ID is required"

    
    profile_pic_url = f"profile_pictures/{worker_id}.png"

    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("""
                UPDATE gig_workers
                SET name = %s, about = %s, skills = %s, rating = %s, trust_score = %s, ninja_level = %s, task_experience = %s, online_status = %s, profile_pic = %s
                WHERE id = %s
            """, (name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_url, worker_id))
            conn.commit()
        logging.info(f"Gig worker {worker_id} updated")
        return f"Gig worker {worker_id} updated"
    except Exception as e:
        logging.error(f"Error updating gig worker: {e}")
        conn.rollback()
        return f"Error updating gig worker: {e}"
    finally:
        conn.close()

def fetch_gig_worker(worker_id):
    if not worker_id:
        logging.error("Worker ID is required")
        return {"error": "Worker ID is required"}

    conn = get_database_connection()
    worker = None
    try:
        with conn.cursor() as curs:
            curs.execute("""
                SELECT name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic
                FROM gig_workers WHERE id = %s
            """, (worker_id,))
            row = curs.fetchone()
            if row:
                skills = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                worker = {
                    "Name": row[0],
                    "About": row[1],
                    "Skills": skills,
                    "Rating": row[3],
                    "Trust Score": row[4],
                    "Ninja Level": row[5],
                    "Task Experience": row[6],
                    "Online Status": row[7],
                    "Profile Pic": row[8]
                }
            else:
                return {"error": f"No worker found with ID {worker_id}"}
    except Exception as e:
        logging.error(f"Error fetching gig worker: {e}")
        return {"error": f"Error fetching gig worker: {e}"}
    finally:
        conn.close()
    return worker

def create_payment_session(amount, currency, success_url, cancel_url):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': currency,
                    'product_data': {
                        'name': 'Service Payment',
                    },
                    'unit_amount': int(amount * 100),  # Stripe uses the smallest currency unit
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url,
        )
        logging.info(f"Payment session created: {session['id']}")
        return session.url
    except Exception as e:
        logging.error(f"Error creating payment session: {e}")
        return f"Error creating payment session: {e}"
    
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

def fetch_user(user_id):
    logging.debug(f"Fetching User 1: {user_id}")
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            logging.debug(f"Fetching User 2: {user_id}")
            curs.execute("SELECT name FROM users WHERE user_id = %s", (user_id,))
            user = curs.fetchone()
            if user:
                logging.debug(f"Fetched User: {user}")
                return True, {"name": user[0], "user_id": user_id}
            else:
                return False, "No user found with ID: {}".format(user_id)
    finally:
        conn.close()

def fetch_ninja(ninja_id):
    conn = get_database_connection()
    try:
        with conn.cursor() as curs:
            curs.execute("SELECT name FROM ninjas WHERE ninja_id = %s", (ninja_id,))
            ninja = curs.fetchone()
            if ninja:
                return True, {"name": ninja[0], "ninja_id": ninja_id}
            else:
                return False, "No ninja found with ID: {}".format(ninja_id)
    finally:
        conn.close()
        
def login(profile_id, profile_type):
    try:
        if profile_type.lower() == 'user':
            success, details = fetch_user(profile_id)
        elif profile_type.lower() == 'ninja':
            success, details = fetch_ninja(profile_id)
        else:
            return "Invalid profile type", False

        if success:
            return "Login Successful", True
        else:
            return details, False
    except Exception as e:
        return str(e), False
    
    
def gradio_interface():
    logging.debug("Starting Gradio interface")
    with gr.Blocks(css=".gradio-container {background-color: #f4f4f4}") as iface:
        ai_tab_visibility=  gr.State(False)
        logging.debug(f"BEFORE VISIBILITY: {ai_tab_visibility.value}")
        ai_tab = gr.TabItem("AI Interaction", visible=ai_tab_visibility.value)
        with gr.Tabs():
            # Existing tabs here...
            
            login_tab = gr.TabItem("Login")
            

            with login_tab:
                profile_id_input = gr.Textbox(label="Enter your User ID")
                profile_type_input = gr.Radio(choices=["User", "Ninja"], label="Select Profile Type")
                login_button = gr.Button("Login")
                login_result = gr.Textbox(label="Login Result", interactive=False)
                ai_tab_visibility = gr.State(False)  # State to control the visibility of the AI Tab

                def update_visibility(result, visible):
                    ai_tab.visible = visible  # Update the visibility based on login success
                    return result, visible

                login_button.click(
                    fn=lambda profile_id, profile_type: update_visibility(*login(profile_id, profile_type)),
                    inputs=[profile_id_input, profile_type_input],
                    outputs=[login_result, ai_tab_visibility]
                )
                

            with ai_tab:
                
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
                    logging.debug(f"Processing task questions for task description: {task_description}")
                    questions, updated_history = ask_model_questions(task_description, question_history)
                    
                    if questions:
                        logging.debug(f"Questions from AI: {questions}")
                        return gr.update(value=questions, visible=True), gr.update(visible=True), gr.update(visible=True), updated_history
                    else:
                        logging.debug("No additional questions from AI.")
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
            
            with gr.TabItem("Task Management"):
                gr.Markdown("Manage tasks.")
        
                # Post Task
                post_task_user_id = gr.Number(label="User ID")
                post_task_description = gr.Textbox(label="Task Description")
                post_task_additional_info = gr.Textbox(label="Additional Info")
                post_task_button = gr.Button(value="Post Task")
                post_task_output = gr.Textbox(label="Post Task Output")
        
                def handle_post_task(user_id, description, additional_info):
                    return post_task(user_id, description, additional_info)
        
                post_task_button.click(handle_post_task, inputs=[post_task_user_id, post_task_description, post_task_additional_info], outputs=post_task_output)
        
                # Update Task Status
                update_task_id = gr.Number(label="Task ID")
                update_task_status_input = gr.Textbox(label="Status")
                update_task_status_button = gr.Button(value="Update Task Status")
                update_task_status_output = gr.Textbox(label="Update Task Status Output")
        
                def handle_update_task_status(task_id, status):
                    return update_task_status(task_id, status)
        
                update_task_status_button.click(handle_update_task_status, inputs=[update_task_id, update_task_status_input], outputs=update_task_status_output)
        
                # Get Task Progress
                get_progress_task_id = gr.Number(label="Task ID")
                get_progress_button = gr.Button(value="Get Task Progress")
                get_progress_output = gr.JSON(label="Task Progress Output")
        
                def handle_get_task_progress(task_id):
                    return get_task_progress(task_id)
        
                get_progress_button.click(handle_get_task_progress, inputs=get_progress_task_id, outputs=get_progress_output)
        
                # Update Task Progress
                update_progress_task_id = gr.Number(label="Task ID")
                update_progress_text = gr.Textbox(label="Update Text")
                update_progress_button = gr.Button(value="Update Task Progress")
                update_progress_output = gr.Textbox(label="Update Task Progress Output")
        
                def handle_update_task_progress(task_id, update_text):
                    return update_task_progress(task_id, update_text)
        
                update_progress_button.click(handle_update_task_progress, inputs=[update_progress_task_id, update_progress_text], outputs=update_progress_output)

                # Upload Task Result
                upload_result_task_id = gr.Number(label="Task ID")
                upload_result_file = gr.File(label="Task Result")
                upload_result_button = gr.Button(value="Upload Task Result")
                upload_result_output = gr.Textbox(label="Upload Task Result Output")
        
                def handle_upload_task_result(task_id, result_file):
                    return upload_task_result(task_id, result_file)
        
                upload_result_button.click(handle_upload_task_result, inputs=[upload_result_task_id, upload_result_file], outputs=upload_result_output)
            
            with gr.TabItem("User Management"):
                gr.Markdown("Manage user profiles.")
                

                # Add User
                level_options = ["Academy Student", "Genin Requester", "Chunin Requester", "Jonin Requester", "Kage Requester"]
                trust_score_options = list(range(1, 11))  # Generates numbers from 1 to 10
                
                # Add User
                add_user_name = gr.Textbox(label="Name")
                add_user_profile_name = gr.Textbox(label="Profile Name")
                add_user_about = gr.Textbox(label="About")
                add_user_level = gr.Dropdown(label="Level", choices=level_options)
                add_user_trust_score = gr.Dropdown(label="Trust Score", choices=trust_score_options)
                add_user_online_status = gr.Checkbox(label="Online Status")
                add_user_profile_pic = gr.File(label="Profile Picture")
                add_user_button = gr.Button(value="Add User")
                add_user_output = gr.Textbox(label="Add User Output")
        
                def handle_add_user(name, profile_name, about, trust_score, online_status, profile_pic_file, level):
                    return add_user(name, profile_name, about, trust_score, online_status, profile_pic_file, level)
        
                add_user_button.click(handle_add_user, inputs=[
                    add_user_name, add_user_profile_name, add_user_about,
                    add_user_trust_score, add_user_online_status,
                    add_user_profile_pic,add_user_level, ], outputs=add_user_output)

                # Fetch Users
                fetch_users_button = gr.Button(value="Fetch Users")
                fetch_users_output = gr.JSON(label="Fetch Users Output")

                                # Update User Online Status
                update_online_status_user_id = gr.Number(label="User ID")
                update_online_status = gr.Checkbox(label="Online Status")
                update_online_status_button = gr.Button(value="Update Online Status")
                update_online_status_output = gr.Textbox(label="Update Online Status Output")
        
                def handle_update_online_status(user_id, online_status):
                    return update_user_online_status(user_id, online_status)
        
                update_online_status_button.click(handle_update_online_status, inputs=[update_online_status_user_id, update_online_status], outputs=update_online_status_output)
        
                # Update User Profile Picture
                update_profile_pic_user_id = gr.Number(label="User ID")
                update_profile_pic_file = gr.File(label="Profile Picture")
                update_profile_pic_button = gr.Button(value="Update Profile Picture")
                update_profile_pic_output = gr.Textbox(label="Update Profile Picture Output")
        
                def handle_update_profile_pic(user_id, profile_pic_file):
                    return update_user_profile_pic(user_id, profile_pic_file)
        
                update_profile_pic_button.click(handle_update_profile_pic, inputs=[update_profile_pic_user_id, update_profile_pic_file], outputs=update_profile_pic_output)

        
                def handle_fetch_users():
                    return fetch_users()
        
                fetch_users_button.click(handle_fetch_users, outputs=fetch_users_output)

            with gr.TabItem("Gig Worker Management"):
                gr.Markdown("Manage gig worker profiles.")
                
                level_options = ["Genin", "Chunin", "Jonin", "ANBU", "Kage", "Sage"]
                rating_score_options = list(range(1, 6))
                trust_score_options = list(range(1, 11))  # Generates numbers from 1 to 10
                # Add Gig Worker
                add_worker_name = gr.Textbox(label="Name")
                add_worker_about = gr.Textbox(label="About")
                add_worker_skills = gr.Textbox(label="Skills (comma-separated)")
                add_worker_rating = gr.Dropdown(label="Rating", choices=rating_score_options)
                add_worker_trust_score = gr.Dropdown(label="Trust Score", choices=trust_score_options)
                add_worker_ninja_level = gr.Dropdown(label="Ninja Level", choices=level_options)
                add_worker_task_experience = gr.Number(label="Task Experience")
                add_worker_online_status = gr.Checkbox(label="Online Status")
                add_worker_profile_pic = gr.File(label="Profile Picture")
                add_worker_button = gr.Button(value="Add Worker")
                add_worker_output = gr.Textbox(label="Add Worker Output")
        
                def handle_add_worker(name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_file):
                    skills_list = skills.split(",") if skills else []
                    return add_gig_worker(name, about, skills_list, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_file)
        
                add_worker_button.click(handle_add_worker, inputs=[add_worker_name, add_worker_about, add_worker_skills, add_worker_rating, add_worker_trust_score, add_worker_ninja_level, add_worker_task_experience, add_worker_online_status, add_worker_profile_pic], outputs=add_worker_output)
        
                # Update Gig Worker
                update_worker_id = gr.Number(label="Worker ID")
                update_worker_name = gr.Textbox(label="Name")
                update_worker_about = gr.Textbox(label="About")
                update_worker_skills = gr.Textbox(label="Skills (comma-separated)")
                update_worker_rating = gr.Number(label="Rating")
                update_worker_trust_score = gr.Number(label="Trust Score")
                update_worker_ninja_level = gr.Number(label="Ninja Level")
                update_worker_task_experience = gr.Number(label="Task Experience")
                update_worker_online_status = gr.Checkbox(label="Online Status")
                update_worker_profile_pic = gr.File(label="Profile Picture")
                update_worker_button = gr.Button(value="Update Worker")
                update_worker_output = gr.Textbox(label="Update Worker Output")
        
                def handle_update_worker(worker_id, name, about, skills, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_file):
                    skills_list = skills.split(",") if skills else []
                    return update_gig_worker(worker_id, name, about, skills_list, rating, trust_score, ninja_level, task_experience, online_status, profile_pic_file)
        
                update_worker_button.click(handle_update_worker, inputs=[update_worker_id, update_worker_name, update_worker_about, update_worker_skills, update_worker_rating, update_worker_trust_score, update_worker_ninja_level, update_worker_task_experience, update_worker_online_status, update_worker_profile_pic], outputs=update_worker_output)
        
                # Fetch Gig Worker
                fetch_worker_id = gr.Number(label="Worker ID")
                fetch_worker_button = gr.Button(value="Fetch Worker")
                fetch_worker_output = gr.JSON(label="Fetch Worker Output")
        
                def handle_fetch_worker(worker_id):
                    return fetch_gig_worker(worker_id)
        
                fetch_worker_button.click(handle_fetch_worker, inputs=fetch_worker_id, outputs=fetch_worker_output)
            
            with gr.TabItem("Payment Management"):
                gr.Markdown("Manage payments using Stripe.")
                
                payment_amount = gr.Number(label="Amount")
                payment_currency = gr.Textbox(label="Currency", value="usd")
                payment_success_url = gr.Textbox(label="Success URL")
                payment_cancel_url = gr.Textbox(label="Cancel URL")
                create_payment_button = gr.Button(value="Create Payment Session")
                payment_session_output = gr.Textbox(label="Payment Session URL")
        
                def handle_create_payment_session(amount, currency, success_url, cancel_url):
                    return create_payment_session(amount, currency, success_url, cancel_url)
        
                create_payment_button.click(handle_create_payment_session, inputs=[payment_amount, payment_currency, payment_success_url, payment_cancel_url], outputs=payment_session_output)
    
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=True)

if __name__ == "__main__":
    gradio_interface()
