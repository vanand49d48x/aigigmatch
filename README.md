1. Clone the repo.

2. Create .env file with the following parameters:

GEMINI_API_KEY=XXXXXXXXXXXXXXX
DATABASE_URL=postgres://default:XXXXXXXXXXXXXXXXXXXX@-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.neon.tech:5432/verceldb

PORT=8760
STRIPE_API_KEY=sk_test_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

3. In your working dir run the following commands:

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt


4. Start the app,
   python aigigmatch.py


You have to create the following :

Create database tables at the database side:

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    profile_name VARCHAR(100) UNIQUE NOT NULL,
    about TEXT,
    skills JSONB,
    rating NUMERIC(3,2),
    trust_score NUMERIC(3,2),
    ninja_level INTEGER,
    task_experience INTEGER,
    online_status BOOLEAN,
    profile_pic TEXT,
    is_pro BOOLEAN DEFAULT FALSE
);

CREATE TABLE gig_workers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    about TEXT,
    skills TEXT[],
    rating NUMERIC(3,2),
    trust_score INTEGER,
    ninja_level VARCHAR(255),
    task_experience INTEGER,
    online_status VARCHAR(255),
    profile_pic TEXT
);


CREATE TABLE tasks (
    task_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    task_description TEXT NOT NULL,
    additional_info TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE task_progress (
    progress_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(task_id),
    update_text TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Please populate data in users and gig_workers if using AI interaction.



