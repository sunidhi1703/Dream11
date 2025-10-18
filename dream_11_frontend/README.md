Dream11 AI Team Predictor

Project Overview

This project is a full-stack web application designed to help users create winning fantasy cricket teams for IPL matches on Dream11. It leverages a machine learning model to predict player performance based on historical data and provides an optimal 11-player team (XI) based on the official fantasy rules and budget constraints.

The application accepts a standard Cricsheet JSON file for an upcoming match, processes the player squads, predicts fantasy points for each player, and presents a "Recommended XI" that maximizes predicted points while adhering to all team composition rules. It also calculates the actual "Dream XI" after the match for performance comparison.

Live Demo Video: [Link to your 3-minute demo video will go here]

Tech Stack

Frontend: React.js, Vite, Tailwind CSS

Backend: Python, Flask

Machine Learning: Scikit-learn, XGBoost

Data Processing: Pandas, NumPy

Optimization: PuLP (Linear Programming Solver)

Folder Structure

The project is organized into a standard client-server architecture.

dream11-app/
├── backend/
│   ├── app.py                   # The Python Flask server
│   ├── requirements.txt         # Backend dependencies
│   ├── model_artifacts/
│   │   └── ProductUI_Model.json # Trained XGBoost model
│   ├── player_match_logs.csv        # Pre-processed historical match data
│   ├── player_roles_by_season.csv   # Seasonal player roles
│   └── player_roles_global.csv      # Global player roles
│
└── frontend/
    ├── src/
    │   └── App.jsx              # The main React component
    ├── package.json
    └── ... (other React files)


How to Run the Project

To run this application, you will need two terminals open: one for the backend and one for the frontend.

1. Backend Setup

First, navigate to the backend directory:

cd backend


Install the required Python libraries:

pip install -r requirements.txt


Run the Flask server. It will start on http://127.0.0.1:5001.

python app.py


Leave this terminal running.

2. Frontend Setup

Open a new terminal and navigate to the frontend directory:

cd frontend


Install the required Node.js packages:

npm install


Run the React development server. It will start on http://localhost:5173 (or another port if 5173 is busy).

npm run dev


Open your web browser and navigate to the URL provided by the command to use the application.

Modeling Approach

1. Feature Engineering

The model's performance relies heavily on a rich set of historical features, all calculated strictly using data from before a given match to prevent data leakage. Key features include:

Recent Form:

fp_rolling_5, fp_rolling_10: Average fantasy points over the last 5 and 10 matches.

fp_ewm_5: An exponentially weighted mean of fantasy points over the last 5 matches, giving more importance to the most recent games.

Player Statistics: Rolling averages for key stats like runs_rolling_5, wickets_rolling_5, balls_faced_rolling_5, etc.

Opponent-Specific Performance: fp_vs_opponent_mean - The player's historical average fantasy points against the specific opponent they are facing in the match.

Venue-Specific Performance: fp_vs_venue_mean - The player's historical average fantasy points at the specific venue where the match is being played.

Categorical Features: Player role, opponent, and venue were one-hot encoded to be used by the model.

Credits: The calculated player credit value was also used as a feature, serving as a strong pre-match indicator of expected performance.

2. Model Selection

An XGBoost Regressor was chosen for its proven performance on tabular data, its ability to handle complex relationships between features, and its built-in mechanisms to prevent overfitting (like early_stopping_rounds). The model was trained using a time-aware validation split (80% older data for training, 20% newer data for validation) to accurately simulate its predictive power on future, unseen matches.