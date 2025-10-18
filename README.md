# Dream11 Team Predictor

A web application to help users create winning fantasy cricket teams for IPL matches on Dream11 using machine learning predictions. The app predicts player performance and provides an optimal 11-player team based on official fantasy rules and budget constraints.

---

## Features

- Accepts a standard JSON file for upcoming matches.
- Processes player squads and predicts fantasy points.
- Recommends an optimal 11-player team based on budget and team composition rules.
- Calculates the actual Dream XI after the match for performance comparison.
- Link for demo video https://drive.google.com/file/d/19IsIbTH9HaRVbmbRamg9k8YFq19PVwkw/view?usp=sharing

---

## Tech Stack

- **Frontend:** React.js, styled components 
- **Backend:** Python, Flask  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Data Processing:** Pandas, NumPy  
- **Optimization:** PuLP (Linear Programming Solver)  

---

## Folder Structure

- **dream11-app/**
  - **backend/**
    - `app.py` — Flask server
    - `requirements.txt` — Backend dependencies
    - **model_artifacts/**
      - `ProductUI_Model.json` — Trained XGBoost model
    - `player_match_logs.csv` — Pre-processed historical match data
    - `player_roles_by_season.csv` — Seasonal player roles
    - `player_roles_global.csv` — Global player roles
  - **frontend/**
    - **src/**
      - `App.jsx` — Main React component
    - `package.json`
    - ... (other React files)

---

## Installation & Setup

- **Backend Setup:**  
  - Navigate to the backend directory:  
    ```bash
    cd backend
    ```  
  - Install Python dependencies:  
    ```bash
    pip install -r requirements.txt
    ```  
  - Run the Flask server (default port: 5001):  
    ```bash
    python app.py
    ```  
  > Keep this terminal running while using the frontend.

- **Frontend Setup:**  
  - Open a new terminal and navigate to the frontend directory:  
    ```bash
    cd frontend
    ```  
  - Install Node.js dependencies:  
    ```bash
    npm install
    ```  
  - Run the React development server:  
    ```bash
    npm start
    ```  
  - Open your browser and go to the URL shown in the terminal (e.g., `http://localhost:5173`).

---

## Modeling Approach

- **Feature Engineering:**  
  - **Recent Form:**  
    - `fp_rolling_5`, `fp_rolling_10` — Average fantasy points over the last 5 and 10 matches  
    - `fp_ewm_5` — Exponentially weighted mean of fantasy points over last 5 matches  
  - **Player Statistics:** Rolling averages of key stats like runs, wickets, balls faced, etc.  
  - **Opponent & Venue Performance:**  
    - `fp_vs_opponent_mean` — Average fantasy points against the opponent  
    - `fp_vs_venue_mean` — Average fantasy points at the venue  
  - **Categorical Features:** Player role, opponent, and venue (one-hot encoded)  
  - **Credits:** Player credit value as a strong pre-match performance indicator  

- **Model Selection:**  
  - **Algorithm:** XGBoost Regressor   
  - **Validation:** Time-aware split (80% older data for training, 20% newer for validation)

---

## Usage

- Upload the match JSON file.  
- The backend predicts fantasy points for each player.  
- The frontend displays a recommended Dream11 XI.  
- Compare predicted vs actual Dream XI points.

---


