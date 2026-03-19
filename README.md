Traffic Signal Optimization Using Reinforcement Learning
A Q-Learning agent that controls traffic signals at a road intersection, reducing average vehicle queue length by 34.6% compared to a fixed-time controller.

Table of Contents

About
Getting Started
Usage
Results
Project Structure
Built With
Author



About
Fixed-time traffic signals run the same cycle regardless of how many cars are waiting. This project trains a Q-Learning reinforcement learning agent to observe real-time queue conditions and select optimal green phase durations — without any pre-programmed rules.
The agent is trained on 43,200 synthetic traffic records spanning 30 days across 5 intersections, with realistic morning and evening peak patterns. After 3,000 training episodes it learns a time-aware policy: longer green phases during peak hours, shorter ones at night.

Getting Started
Prerequisites

Python 3.9+
Git

Installation
bash# 1. Clone the repository
git clone https://github.com/Sudarshan-info/traffic-rl-project.git
cd traffic-rl-project

# 2. Create and activate virtual environment

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

Usage
All commands must be run from the project root with the virtual environment active.
bash# Full pipeline — generate data, train ML models, train agent, evaluate, plot charts
python main.py --episodes 3000

# Quick test run
python main.py --episodes 100

# Skip data generation if traffic_data.csv already exists
python main.py --episodes 3000 --skip-data

# Skip ML model training
python main.py --episodes 3000 --skip-ml

# Generate PDF report after training
python main.py --episodes 3000 --report
All charts are saved to results/ and all logs to logs/ after each run.

Results
Evaluated over 100 test episodes with exploration disabled (ε = 0):
ControllerMean Queue (vehicles)vs. Fixed TimerQ-Learning Agent6.7034.6% fewer vehiclesFixed Timer (30 s)10.25BaselineRandom Agent9.1111.1% fewer vehicles
ML baselines trained on the same dataset for green-time prediction:
ModelMAE (seconds)R²Random Forest4.330.731SVR (RBF)4.460.750Linear Regression6.690.610

Project Structure
traffic_rl_project/
├── data/
│   ├── __init__.py
│   ├── generate_synthetic_data.py
│   └── traffic_data.csv               # generated, not committed
├── src/
│   ├── __init__.py
│   ├── environment.py                 # MDP: state, action, reward, step
│   ├── agent.py                       # Q-table, epsilon-greedy, update rule
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Controller comparison
│   ├── ml_models.py                   # RF, SVR, Linear Regression
│   ├── visualize.py                   # Chart generation
│   ├── config.py                      # Config loader
│   ├── config_loader.py               # Singleton config cache
│   └── report.py                      # PDF report generator
├── results/                           # Output charts (PNG)
├── models/                            # Saved Q-table (.npy)
├── logs/                              # Training history (JSON)
├── reports/                           # Generated reports
├── config.json                        # All hyperparameters
├── main.py                            # Entry point
├── requirements.txt
└── .gitignore

Built With

Python 3.11
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
tqdm


Author
Sudarshan Adhikari
MS Data Science and Artificial Intelligence
South Asian University, New Delhi