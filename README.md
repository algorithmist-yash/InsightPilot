# InsightPilot
Autonomously inspect, clean, visualize, model, and report — all with AI agents.

InsightPilot is an AI-powered multi-agent system that performs end-to-end exploratory data analysis (EDA) and baseline modeling on tabular datasets — automatically.

🚀 Features

Modular agents for each step: planning, cleaning, EDA, visualization, modeling, reporting

Automatically generates CSV reports, charts, and Markdown summary

Baseline models (Logistic Regression and Random Forest) if target variable available

Extensible architecture: easily add new agents (e.g., AutoML, data validation)

Clear logging and audit trail via agent message logs

📊 Demo (Using Titanic Dataset)

Load the Titanic dataset (e.g., train.csv) into the root directory.

Run the main pipeline: python main.py

Outputs are generated in the output/ folder:

summary_stats.csv

missing_report.csv

figures/ (PNG plots)

report.md (final insights)

model_results.json

agent_log.json

🛠️ How It Works

Planner Agent: Inspects dataset, sets workflow

Cleaning Agent: Drops/reports missing columns, identifies types

EDA Agent: Produces descriptive statistics and missing-value report

Visualization Agent: Creates histograms, correlation heatmaps, etc.

Modeling Agent: If target found, trains baselines

Reporter Agent: Combines everything into a Markdown report

💻 Tech Stack

Python 3

pandas, numpy

matplotlib, seaborn

scikit-learn

📦 Installation & Usage

Clone this repository

git clone (https://github.com/algorithmist-yash/InsightPilot/tree/main)
cd insightpilot


(Optional) Set up a virtual environment

python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows


Install requirements

pip install -r requirements.txt


Place your train.csv (e.g., from Titanic) in the root directory.

Run the pipeline

python main.py


Check the output/ directory for your results.

🔭 Future Improvements

Add an interactive dashboard (Streamlit / Dash)

Integrate AutoML / HPO for better modeling

Use persistent memory for agents to improve across runs

Support for time-series, text, or image datasets via agent extension

Agent performance evaluation and human-in-the-loop corrections

👤 Author

Yash Raj (future AI & ML Researcher)

[https://www.linkedin.com/in/yash-raj-476290369/]

📜 License

This project is licensed under the MIT License — feel free to fork and adapt!
