Startup AI Analyzer

An AI-powered system that evaluates startup ideas using machine learning, semantic similarity, and LLM-based strategy insights.

The system predicts the success probability of a startup idea, identifies similar existing startups, explains predictions using SHAP explainability, and generates a strategic report using an LLM.

Features

✔ Predict startup success probability using a trained ML model
✔ Compare startup ideas with real companies using semantic embeddings
✔ Explain model decisions using SHAP explainability
✔ Generate strategic analysis using LLM (Groq API)
✔ REST API built with FastAPI

⚙️ Tech Stack

Backend
FastAPI
Machine Learning
Scikit-learn
SentenceTransformers
SHAP
AI Integration
Groq API (Llama 3)
Data Processing
NumPy
Pandas
Deployment / API
Uvicorn

🏗 Architecture

Startup Idea Input
↓
Sentence Embedding Generation
↓
ML Model Prediction (Success Probability)
↓
SHAP Explainability
↓
Semantic Similarity with Known Startups
↓
LLM Strategy Report Generation

📌 API Endpoints
1️⃣ Analyze Startup

POST /analyze

Input

{
"idea": "AI powered mental health assistant",
"budget": 50000,
"team_size": 4,
"timeline": 12
}

Output

success probability

similar startups

SHAP feature explanation

AI strategy report

2️⃣ Compare Two Startups

POST /compare

Input

{
"idea_a": "AI tutoring platform",
"budget_a": 50000,
"team_a": 4,
"timeline_a": 12,

"idea_b": "food delivery subscription",
"budget_b": 80000,
"team_b": 6,
"timeline_b": 10
}

Output
probability comparison
structured feature impact
AI-generated comparison report

▶️ How to Run Locally

Clone the repository
git clone https://github.com/yourusername/startup-ai-analyzer.git
cd startup-ai-analyzer
Install dependencies
pip install -r requirements.txt
Add environment variable
Create .env
GROQ_API_KEY=your_api_key_here
Run the API
uvicorn api:app --reload

Open API docs

http://127.0.0.1:8000/docs

Demo

[Watch Project Demo](https://drive.google.com/file/d/1NJ5M8HPws3Tku3_Wz2E8AbQBJOr2CMQH/view?usp=sharing)

📊 Model Details

The system uses a machine learning classifier trained using startup-related features including:
semantic embedding of the idea
budget
team size
development timeline
SHAP is used to interpret the influence of these features on predictions.

📌 Future Improvements

Add startup dataset expansion
Improve prediction model with more features
Deploy scalable cloud version

Author

Prajna
Computer Science (Data Science) Student
Interested in AI, Backend Systems, and Machine Learning Applications.
