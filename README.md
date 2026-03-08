# Startup AI Analyzer

**An AI-powered system that evaluates startup ideas using Machine Learning, Semantic Similarity, and LLM-based Strategy Insights.**

The system predicts the **success probability of a startup idea**, identifies **similar existing startups**, explains predictions using **SHAP explainability**, and generates a **strategic report using an LLM**.

---

## Features

- **Predict startup success probability** using a trained ML model  
- **Compare startup ideas with real companies** using semantic embeddings  
- **Explain model decisions using SHAP explainability**  
- **Generate strategic analysis using LLM (Groq API)**  
- **REST API built using FastAPI**

---

## Tech Stack

### Backend
- FastAPI

### Machine Learning
- Scikit-learn  
- SentenceTransformers  
- SHAP  

### AI Integration
- Groq API (Llama 3)

### Data Processing
- NumPy  
- Pandas  

### Deployment / API
- Uvicorn

---

## System Architecture

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

---

## API Endpoints

### Analyze Startup

**POST** `/analyze`

#### Example Input

```json
{
  "idea": "AI powered mental health assistant",
  "budget": 50000,
  "team_size": 4,
  "timeline": 12
}
Output

Success Probability
Similar Startups
SHAP Feature Explanation
AI Strategy Report
Compare Two Startups

POST /compare

Example Input
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

Probability Comparison

Structured Feature Impact

AI-generated Comparison Report

##How to Run Locally
Clone the repository
git clone https://github.com/prajna-17/startup-ai.git
cd startup-ai
Install dependencies
pip install -r requirements.txt
Add environment variable

Create a .env file

GROQ_API_KEY=your_api_key_here
Run the API
uvicorn api:app --reload
Open API documentation
http://127.0.0.1:8000/docs
Demo

##Project Demo Video

https://drive.google.com/file/d/1NJ5M8HPws3Tku3_Wz2E8AbQBJOr2CMQH/view?usp=sharing

##Model Details

The system uses a machine learning classifier trained using startup-related features including:
semantic embedding of the idea
budget
team size
development timeline
SHAP is used to interpret the influence of these features on predictions.
Future Improvements
Expand startup dataset
Improve prediction model with additional features
Deploy scalable cloud version

Author
##Prajna
Computer Science (Data Science) Student
Interested in AI, Backend Systems, and Machine Learning Applications.
Computer Science (Data Science) Student

Interested in AI, Backend Systems, and Machine Learning Applications.
