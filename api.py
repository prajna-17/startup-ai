from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import shap
from groq import Groq
import os
from dotenv import load_dotenv
import torch

torch.set_num_threads(1)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/huggingface"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global models
# -----------------------------

clf = None
embedding_model = None
explainer = None
startup_embeddings = None

startup_db = {
    "Uber": "ride sharing platform connecting drivers and passengers",
    "Airbnb": "platform for renting homes and travel accommodation",
    "Swiggy": "online food delivery platform connecting restaurants and customers",
    "Stripe": "online payment processing platform for internet businesses",
    "Notion": "productivity and workspace management software",
    "Spotify": "music streaming platform with personalized recommendations",
    "Zomato": "restaurant discovery and food delivery platform",
    "Amazon": "large scale ecommerce marketplace"
}

startup_names = list(startup_db.keys())
startup_desc = list(startup_db.values())

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Lazy model loading
# -----------------------------

def load_models():
    global clf, embedding_model, explainer, startup_embeddings

    if clf is None:
        clf = joblib.load("model/startup_model.pkl")

    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if explainer is None:
        explainer = shap.TreeExplainer(clf)

    if startup_embeddings is None:
        startup_embeddings = embedding_model.encode(startup_desc)


# -----------------------------
# Models
# -----------------------------

class StartupInput(BaseModel):
    idea: str
    budget: float
    team_size: int
    timeline: int


class CompareInput(BaseModel):
    idea_a: str
    budget_a: float
    team_a: int
    timeline_a: int

    idea_b: str
    budget_b: float
    team_b: int
    timeline_b: int


@app.get("/")
def home():
    return {"message": "Startup AI API running"}


@app.post("/analyze")
def analyze_startup(data: StartupInput):

    load_models()

    embedding = embedding_model.encode(data.idea, convert_to_numpy=True)

    numeric = np.array([data.budget, data.team_size, data.timeline])

    features = np.concatenate([embedding, numeric])

    prob = clf.predict_proba([features])[0][1]

    probability = 0.6 * prob + 0.4 * 0.5

    similarities = cosine_similarity(
        embedding.reshape(1, -1),
        startup_embeddings
    )[0]

    top_idx = similarities.argsort()[-3:][::-1]

    similar = [
        {
            "name": startup_names[i],
            "score": round(float(similarities[i]) * 100, 1),
            "description": startup_desc[i]
        }
        for i in top_idx
    ]

    shap_values = explainer.shap_values(np.array([features]))

    if isinstance(shap_values, list):
        shap_array = shap_values[1].flatten()
    else:
        shap_array = shap_values.flatten()

    feature_names = [f"text_{i}" for i in range(384)] + ["budget","team_size","timeline"]

    shap_array = shap_array[:len(feature_names)]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "value": shap_array
    })

    top_shap = shap_df[
        shap_df["feature"].isin(["budget", "team_size", "timeline"])
    ]

    top_features = [
        {
            "feature": row.feature,
            "value": round(float(row.value)*1000, 3)
        }
        for _, row in top_shap.iterrows()
    ]

    structured = shap_array[-3:]

    shap_structured = {
        "budget": round(float(structured[0]) * 1000, 3),
        "team_size": round(float(structured[1]) * 1000, 3),
        "timeline": round(float(structured[2]) * 1000, 3)
    }

    def interpret_shap(value):
        if abs(value) < 0.00002:
            return "very small impact"
        elif value > 0:
            return "positive impact"
        else:
            return "negative impact"

    human_explanation = {
        "budget": interpret_shap(shap_structured["budget"]),
        "team_size": interpret_shap(shap_structured["team_size"]),
        "timeline": interpret_shap(shap_structured["timeline"])
    }

    prompt = f"""
You are a startup strategy analyst.

Startup idea:
{data.idea}

Closest similar startup:
{similar[0]["name"]}

Description:
{similar[0]["description"]}

Budget: {data.budget}
Team Size: {data.team_size}
Timeline: {data.timeline}

Success probability: {round(probability*100,2)}%

Feature impact (interpreted):

Budget: {human_explanation["budget"]}
Team Size: {human_explanation["team_size"]}
Timeline: {human_explanation["timeline"]}

Provide a clear structured analysis with these sections:

1. Startup Idea Evaluation
2. Key Risks
3. Market Challenges
4. Practical Improvement Strategies
5. What actions could increase the success probability

Explain in simple startup language with bullet points.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=500
    )

    report = response.choices[0].message.content

    return {
        "success_probability": round(probability * 100, 2),
        "closest_startup": similar[0],
        "similar_startups": similar,
        "structured_shap": shap_structured,
        "top_shap": top_features,
        "ai_report": report
    }


@app.post("/compare")
def compare_startups(data: CompareInput):

    load_models()

    emb_a = embedding_model.encode(data.idea_a, convert_to_numpy=True)
    emb_b = embedding_model.encode(data.idea_b, convert_to_numpy=True)

    feat_a = np.concatenate([emb_a, np.array([data.budget_a, data.team_a, data.timeline_a])])
    feat_b = np.concatenate([emb_b, np.array([data.budget_b, data.team_b, data.timeline_b])])

    prob_a = clf.predict_proba([feat_a])[0][1]
    prob_b = clf.predict_proba([feat_b])[0][1]

    prob_a = 0.6 * prob_a + 0.4 * 0.5
    prob_b = 0.6 * prob_b + 0.4 * 0.5

    shap_a = explainer.shap_values(np.array([feat_a]), check_additivity=False)

    if isinstance(shap_a, list):
        shap_a = shap_a[1].flatten()
    else:
        shap_a = shap_a.flatten()

    shap_b = explainer.shap_values(np.array([feat_b]), check_additivity=False)

    if isinstance(shap_b, list):
        shap_b = shap_b[1].flatten()
    else:
        shap_b = shap_b.flatten()

    structured_a = shap_a[-3:]
    structured_b = shap_b[-3:]

    shap_structured_a = {
        "budget": round(float(structured_a[0]), 6),
        "team_size": round(float(structured_a[1]), 6),
        "timeline": round(float(structured_a[2]), 6)
    }

    shap_structured_b = {
        "budget": round(float(structured_b[0]), 6),
        "team_size": round(float(structured_b[1]), 6),
        "timeline": round(float(structured_b[2]), 6)
    }

    prompt = f"""
You are a startup strategy analyst.

Startup A:
Idea: {data.idea_a}
Success Probability: {round(prob_a*100,2)}%

Startup B:
Idea: {data.idea_b}
Success Probability: {round(prob_b*100,2)}%

Compare the two startups.
Explain:
1. Why one scores higher
2. Budget, team and timeline risks
3. Which startup structure is stronger
4. Improvements for the weaker startup
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=500
    )

    report = response.choices[0].message.content

    return {
        "prob_a": round(prob_a*100,2),
        "prob_b": round(prob_b*100,2),
        "shap_a": shap_structured_a,
        "shap_b": shap_structured_b,
        "report": report
    }