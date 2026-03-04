import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import joblib
import random

model = SentenceTransformer('all-MiniLM-L6-v2')

X = []
y = []

for i in range(500):
    idea = random.choice([
        "AI healthcare platform",
        "Food delivery app",
        "Blockchain fintech startup",
        "Edtech learning app",
        "E-commerce fashion brand",
        "Climate tech sustainability startup",
        "Gaming mobile app"
    ])
    
    embedding = model.encode(idea)
    
    budget = random.randint(5, 120)
    team_size = random.randint(1, 25)
    timeline = random.randint(3, 36)

    features = np.concatenate([embedding, [budget, team_size, timeline]])

    # Smarter synthetic logic
    score = 0

    # Budget impact
    if budget > 50:
        score += 2
    elif budget > 25:
        score += 1

    # Team impact
    if team_size > 8:
        score += 2
    elif team_size > 3:
        score += 1

    # Timeline impact (too short = risky)
    if timeline < 6:
        score -= 2
    elif timeline < 12:
        score -= 1

    # Idea type boost
    if "AI" in idea or "fintech" in idea or "Climate" in idea:
        score += 1

    # Add randomness to simulate real-world uncertainty
    noise = random.uniform(-1, 1)
    final_score = score + noise

    success = 1 if final_score >= 3 else 0

    X.append(features)
    y.append(success)

X = np.array(X)
y = np.array(y)

clf = RandomForestClassifier()
clf.fit(X, y)

joblib.dump(clf, "model/startup_model.pkl")

print("Model trained and saved!")