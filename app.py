import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from groq import Groq
import shap
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
# -----------------------------
# SET YOUR GROQ API KEY HERE
# -----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
# -----------------------------
# Load Models
# -----------------------------

@st.cache_resource
def load_classifier():
    return joblib.load("model/startup_model.pkl")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

clf = load_classifier()
embedding_model = load_embedding_model()
# -----------------------------
# Startup Knowledge Base
# -----------------------------

startup_db = {
    "Uber": "ride sharing platform connecting drivers and passengers",
    "Airbnb": "platform for renting homes and travel accommodation",
    "Swiggy": "online food delivery platform connecting restaurants and customers",
    "Stripe": "online payment processing platform for internet businesses",
    "Notion": "productivity and workspace management software",
    "Spotify": "music streaming platform with personalized recommendations",
    "Zomato": "restaurant discovery and food delivery platform",
    "Amazon": "large scale ecommerce marketplace for buying and selling products"
}

startup_names = list(startup_db.keys())
startup_desc = list(startup_db.values())

startup_embeddings = embedding_model.encode(startup_desc)
explainer = load_explainer(clf)

# -----------------------------
# UI
# -----------------------------

st.title("🚀 AI Startup Intelligence Engine (LLM Powered)")
st.markdown("Multimodal ML + SHAP + LLM Strategic Analysis System")

mode = st.radio(
    "Choose Mode",
    ["Single Startup Analysis", "Compare Two Startups"]
)

if mode == "Single Startup Analysis":

    idea = st.text_area("📝 Describe your startup idea")

    budget = st.number_input("💰 Budget (in lakhs)", min_value=1)
    team_size = st.slider("👥 Team Size", 1, 25)
    timeline = st.slider("⏳ Timeline (months)", 1, 36)

elif mode == "Compare Two Startups":

    st.subheader("Startup A")

    idea_a = st.text_area("Idea A")

    budget_a = st.number_input("Budget A (lakhs)", min_value=1)
    team_a = st.slider("Team Size A", 1, 25, key="team_a")
    timeline_a = st.slider("Timeline A (months)", 1, 36, key="timeline_a")

    st.divider()

    st.subheader("Startup B")

    idea_b = st.text_area("Idea B")

    budget_b = st.number_input("Budget B (lakhs)", min_value=1)
    team_b = st.slider("Team Size B", 1, 25, key="team_b")
    timeline_b = st.slider("Timeline B (months)", 1, 36, key="timeline_b")

if mode == "Compare Two Startups":

    if st.button("Compare Startups"):

        emb_a = embedding_model.encode(idea_a)
        emb_b = embedding_model.encode(idea_b)

        feat_a = np.concatenate([emb_a, np.array([budget_a, team_a, timeline_a])])
        feat_b = np.concatenate([emb_b, np.array([budget_b, team_b, timeline_b])])

        prob_a = clf.predict_proba([feat_a])[0][1]
        prob_b = clf.predict_proba([feat_b])[0][1]

        prob_a = 0.6 * prob_a + 0.4 * 0.5
        prob_b = 0.6 * prob_b + 0.4 * 0.5

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Startup A Success Probability", f"{round(prob_a*100,2)}%")

        with col2:
            st.metric("Startup B Success Probability", f"{round(prob_b*100,2)}%")

        # Comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Startup A"],
            y=[prob_a*100],
            name="Startup A"
        ))

        fig.add_trace(go.Bar(
            x=["Startup B"],
            y=[prob_b*100],
            name="Startup B"
        ))

        st.subheader("📊 Startup Comparison")
        st.plotly_chart(fig, use_container_width=True)

                # -----------------------------
        # SHAP COMPARISON
        # -----------------------------

        shap_a = explainer.shap_values(np.array([feat_a]))
        shap_b = explainer.shap_values(np.array([feat_b]))

        if isinstance(shap_a, list):
            shap_a = shap_a[1].flatten()
            shap_b = shap_b[1].flatten()
        else:
            shap_a = shap_a.flatten()
            shap_b = shap_b.flatten()

        feature_names = [f"text_{i}" for i in range(384)] + ["budget", "team_size", "timeline"]

        shap_a = shap_a[:len(feature_names)]
        shap_b = shap_b[:len(feature_names)]

        shap_df_a = pd.DataFrame({
            "feature": feature_names,
            "value": shap_a
        })

        shap_df_b = pd.DataFrame({
            "feature": feature_names,
            "value": shap_b
        })

        structured_a = shap_df_a[shap_df_a.feature.isin(["budget","team_size","timeline"])]
        structured_b = shap_df_b[shap_df_b.feature.isin(["budget","team_size","timeline"])]

        comparison_df = pd.DataFrame({
            "feature": ["budget","team_size","timeline"],
            "Startup A": structured_a["value"].values,
            "Startup B": structured_b["value"].values
        }).set_index("feature")

        st.subheader("📊 Structured Feature Comparison (SHAP)")
        st.bar_chart(comparison_df)

                # -----------------------------
        # LLM COMPARISON ANALYSIS
        # -----------------------------

        structured_text_a = "\n".join(
            [f"{f}: {round(v,6)}" for f,v in zip(comparison_df.index, comparison_df["Startup A"])]
        )

        structured_text_b = "\n".join(
            [f"{f}: {round(v,6)}" for f,v in zip(comparison_df.index, comparison_df["Startup B"])]
        )

        with st.spinner("Generating AI Comparison Report..."):

            prompt = f"""
You are a startup strategy analyst.

Compare two startup ideas using the ML predictions and SHAP feature signals.

Startup A:
Idea: {idea_a}
Success Probability: {round(prob_a*100,2)}%

Structured SHAP signals:
{structured_text_a}

Startup B:
Idea: {idea_b}
Success Probability: {round(prob_b*100,2)}%

Structured SHAP signals:
{structured_text_b}

Tasks:
1. Explain why one startup scores higher.
2. Compare budget, team size and timeline risk.
3. Identify which startup has stronger structure.
4. Suggest improvements for the weaker startup.
Explain clearly in simple startup language. Focus on practical strategy."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=0.6
            )

            comparison_output = response.choices[0].message.content

        st.subheader("🧠 AI Startup Comparison Report")
        st.markdown(comparison_output)

# -----------------------------
# Prediction + SHAP + LLM
# -----------------------------

if mode == "Single Startup Analysis" and st.button("Analyze"):
    if idea.strip() == "":
        st.warning("Please enter a startup idea.")
        st.stop()

    # -------- Embedding --------
    embedding = embedding_model.encode(idea)
    # -----------------------------
    # Competitor Similarity
    # -----------------------------

    idea_embedding = embedding.reshape(1, -1)

    similarities = cosine_similarity(idea_embedding, startup_embeddings)[0]

    top_indices = similarities.argsort()[-3:][::-1]

    similar_startups = [
        (startup_names[i], similarities[i])
        for i in top_indices
    ]
    numeric_features = np.array([budget, team_size, timeline])
    final_features = np.concatenate([embedding, numeric_features])

    # -------- Prediction --------
    raw_prob = clf.predict_proba([final_features])[0][1]
    probability = 0.6 * raw_prob + 0.4 * 0.5

    st.metric("🚀 Success Probability", f"{round(probability*100,2)}%")

    # -------- Gauge --------
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "AI Success Score"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

    st.subheader("🏢 Similar Existing Startups")

    sim_df = pd.DataFrame(similar_startups, columns=["Startup","Similarity"])
    sim_df["Similarity"] = sim_df["Similarity"].round(2)
    st.dataframe(sim_df, use_container_width=True)

    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    # -------- SHAP Explanation --------
    shap_values = explainer.shap_values(np.array([final_features]))

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    # Convert to numpy array
    shap_array = np.array(shap_array)

    # Flatten to 1D
    shap_array = shap_array.flatten()

    # Feature names (must match feature count)
    feature_names = [f"text_{i}" for i in range(384)] + ["budget", "team_size", "timeline"]

    # Ensure correct length
    shap_array = shap_array[:len(feature_names)]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_array
    })

    # Top 5 impactful features
    top_shap = shap_df.loc[
        shap_df.shap_value.abs().sort_values(ascending=False).index
    ].head(5)   

    # -------- SHAP Chart --------
    st.subheader("📊 Model Feature Impact (SHAP)")

    shap_fig = go.Figure(go.Bar(
        x=top_shap["shap_value"],
        y=top_shap["feature"],
        orientation='h',
        marker=dict(
            color=[
                "green" if val > 0 else "red"
                for val in top_shap["shap_value"]
            ]
        )
    ))

    st.plotly_chart(shap_fig, use_container_width=True)

    # -------- Structured Feature SHAP --------
    structured_shap = shap_df[
        shap_df["feature"].isin(["budget", "team_size", "timeline"])
    ]

    st.subheader("📊 Structured Feature Impact")
    st.bar_chart(structured_shap.set_index("feature"))

    # -------- Prepare SHAP Text for LLM --------
    structured_shap = shap_df[
        shap_df["feature"].isin(["budget", "team_size", "timeline"])
    ]

    structured_text = "\n".join(
        [
            f"{row.feature}: {row.shap_value:.5f}"
            for _, row in structured_shap.iterrows()
        ]
    )

    # -------- LLM Analysis --------
    with st.spinner("Generating AI Strategic Report..."):

        prompt = f"""
You are a startup risk analysis AI.

IMPORTANT:
- Do NOT invent data.
- Only reason from provided inputs and SHAP signals.
- Be analytical and grounded.

Startup Idea:
{idea}

Structured Inputs:
Budget: {budget} lakhs
Team Size: {team_size}
Timeline: {timeline} months
Predicted Success Probability: {round(probability*100,2)}%

Structured Feature SHAP Contributions:
(Positive increases probability, negative decreases)

{structured_text}
Tasks:
1. Explain why the predicted probability is low or high.
2. Reference budget, team size, timeline in reasoning.
3. Identify structural weaknesses.
4. Provide practical improvements.
5. Keep explanation factual and analytical.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        llm_output = response.choices[0].message.content

    st.subheader("🧠 AI Strategic Intelligence Report")
    st.markdown(llm_output)