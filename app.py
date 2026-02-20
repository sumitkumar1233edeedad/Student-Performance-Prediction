import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #2196F3, #4CAF50);
    color: white;
}
.metric-box {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction System")
st.markdown("### 🚀 Predict Student Final Score Using Machine Learning")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Enter Student Details")

study_hours = st.sidebar.slider("📚 Study Hours (per day)", 0, 12, 6)
attendance = st.sidebar.slider("🏫 Attendance (%)", 0, 100, 80)
previous_score = st.sidebar.slider("📝 Previous Exam Score", 0, 100, 65)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0, 10, 7)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the sliders and click Predict.")

# ---------------- MAIN DASHBOARD ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Student Input Summary")
    st.metric("📚 Study Hours", f"{study_hours} hrs")
    st.metric("🏫 Attendance", f"{attendance}%")
    st.metric("📝 Previous Score", previous_score)
    st.metric("😴 Sleep Hours", f"{sleep_hours} hrs")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=220)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Final Score"):

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    input_data = np.array([[study_hours, attendance, previous_score, sleep_hours]])
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Final Score: {prediction:.2f}")

    # Performance Label
    if prediction >= 85:
        st.balloons()
        level = "🌟 Excellent"
    elif prediction >= 70:
        level = "👍 Good"
    elif prediction >= 50:
        level = "⚠️ Average"
    else:
        level = "❌ Needs Improvement"

    st.info(f"Performance Level: {level}")

    # ---------------- GAUGE CHART ----------------
    st.subheader("📈 Score Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Final Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4d4d"},
                {'range': [50, 70], 'color': "#ffa500"},
                {'range': [70, 85], 'color': "#00bfff"},
                {'range': [85, 100], 'color': "#4CAF50"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE ----------------
    if hasattr(model, "feature_importances_"):
        st.subheader("📊 Feature Importance")

        features = ["Study Hours", "Attendance", "Previous Score", "Sleep Hours"]
        importances = model.feature_importances_

        fig2, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")

        st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("<br><br><br>", unsafe_allow_html=True)

current_year = datetime.now().year

st.markdown(f"""
<style>
.footer {{
    width: 100%;
    background: linear-gradient(135deg, #1f1f1f, #2c3e50);
    color: white;
    text-align: center;
    padding: 25px 10px;
    margin-top: 50px;
    border-radius: 15px 15px 0 0;
    box-shadow: 0 -5px 20px rgba(0,0,0,0.3);
}}

.footer a {{
    color: #00d4ff;
    text-decoration: none;
    margin: 0 12px;
    font-weight: 500;
    transition: 0.3s;
}}

.footer a:hover {{
    color: #4CAF50;
    text-shadow: 0 0 10px #4CAF50;
}}

.footer p {{
    margin: 6px;
    font-size: 14px;
}}
</style>

<div class="footer">
    <h4>🎓 Student Performance Prediction System</h4>
    <p>Empowering Education with Machine Learning 🚀</p>
    <p>
        <a href="https://github.com/sumitkumar1233edeedad" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/sumit-kumar-42b09a296" target="_blank">LinkedIn</a> |
        <a href="mailto:sumitsohal963@gmail.com">Email</a>
    </p>
    <p>© {current_year} | Built with ❤️ by Vanshuu</p>
</div>
""", unsafe_allow_html=True)