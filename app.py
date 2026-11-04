import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "student_performance.csv")

st.title("🎓 Student Performance Prediction")
st.write("Predict expected final marks from academic and behavioural indicators.")
st.info("This project includes an illustrative synthetic dataset for demonstration. Replace it with verified student data before drawing real-world conclusions.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model_bundle = load_model()
if model_bundle is None:
    st.warning("The model has not been trained yet. Run `python train_model.py` in your project folder, then refresh this page.")
    st.stop()

model = model_bundle["model"]
features = model_bundle["features"]

st.sidebar.header("Enter student details")
attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
study_hours = st.sidebar.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
assignment = st.sidebar.slider("Assignment score (out of 100)", 0.0, 100.0, 80.0, 1.0)
participation = st.sidebar.slider("Class participation (1–10)", 1, 10, 7)
previous = st.sidebar.slider("Previous marks (out of 100)", 0.0, 100.0, 75.0, 1.0)

if st.sidebar.button("Predict performance", type="primary", use_container_width=True):
    sample = pd.DataFrame([{
        "attendance_percentage": attendance,
        "study_hours_per_day": study_hours,
        "assignment_score": assignment,
        "class_participation": participation,
        "previous_marks": previous,
    }])[features]
    prediction = float(model.predict(sample)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.subheader("Prediction result")
    c1, c2 = st.columns(2)
    c1.metric("Expected final marks", f"{prediction:.1f} / 100")
    category = "Excellent" if prediction >= 85 else "Good" if prediction >= 70 else "Average" if prediction >= 50 else "Needs support"
    c2.metric("Performance category", category)
    st.caption("This is a model estimate, not a guaranteed exam result.")

st.divider()
st.header("Dataset overview")
df = load_data()
a, b, c = st.columns(3)
a.metric("Student records", f"{len(df):,}")
b.metric("Average final marks", f"{df['final_marks'].mean():.1f}")
c.metric("Average attendance", f"{df['attendance_percentage'].mean():.1f}%")

tab1, tab2, tab3 = st.tabs(["Sample data", "Performance analysis", "Model information"])
with tab1:
    st.dataframe(df.head(15), use_container_width=True)
with tab2:
    st.subheader("Final marks distribution")
    hist = pd.cut(df["final_marks"], bins=10).value_counts().sort_index()
    # Convert pandas Interval categories to plain strings for Altair compatibility.
    hist_df = hist.rename_axis("Marks range").reset_index(name="Number of students")
    hist_df["Marks range"] = hist_df["Marks range"].astype(str)
    st.bar_chart(hist_df, x="Marks range", y="Number of students")
    st.subheader("Average final marks by attendance group")
    bins = [0, 50, 60, 70, 80, 90, 100]
    labels = ["≤50", "51–60", "61–70", "71–80", "81–90", "91–100"]
    grouped = df.assign(attendance_group=pd.cut(df["attendance_percentage"], bins=bins, labels=labels, include_lowest=True))
    st.bar_chart(grouped.groupby("attendance_group", observed=False)["final_marks"].mean())
with tab3:
    st.write(f"**Selected model:** {model_bundle['model_name']}")
    st.write(f"**Test MAE:** {model_bundle['metrics']['MAE']:.2f} marks")
    st.write(f"**Test RMSE:** {model_bundle['metrics']['RMSE']:.2f} marks")
    st.write(f"**Test R²:** {model_bundle['metrics']['R2']:.3f}")
    st.write("Feature importance / coefficients:")
    importance = model_bundle.get("feature_importance", {})
    if importance:
        imp_df = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())})
        st.bar_chart(imp_df.set_index("Feature"))
    st.caption("Feature importance describes model relationships, not proof of cause and effect.")
