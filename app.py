import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart Student Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #243b75;
}

.subtitle {
    font-size: 16px;
    color: #64748b;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dbeafe;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
}

.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #3730a3;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Previous_Marks": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Final_Marks": [30, 38, 43, 48, 55, 62, 68, 74, 82, 90]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Previous_Marks", "Attendance"]]
y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎓 Student Dashboard")
st.sidebar.write("Enter student details below.")

hours = st.sidebar.slider(
    "Daily Study Hours",
    0.0, 12.0, 5.0, 0.5
)

previous = st.sidebar.slider(
    "Previous Marks",
    0, 100, 70
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0, 100, 85
)

st.sidebar.divider()
st.sidebar.info(
    "This project demonstrates Machine Learning "
    "using a sample student dataset."
)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎓 Smart Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Dashboard | Student Performance Analytics</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- NAVIGATION ----------------

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "📁 Dataset",
    "ℹ️ About Project"
])

# ---------------- DASHBOARD ----------------

with tab1:

    st.subheader("📌 Student Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Study Hours", f"{hours} hrs/day")
    col2.metric("Previous Marks", f"{previous}/100")
    col3.metric("Attendance", f"{attendance}%")

    st.divider()

    if st.button("🔮 Predict Student Performance", use_container_width=True):

        student = pd.DataFrame(
            [[hours, previous, attendance]],
            columns=[
                "Study_Hours",
                "Previous_Marks",
                "Attendance"
            ]
        )

        prediction = model.predict(student)[0]
        prediction = max(0, min(100, prediction))

        st.subheader("🎯 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Final Marks",
            f"{prediction:.2f}/100"
        )

        result_col2.metric(
            "Attendance",
            f"{attendance}%"
        )

        if prediction >= 75:
            st.success(
                "Good predicted performance! Keep practising consistently."
            )
        elif prediction >= 50:
            st.info(
                "Moderate predicted performance. Regular revision may help."
            )
        else:
            st.warning(
                "Consider reviewing your study routine and practising regularly."
            )

        st.subheader("📈 Performance Visualization")

        chart_data = pd.DataFrame({
            "Category": [
                "Previous Marks",
                "Predicted Final Marks"
            ],
            "Marks": [
                previous,
                prediction
            ]
        })

        fig = px.bar(
            chart_data,
            x="Category",
            y="Marks",
            color="Category",
            text="Marks",
            title="Previous Marks vs Predicted Final Marks",
            range_y=[0, 100],
            color_discrete_sequence=["#6366f1", "#14b8a6"]
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Prediction is an educational estimate based on a small sample dataset, "
            "not a guarantee of actual examination marks."
        )

    else:
        st.info(
            "Adjust the student details in the sidebar and click "
            "'Predict Student Performance' to see the result."
        )

# ---------------- DATASET ----------------

with tab2:

    st.subheader("📁 Sample Training Dataset")

    st.write(
        "The model learns patterns from these example student records."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This is a small demonstration dataset, not real student records."
    )

# ---------------- ABOUT ----------------

with tab3:

    st.subheader("ℹ️ About This Project")

    st.write("""
    **Smart Student Performance Predictor** is a beginner-level
    Machine Learning demonstration.

    **Objective:** Estimate student marks from study hours,
    previous marks and attendance.

    **Algorithm:** Linear Regression

    **Libraries:** Python, Pandas, Scikit-learn, Streamlit and Plotly

    **Workflow:**
    1. Prepare the sample dataset.
    2. Separate input features and target marks.
    3. Train a Linear Regression model.
    4. Accept student inputs.
    5. Predict estimated marks and visualize results.

    **Limitation:** The model uses only ten synthetic sample records.
    Its predictions are for demonstration and should not be treated
    as reliable real-world academic forecasts.
    """)

st.divider()

st.caption(
    "Developed as a B.Tech CSE Machine Learning project | Python + Scikit-learn"
)