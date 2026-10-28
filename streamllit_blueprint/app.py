"""
Streamlit application for House Price Prediction.

Run:
    streamlit run app.py
"""
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

@st.cache_resource
def load_model():
    with open("model/house_price_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("🏠 House Price Prediction")
st.caption("Simple Linear Regression + Scikit-learn Pipeline + Streamlit")

st.markdown(
    "Enter the house details below. The saved pipeline applies the same "
    "preprocessing used during training and returns the predicted price."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=300.0,
            max_value=10000.0,
            value=1500.0,
            step=50.0,
        )
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )
        age = st.number_input(
            "House Age (years)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame(
        {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "age": [age],
        }
    )

    prediction = float(model.predict(input_data)[0])

    st.success(f"Estimated House Price: ₹{prediction:,.0f}")

    with st.expander("View input data"):
        st.dataframe(input_data, use_container_width=True)

st.divider()
st.markdown("### Project Flow")
st.code(
    "User Input → DataFrame → Saved Pipeline → Imputer → StandardScaler "
    "→ Linear Regression → Prediction"
)
