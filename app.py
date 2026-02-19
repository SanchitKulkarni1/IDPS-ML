import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------
# Load model and training data
# --------------------------
@st.cache_resource
def load_model():
    model = joblib.load(open("models/best_multi_model.pkl", "rb"))
    X_train = joblib.load(open("models/X_train_multi_selected.pkl", "rb"))
    return model, X_train

model, X_train = load_model()
feature_names = X_train.columns.tolist()

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="AnyScanner IDS - Multiclass Detection", layout="wide")
st.title("🧠 AnyScanner - Intrusion Detection System (Multiclass)")
st.write("Predict the type of attack dynamically using network traffic features.")

st.sidebar.header("⚙️ Input Configuration")

# Input selection mode
input_mode = st.sidebar.radio(
    "Select input mode:",
    ["Manual Entry", "Use Random Dataset Entry", "Generate Synthetic Random Entry"]
)

input_data = {}

# --------------------------
# 1️⃣ Manual Entry
# --------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("Enter network traffic features:")
    for feature in feature_names:
        if "flag" in feature.lower() or "proto" in feature.lower() or "service" in feature.lower():
            input_data[feature] = st.sidebar.text_input(f"{feature}")
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# --------------------------
# 2️⃣ Use Random Dataset Entry
# --------------------------
elif input_mode == "Use Random Dataset Entry":
    random_row = X_train.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
    for feature in feature_names:
        input_data[feature] = random_row[feature]
    st.sidebar.success("✅ Random sample from training dataset selected!")

# --------------------------
# 3️⃣ Generate Synthetic Random Entry
# --------------------------
elif input_mode == "Generate Synthetic Random Entry":
    for feature in feature_names:
        if "flag" in feature.lower() or "proto" in feature.lower() or "service" in feature.lower():
            # pick a random unique category from X_train column if available
            if feature in X_train.columns and X_train[feature].nunique() > 0:
                input_data[feature] = np.random.choice(X_train[feature].unique())
            else:
                input_data[feature] = "unknown"
        else:
            # sample random value based on mean and std from training data
            mean = X_train[feature].mean()
            std = X_train[feature].std()
            # clip to min and max range to avoid outliers
            val = np.random.normal(mean, std)
            input_data[feature] = float(np.clip(val, X_train[feature].min(), X_train[feature].max()))

    st.sidebar.success("🎲 Synthetic random entry generated using feature statistics!")

# --------------------------
# Convert to DataFrame
# --------------------------
input_df = pd.DataFrame([input_data])

# --------------------------
# Display selected/generated values
# --------------------------
with st.expander("🔍 View Input Features"):
    st.dataframe(input_df.T, use_container_width=True)

# --------------------------
# Prediction
# --------------------------
if st.sidebar.button("🚀 Predict Attack Type"):
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        st.success(f"### 🧩 Predicted Attack Type: `{prediction}`")

        # Display probability visualization
        prob_df = pd.DataFrame({
            'Attack Type': model.classes_,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False)

        st.subheader("Model Confidence Levels")
        st.bar_chart(prob_df.set_index('Attack Type'))
        st.caption("The above chart shows model confidence levels for each attack category.")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👈 Choose an input method and click **Predict Attack Type** to see results.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("Developed as part of BE Project | Phase 1: Detection & Visualization | Phase 2: Prevention & Database Logging (upcoming)")
