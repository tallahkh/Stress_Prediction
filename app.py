
import streamlit as st
import joblib
import re
import string

# --- Page Configuration ---
st.set_page_config(page_title="Stress Detection System", layout="centered")

# --- Function to Clean Text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Load Assets ---
@st.cache_resource # To load models only once and save memory
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    models = {
        "XGBoost": joblib.load('xgb_model.pkl'),
        "SVM": joblib.load('svm_model.pkl'),
        "Logistic Regression": joblib.load('logit_model.pkl'),
        "Random Forest": joblib.load('rf_model.pkl')
    }
    return vectorizer, models

vectorizer, models = load_assets()

# --- UI Layout ---
st.image('https://static.vecteezy.com/system/resources/thumbnails/003/317/129/small/female-stressed-with-mouth-open-irritation-factor-vector.jpg', width = 200)
st.title("Stress Detection Hub")
st.write("Detect stress levels in text using multiple Machine Learning models.")

# Model Selection Dropdown
selected_model_name = st.selectbox("Select a Model for Prediction:", list(models.keys()))

# Text Input
user_input = st.text_area("Enter the text to be analyzed:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # 1. Preprocess
        cleaned_input = clean_text(user_input)
        
        # 2. Vectorize
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Predict
        current_model = models[selected_model_name]
        prediction = current_model.predict(vectorized_input)[0]
        
        # 4. Display Result
        st.divider()
        if prediction == 1:
            st.error(f"**Result from {selected_model_name}: Stress Detected**")
        else:
            st.success(f"**Result from {selected_model_name}: No Stress Detected**")

st.info("Note: This tool is for project demonstration purposes and uses 4 different ML architectures.")



