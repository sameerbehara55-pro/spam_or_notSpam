import streamlit as st
import pickle
import re
import string

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("sms_spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, tfidf = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 SMS Spam Detection App")
st.write("Enter an SMS message to check whether it is **Spam** or **Not Spam (Ham)**.")

sms_text = st.text_area("✉️ Enter SMS message here:")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Spam"):
    if sms_text.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned_text = clean_text(sms_text)
        vector = tfidf.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM (HAM)")
