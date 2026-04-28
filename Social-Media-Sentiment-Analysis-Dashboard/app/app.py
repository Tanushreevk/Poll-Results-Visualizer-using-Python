import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Sentiment Dashboard", layout="centered")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Title
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.write("Analyze customer opinions in real-time")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🔍 Analyze New Comment")

user_input = st.text_area("Enter a comment:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == "positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif prediction == "negative":
            st.error(f"😡 Sentiment: {prediction}")
        else:
            st.info(f"😐 Sentiment: {prediction}")

# -----------------------------
# LOAD DATA
# -----------------------------
st.subheader("📂 Dataset Preview")

df = pd.read_csv("data/social_media_data.csv")
st.dataframe(df)

# -----------------------------
# SENTIMENT COUNTS
# -----------------------------
st.subheader("📊 Sentiment Distribution")

sentiment_counts = df['sentiment'].value_counts()

col1, col2 = st.columns(2)

# Bar Chart
with col1:
    st.write("Bar Chart")
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Pie Chart
with col2:
    st.write("Pie Chart")
    fig2, ax2 = plt.subplots()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    st.pyplot(fig2)

st.subheader("📊 Model Performance")

st.write("Accuracy: ~0.95")
st.image("outputs/confusion_matrix.png")
# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("📈 Insights")

total = len(df)
positive = sentiment_counts.get("positive", 0)
negative = sentiment_counts.get("negative", 0)
neutral = sentiment_counts.get("neutral", 0)

st.write(f"Total Comments: {total}")
st.write(f"Positive: {positive}")
st.write(f"Negative: {negative}")
st.write(f"Neutral: {neutral}")

if negative > positive:
    st.warning("⚠️ More negative feedback detected!")
else:
    st.success("✅ Overall sentiment is positive!")