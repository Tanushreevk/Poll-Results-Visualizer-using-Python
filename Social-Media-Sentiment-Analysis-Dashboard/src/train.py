from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import load_and_clean_data

print("🚀 Training Started...")

# Load data
df = load_and_clean_data("data/social_media_data.csv")

print("✅ Data Loaded")
print(df.head())

X = df['clean_text']
y = df['sentiment']

# ✅ SPLIT FIRST (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ TF-IDF ONLY ON TRAIN DATA
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("✅ Text Vectorized (No Data Leakage)")

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("✅ Model Trained")

# Evaluate
pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, pred)
print("🎯 Accuracy:", accuracy)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save image
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")

print("📊 Confusion matrix saved in outputs/")

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("💾 Model Saved Successfully!")