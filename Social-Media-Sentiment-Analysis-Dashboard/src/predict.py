import pickle

# Load saved model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Run from terminal
if __name__ == "__main__":
    print("🔍 Sentiment Prediction Tool")
    
    while True:
        text = input("Enter text (or type 'exit'): ")
        
        if text.lower() == "exit":
            print("👋 Exiting...")
            break
        
        result = predict_sentiment(text)
        print("👉 Sentiment:", result)