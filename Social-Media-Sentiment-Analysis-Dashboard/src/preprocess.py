import re
import pandas as pd

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove special chars
    return text

def load_and_clean_data(path):
    df = pd.read_csv(path)
    
    # Clean text column
    df['clean_text'] = df['text'].apply(clean_text)
    
    return df