import os
import pandas as pd
import numpy as np
import nltk
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not already present
nltk.download("stopwords")
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load dataset with error handling
DATA_FILE = "student_faq_dataset_updated.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    if "Question" not in df.columns or "Response" not in df.columns:
        raise ValueError("CSV file must contain 'question' and 'response' columns")
else:
    raise FileNotFoundError(f"Dataset file '{DATA_FILE}' not found.")

# Preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

df["Question"] = df["Question"].apply(clean_text)

# Train TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Question"])

# Chatbot function
def chatbot_response(user_input):
    user_input = clean_text(user_input)
    if not user_input.strip():
        return "Please enter a valid question."
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    best_match_index = np.argmax(similarity)
    confidence = similarity[0, best_match_index]

    if confidence > 0.3:  # Adjust threshold for better accuracy
        return df.iloc[best_match_index]["Response"]
    else:
        return "I'm not sure about that. Can you provide more details?"

# Flask API Route
@app.route("/chatbot", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Please enter a valid message."})

        response = chatbot_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(port=5000, debug=True)