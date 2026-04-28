from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Initialize Flask
# -----------------------------

app = Flask(__name__)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# NLP Tools
# -----------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Tricky words lists
tricky_words = ["growth","increase","profit","develop","improved","strong","expanding","acquire","strategic","revenue"]
strong_positive_words = ["high","boost","excellent","record","optimistic","outperform","surpass","soar","success"]
strong_negative_words = ["loss","decline","crisis","weakness","fall","drop","risk","cut","slowdown"]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    words = word_tokenize(text)
    processed = []

    for word in words:
        if word not in stop_words and word.isalpha():
            processed.append(lemmatizer.lemmatize(word))

    return " ".join(processed)

def count_words(text, word_list):
    count = 0
    for word in text.split():
        if word in word_list:
            count += 1
    return count


# -----------------------------
# Home Page
# -----------------------------

@app.route("/", methods=["GET","POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        clean_text = preprocess_text(news)

        # TF-IDF
        vector = vectorizer.transform([clean_text])

        # Custom features
        tricky = count_words(clean_text, tricky_words)
        pos = count_words(clean_text, strong_positive_words)
        neg = count_words(clean_text, strong_negative_words)

        extra_features = scaler.transform([[tricky, pos, neg]])

        from scipy.sparse import hstack
        combined = hstack([vector, extra_features])

        result = model.predict(combined.toarray())

        prediction = label_encoder.inverse_transform(result)[0]

    return render_template("index.html", prediction=prediction)


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)