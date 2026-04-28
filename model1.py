# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from scipy.sparse import hstack, csr_matrix

# Step 2: Load dataset
df = pd.read_csv(
    'all-data.csv',
    encoding='latin1',
    names=['Sentiment', 'Sentence']
)

# Step 3: Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 4: Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

df['Clean_Text'] = df['Sentence'].apply(preprocess_text)

# Step 5: Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Sentiment']).astype(int)

# Step 6: TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Clean_Text'])

# Step 7: Custom Features
tricky_words = ["growth","increase","profit","develop","improved","strong","expanding","acquire","strategic","revenue"]

strong_positive_words = ["high","boost","excellent","record","optimistic","outperform","surpass","soar","success"]

strong_negative_words = ["loss","decline","crisis","weakness","fall","drop","risk","cut","slowdown"]

def count_words(text, word_list):
    count = 0
    for word in text.split():
        if word in word_list:
            count += 1
    return count

df['Tricky'] = df['Clean_Text'].apply(lambda x: count_words(x, tricky_words))
df['Pos'] = df['Clean_Text'].apply(lambda x: count_words(x, strong_positive_words))
df['Neg'] = df['Clean_Text'].apply(lambda x: count_words(x, strong_negative_words))

# Step 8: Scaling
scaler = StandardScaler()
extra_features = scaler.fit_transform(df[['Tricky','Pos','Neg']])

# Convert to sparse
extra_features_sparse = csr_matrix(extra_features)

# Combine features
X_combined = hstack([X, extra_features_sparse])

# Step 9: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Step 10: SMOTE (convert to dense FIRST)
X_train_dense = X_train.toarray()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_dense, y_train)

# Step 11: Model
model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_res, y_res)

# Step 12: Prediction
y_pred = model.predict(X_test.toarray())

# Step 13: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# Step 15: Save Model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

