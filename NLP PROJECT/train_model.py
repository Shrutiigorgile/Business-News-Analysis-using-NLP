# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download once
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# 2. Load Dataset
# =========================
data = pd.read_csv("final_dataset.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop_duplicates()

# =========================
# 3. Cleaning
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

# =========================
# 4. TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(data['clean_text'])

# =========================
# 5. Targets
# =========================
y_sent = data['sentiment']
y_cat = data['category']

# =========================
# 6. Train Sentiment Model
# =========================
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_sent,
    test_size=0.2,
    random_state=42,
    stratify=y_sent
)

model_sent = LogisticRegression(max_iter=1000, class_weight='balanced')
model_sent.fit(X_train_s, y_train_s)

# =========================
# 7. Train Category Model
# =========================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42
)

model_cat = LogisticRegression(max_iter=1000)
model_cat.fit(X_train_c, y_train_c)

# =========================
# 8. Save Models
# =========================
pickle.dump(model_sent, open("sentiment_model.pkl", "wb"))
pickle.dump(model_cat, open("category_model.pkl", "wb"))

pickle.dump(tfidf, open("vectorizer_sentiment.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer_category.pkl", "wb"))

print("✅ Both models trained & saved successfully!")