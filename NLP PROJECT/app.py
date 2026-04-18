# =========================
# 1. Import Libraries
# =========================
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# 2. Page Config
# =========================
st.set_page_config(page_title="Business NLP Analyzer", layout="wide")

# =========================
# 3. DARK UI Styling
# =========================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Title */
.main-title {
    font-size:40px;
    font-weight:700;
    color:#ffffff;
}

/* Cards */
.card {
    padding:20px;
    border-radius:15px;
    background-color:#1c1f26;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
}

/* Text area */
textarea {
    background-color:#1c1f26 !important;
    color:white !important;
}

/* Labels */
label {
    color:white !important;
}

/* Buttons */
button {
    background-color:#262730 !important;
    color:white !important;
    border-radius:10px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color:#111318;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color:#1c1f26;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 4. Load Models
# =========================
sentiment_model = pickle.load(open('sentiment_model.pkl', 'rb'))
category_model = pickle.load(open('category_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_sentiment.pkl', 'rb'))

# =========================
# 5. Load Dataset
# =========================
data = pd.read_csv("final_dataset.csv")

# =========================
# 6. Text Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# 🔥 ADD HERE (NEW CODE)
# =========================
import re

BUSINESS_KEYWORDS = {
    "market","stock","economy","inflation","gdp","rbi","bank","policy",
    "finance","growth","profit","loss","trade","investment","share","index"
}

def is_gibberish(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 3:
        return True
    short_ratio = sum(len(w) <= 2 for w in words) / len(words)
    repeat_ratio = sum(len(set(w)) == 1 for w in words) / len(words)
    return short_ratio > 0.6 or repeat_ratio > 0.3

def is_business_like(text):
    text = text.lower()
    return any(k in text for k in BUSINESS_KEYWORDS)
# =========================
# 7. Title
# =========================
st.markdown("<div class='main-title'>🌙 Business News NLP Analyzer</div>", unsafe_allow_html=True)
st.write("Analyze financial news sentiment and category using NLP")

# =========================
# Sidebar
# =========================
st.sidebar.header("ℹ️ About")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Features: TF-IDF")
st.sidebar.write("Tasks: Sentiment + Category")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Dashboard", "☁️ WordCloud"])

# =========================
# TAB 1: Prediction
# =========================
with tab1:
    st.subheader("🧠 Analyze News")

    user_input = st.text_area("📝 Enter News Text")
    
    if st.button("📌 Try Sample"):
        user_input = "India market shows strong economic growth"
        st.write(user_input)

    if st.button("🚀 Analyze Now"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter valid text")

        elif is_gibberish(user_input):
            st.error("❌ Input looks like random / meaningless text")

        elif not is_business_like(user_input):
            st.warning("⚠️ This may not be business-related text")

        else:
            clean_input = clean_text(user_input)

            vec = vectorizer.transform([clean_input])

            sent = sentiment_model.predict(vec)[0]
            cat = category_model.predict(vec)[0]

            probs = sentiment_model.predict_proba(vec)[0]
            confidence = max(probs) * 100

            st.markdown("### 📊 Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("📂 Category", cat)

            with col2:
                st.metric("📈 Sentiment", "Positive" if sent == 1 else "Negative")

            if sent == 1:
                st.success("😊 Positive News")
            else:
                st.error("😟 Negative News")

            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
# =========================
# TAB 2: Dashboard
# =========================
with tab2:
    st.subheader("📊 Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Sentiment Distribution")
        st.bar_chart(data['sentiment'].value_counts())

    with col2:
        st.write("Category Distribution")
        st.bar_chart(data['category'].value_counts())

    st.write("Sample Data")
    st.dataframe(data.head())

# =========================
# TAB 3: WordCloud
# =========================
with tab3:
    st.subheader("☁️ WordCloud")

    text_data = " ".join(data['text'].astype(str))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black'   # 🔥 changed to black
    ).generate(text_data)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)
    
