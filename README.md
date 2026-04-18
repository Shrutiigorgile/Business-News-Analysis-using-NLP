# Business-News-Analysis-using-NLP
 NLP-Based Business News Sentiment & Category Analyzer

An interactive **Natural Language Processing (NLP)** web application that analyzes business news text to predict **sentiment (positive/negative)** and **category (Finance, Market, Economy, etc.)** using Machine Learning.

---

## 🚀 Features

* 🧠 **Sentiment Analysis** (Positive / Negative)
* 📂 **News Category Classification**
* 📊 **Interactive Dashboard**
* ☁️ **WordCloud Visualization**
* 🌙 **Modern Dark UI (Streamlit)**
* ⚡ **Real-time Prediction**

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (UI)
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Logistic Regression**
* **NLTK (Text Processing)**
* **Matplotlib & WordCloud**

---

## 📂 Project Structure

```
project/
│
├── app.py                      # Streamlit App
├── train_model.py              # Model Training Script
├── final_dataset.csv           # Dataset
├── sentiment_model.pkl         # Sentiment Model
├── category_model.pkl          # Category Model
├── vectorizer_sentiment.pkl    # TF-IDF Vectorizer
├── vectorizer_category.pkl     # TF-IDF Vectorizer
├── requirements.txt
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/business-nlp-analyzer.git
cd business-nlp-analyzer
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run app.py
```

---

## 📊 How It Works

1. Input news text
2. Text is cleaned (stopwords removal, lemmatization)
3. Converted into TF-IDF features
4. Two models are used:

   * Sentiment Model
   * Category Model
5. Predictions are displayed with confidence score

---

## 🧪 Example Inputs

* "India market shows strong economic growth"
* "Stock market crashes due to inflation crisis"
* "RBI introduces new banking policy"

---

## ⚠️ Limitations

* Works best on **news-like text**
* May give incorrect predictions for **random or meaningless sentences**
* Based on classical ML (not deep learning)

---

## 🔮 Future Improvements

* 🔥 Integrate **BERT / Transformers**
* 📰 Real-time news API integration
* 📈 Advanced analytics dashboard
* 🌐 Deployment with live data pipeline

---

## 🌍 Deployment

This project can be deployed on:

* **Streamlit Cloud**
* **Render**
* **Hugging Face Spaces**

---

## 💼 Resume Description

Developed an NLP-based system to analyze business news using TF-IDF and machine learning models for sentiment and category classification, deployed with an interactive Streamlit interface.

---

## 👩‍💻 Author

**Shruti Gorgile**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
