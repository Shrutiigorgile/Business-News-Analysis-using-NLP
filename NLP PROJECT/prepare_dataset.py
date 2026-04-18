import pandas as pd

# load your dataset
data = pd.read_csv(r"C:\Users\ASUS\Downloads\NLP PROJECT\india_inflation_translated.csv")

# use text column
data['text'] = data['Title_English']

# =========================
# 1. Create Sentiment (Rule-based)
# =========================
positive_words = ['growth', 'rise', 'increase', 'strong', 'gain', 'improve']
negative_words = ['fall', 'decline', 'crisis', 'drop', 'inflation', 'loss']

def get_sentiment(text):
    text = str(text).lower()
    if any(word in text for word in positive_words):
        return 1
    elif any(word in text for word in negative_words):
        return 0
    else:
        return 1  # default

data['sentiment'] = data['text'].apply(get_sentiment)

# =========================
# 2. Create Category (Business classification)
# =========================
def get_category(text):
    text = str(text).lower()
    
    if 'inflation' in text or 'cpi' in text:
        return 'Economy'
    elif 'market' in text or 'stock' in text:
        return 'Market'
    elif 'policy' in text or 'rbi' in text:
        return 'Policy'
    elif 'bank' in text or 'loan' in text:
        return 'Finance'
    else:
        return 'Business'

data['category'] = data['text'].apply(get_category)

# save new dataset
data[['text','sentiment','category']].to_csv("final_dataset.csv", index=False)

print("✅ Dataset ready!")