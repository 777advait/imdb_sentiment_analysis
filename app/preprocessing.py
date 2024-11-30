from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stopwords = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Cleans the text by making everything lowercase and removing stopwords or special characters"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = [word for word in text.split() if word not in stopwords]

    return " ".join(tokens)


class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)
