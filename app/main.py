from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
from app.preprocessing import clean_text

# Initialize FastAPI app
app = FastAPI(title="IMDB Movie Reviews Sentiment Analysis", version="1.0")


def load_model():
    # Get absolute path of current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    model_path = os.path.join(
        base_dir, "models", "sentiment_analysis_model.pkl")

    try:
        with open(vectorizer_path, "rb") as vec_file, open(model_path, "rb") as model_file:
            vectorizer = pickle.load(vec_file)
            model = pickle.load(model_file)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"Model files not found: {str(e)}")

    return vectorizer, model


vectorizer, model = load_model()


class ReviewRequest(BaseModel):
    input_text: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}


@app.post("/analyze/")
async def analyze_sentiment(review: ReviewRequest):
    if not review.input_text:
        raise HTTPException(status_code=400, detail="No input text provided!")

    # Preprocess the text
    cleaned_review = clean_text(review.input_text)
    vectorized_review = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(vectorized_review)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return {"error": False, "review": review.input_text, "sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
