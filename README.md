# NLP Challenge - IMDB Dataset of 50K Movie Reviews to perform Sentiment analysis

This solution is based on the [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) of 50K Movie Reviews. The dataset is used to perform sentiment analysis on the movie reviews. The trained model is served as a REST API using FastAPI and dockerized using Docker.

> You can find the API hosted at [https://sentiment-analysis.astro-dev.tech](https://sentiment-analysis.astro-dev.tech)

## Tech Stack

- Python 3.12
- Scikit-learn
- Pandas
- FastAPI
- Docker

## Training the Model

The model is trained using a dataset of 50K IMDB reviews:

- Open the app/sentiment_analysis.ipynb notebook.
- Follow the steps for Exploratory Data Analysis (EDA) and model training.
- Save the trained model and vectorizer as .pkl files in the models directory.

## API endpoints

- `GET /`: Returns a welcome message
- `POST /analyze/`: Perform sentiment analysis on the input text

## Installation

> Make sure to execute all the cells in the notebook (app/sentiment_analysis.ipynb) sequentially to train and save the model before running the API.

### Using docker

1. Build the docker image

```bash
docker build -t imdb-sentiment-analysis .
```

2. Run the docker image

```bash
docker run -p 8000:8000 imdb-sentiment-analysis
```

3. Access the API

```bash
http POST localhost:8000/analyze input_text="i really like react hooks" --follow
```

### Using pip

1. Install the requirements

```bash
pip install -r requirements.txt
```

2. Run the app

```bash
python app/main.py
```

3. Access the API

```bash
http POST localhost:8000/analyze input_text="i really like react hooks" --follow
```
