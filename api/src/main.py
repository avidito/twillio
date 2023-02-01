from fastapi import FastAPI
from src.process import predict_sentiment, count_sentiment


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/sentiment")
def get_sentiment(query: str):
    labels, prediction = predict_sentiment(query)
    sentiment_cnt = count_sentiment(labels, prediction)
    sentiment_pct = {
        key: round((value/len(prediction))*100, 2)
        for key, value in sentiment_cnt.items()
    }

    return {
        "query": query,
        "sentiment": {
            "result": max(sentiment_cnt, key=sentiment_cnt.get),
            "count": sentiment_cnt,
            "percentage": sentiment_pct
        }
    }