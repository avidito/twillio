FROM python:3.9-slim

RUN pip install nltk
RUN python -c "import nltk; nltk.download('stopwords')"

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src/ src

CMD uvicorn src.main:app --host 0.0.0.0 --port 80