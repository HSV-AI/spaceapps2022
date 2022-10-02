import click
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from bertopic import BERTopic

from sklearn.feature_extraction.text import CountVectorizer


@click.command()
@click.argument('input_parquet')
@click.argument('output')
def main(input_parquet, output):
    # sentence_model = SentenceTransformer('msmarco-distilbert-base-v4')
    # topic_model = BERTopic(embedding_model=sentence_model)

    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=5)

    # topic_model = BERTopic(verbose=True)

    df = pd.read_parquet(input_parquet)

    documents = df['text'].tolist()
    topics, probs = topic_model.fit_transform(documents)

    # Initial work to print. Eventually need to do something interesting with this.
    print(topic_model.get_topic_info())

    topic_model.save(output)

if __name__ == '__main__':
    main()