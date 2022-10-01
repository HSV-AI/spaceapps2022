import click
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_parquet')
@click.argument('output')
def main(input_parquet, output):
    model = SentenceTransformer('msmarco-distilbert-base-v4')
    df = pd.read_parquet(input_parquet)
    embeddings = model.encode(df['text'].tolist())
    np.save(output, embeddings)


if __name__ == '__main__':
    main()