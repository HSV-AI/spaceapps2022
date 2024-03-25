import click
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_parquet')
@click.argument('input_key')
@click.argument('output')
def main(input_parquet: str, input_key: str, output: str):

    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')
    df = pd.read_parquet(input_parquet)

    embeddings = model.encode(df[input_key].to_list())

    np.save(output, embeddings)


if __name__ == '__main__':
    main()