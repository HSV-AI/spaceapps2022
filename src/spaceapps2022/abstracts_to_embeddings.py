import click
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_parquet')
@click.argument('output')
def main(input_parquet, output):
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')
    df = pd.read_parquet(input_parquet)
    abstracts = df['abstract'].tolist()
    titles = df['title'].tolist()

    # Combine abstracts and titles
    abstracts_and_titles = []
    for i in range(len(abstracts)):
        abstracts_and_titles.append(abstracts[i] + titles[i])
    
    embeddings = model.encode(abstracts_and_titles)

    np.save(output, embeddings)


if __name__ == '__main__':
    main()