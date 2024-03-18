import click
import pandas as pd
import numpy as np
import chromadb

@click.command()
@click.argument('input_parquet')
@click.argument('input_embedding')
@click.argument('chromadb_path')
def main(input_parquet, input_embedding, chromadb_path):

    chroma_client = chromadb.PersistentClient(path=chromadb_path)

    df = pd.read_parquet(input_parquet)
    abstracts = df['abstract'].tolist()
    titles = df['title'].tolist()
    ids = df['id'].tolist()
    embeddings = np.load(input_embedding)

    collection = chroma_client.get_or_create_collection(name="spaceapps")

    for embedding, abstract, title, doc_id in zip(embeddings, abstracts, titles, ids):
        collection.upsert(
            ids=doc_id,
            embeddings=embedding.tolist(),
            metadatas={"title": title},
            documents=abstract
        )


if __name__ == '__main__':
    main()