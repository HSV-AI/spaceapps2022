import click
import pandas as pd
import numpy as np
import chromadb
import time


@click.command()
@click.argument('input_parquet')
@click.argument('input_embedding')
@click.argument('chromadb_path')
def main(input_parquet, input_embedding, chromadb_path):

    chroma_client = chromadb.PersistentClient(path=chromadb_path)

    df = pd.read_parquet(input_parquet)
    chunks = df['chunk'].tolist()
    pages = df['page'].tolist()
    ids = df['id'].tolist()
    embeddings = np.load(input_embedding)

    collection = chroma_client.get_or_create_collection(name="chunks")

    for index, (embedding, chunk, page, doc_id) in enumerate(zip(embeddings, chunks, pages, ids)):
        start = time.perf_counter_ns()
        collection.upsert(
            ids=str(index),
            embeddings=embedding.tolist(),
            metadatas={"key": doc_id, "page": page},
            documents=chunk
        )
        stop = time.perf_counter_ns()
        print(f'Upsert time for {doc_id}: {(stop - start)/1e6:0.4f}ms')


if __name__ == '__main__':
    main()