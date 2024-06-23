import time
import os
import click
import pandas as pd
import numpy as np
import weaviate
import weaviate.classes as wvc

@click.command()
@click.argument('input_parquet')
@click.argument('input_embedding')
def main(input_parquet, input_embedding):

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WCD_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY")),
        additional_config=wvc.init.AdditionalConfig(timeout=(180, 180))
    )

    # Check connection
    client.is_ready()

    df = pd.read_parquet(input_parquet)
    chunks = df['chunk'].tolist()
    pages = df['page'].tolist()
    ids = df['id'].tolist()
    embeddings = np.load(input_embedding)

    collection = None
    if client.collections.exists("Chunks"):
        collection = client.collections.get("Chunks")
    else:
        collection = client.collections.create(
            name="Chunks",
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )

    question_objs = list()
    for index, (embedding, chunk, page, doc_id) in enumerate(zip(embeddings, chunks, pages, ids)):
        if chunk.count(' ') > 9:
            question_objs.append(wvc.data.DataObject(
                properties={
                    "key": doc_id,
                    "page": page,
                    "chunk": chunk,
                },
                vector=embedding.tolist()
            ))
        # else:
        #     print(f'Skipping: {chunk}')
        
        # Add 1000 chunks at a time to keep the weaviate resource from running out of memory
        if index % 1000 == 0:
            start = time.perf_counter_ns()
            collection.data.insert_many(question_objs)    # This uses batching under the hood
            stop = time.perf_counter_ns()
            print(f'Upsert time for: {(stop - start)/1e6:0.4f}ms') 
            question_objs.clear()

    start = time.perf_counter_ns()
    collection.data.insert_many(question_objs)    # This uses batching under the hood
    stop = time.perf_counter_ns()
    print(f'Upsert time for: {(stop - start)/1e6:0.4f}ms') 
    question_objs.clear()

    client.close()

if __name__ == '__main__':
    main()