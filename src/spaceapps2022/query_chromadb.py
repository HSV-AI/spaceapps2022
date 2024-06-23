import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

collection = chroma_client.get_or_create_collection(name="chunks")

model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')

with open("data/01_raw/ntrs-public-metadata.json", 'r') as f:
    ntrs = json.load(f)

text = "What is the ration of fuel to weight necessary for lunar liftoff?"
text_embed = model.encode(text)

print(text_embed)

rvalue = collection.query(
    query_embeddings=text_embed.tolist(),
    n_results=10
)


df = pd.read_parquet("data/04_feature/abstracts.parquet")

for distance, document, meta in zip(rvalue['distances'][0], rvalue['documents'][0], rvalue['metadatas'][0]):
    key = meta['key']
    title = ntrs[key]['title']
    # title = df.loc[df['id'] == key].to_dict(orient='records')[0]['title']
    print(f'{key} Document: {title}, page {meta["page"]} - "{document}"\n')
