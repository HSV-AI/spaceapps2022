import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="../../data/05_model_input/chromadb")

collection = chroma_client.get_or_create_collection(name="spaceapps")

model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')

text = "What is the ration of fuel to weight necessary for lunar liftoff?"
text_embed = model.encode(text)

print(text_embed)

rvalue = collection.query(
    query_embeddings=text_embed.tolist(),
    n_results=10
)

print(rvalue)
