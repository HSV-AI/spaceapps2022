import json
import os
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery
import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WCD_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY")),
    additional_config=wvc.init.AdditionalConfig(timeout=(180, 180))
)

# Check connection
client.is_ready()

model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')

with open("data/01_raw/ntrs-public-metadata.json", 'r', encoding="utf-8") as f:
    ntrs = json.load(f)

text = "What is the ration of fuel to weight necessary for lunar liftoff?"
text_embed = model.encode(text)

print(text_embed)

collection = client.collections.get("Chunks")

# response = collection.query.fetch_objects(limit=2, include_vector=True)

response = collection.query.near_vector(
    near_vector=text_embed.tolist(), # your query vector goes here
    limit=5,
    return_metadata=MetadataQuery(distance=True)
)

for o in response.objects:
    print(o.properties)
    print(o.metadata.distance)

client.close()
