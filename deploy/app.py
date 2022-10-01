import gradio as gr
import numpy as np
from sentence_transformers import util, SentenceTransformer
import pandas as pd


embeddings = np.load('data/05_model_input/embeddings.npy')
df = pd.read_parquet('data/04_feature/text.parquet')
model = SentenceTransformer('msmarco-distilbert-base-v4')
print(df.columns)
# TODO upgrade to cross-encoder after semantic search

def semantic_search(text):
    text_embed = model.encode(text)
    # NOTE using cosine similarity here, but can use other metrics
    hits = util.semantic_search(text_embed, embeddings, top_k=1)
    # print(hits)
    id = hits[0][0]['corpus_id']
    score = hits[0][0]['score']
    text = df.iloc[id]['text']
    file = df.iloc[id]['pdf_path']

    # print(type(text))
    return file, score, text


demo = gr.Interface(
    fn=semantic_search, 
    inputs="text", 
    outputs=[gr.outputs.File(), gr.outputs.Label(), gr.outputs.Textbox()],
)
demo.launch()   