import gradio as gr
import numpy as np
from sentence_transformers import util, SentenceTransformer
import pandas as pd


embeddings = np.load('data/05_model_input/embeddings.npy')
data = pd.read_parquet('data/04_feature/text.parquet')
model = SentenceTransformer('msmarco-distilbert-base-v4')
# TODO upgrade to cross-encoder after semantic search

def semantic_search(text):
    text_embed = model.encode(text)
    # NOTE using cosine similarity here, but can use other metrics
    hits = util.semantic_search(text_embed, embeddings, top_k=1)
    print(hits)
    id = hits[0][0]['corpus_id']
    score = hits[0][0]['score']
    text = data.iloc[id]['text']
    print(type(text))
    return text, score


demo = gr.Interface(fn=semantic_search, inputs="text", outputs=[gr.outputs.Textbox(), gr.outputs.Label()])

demo.launch()   