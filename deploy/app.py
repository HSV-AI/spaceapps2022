import gradio as gr
import numpy as np
from sentence_transformers import util, SentenceTransformer
import pandas as pd


embeddings = np.load('data/05_model_input/embeddings.npy')
df = pd.read_parquet('data/04_feature/text.parquet')
model = SentenceTransformer('msmarco-distilbert-base-v4')
print(df.columns)
# TODO upgrade to cross-encoder after semantic search

def semantic_search(text, k):
    text_embed = model.encode(text)
    # NOTE using cosine similarity here, but can use other metrics
    hits = util.semantic_search(text_embed, embeddings, top_k=k)
    # print(hits)
    # scores = {}
    scores = []
    files = []
    texts = []
    for i, hit in enumerate(hits[0]):
        id = hit['corpus_id']
        # scores[i] = hit['score']
        scores.append(hit['score'])
        texts.append(df.iloc[id]['text'])
        files.append(df.iloc[id]['pdf_path'])

    # print(type(text))
    if len(texts) == 1:
        return *files, *scores, *texts
    return files, scores, texts

examples = [
    ["third rock from the sun", 1],
    ["fourth rock from the sun", 1],
    ["moons", 1],
    ["The Moon is a barren, rocky world without air or water. It has dark, mysterious craters and only one-eighth the gravity of Earth. It is the only place beyond Earth where humans have set foot. The Moon has been the goal of human exploration.", 1],
]

demo = gr.Interface(
    fn=semantic_search, 
    inputs=[gr.Text(label="Query"), gr.Slider(1, 10, 1, step=1, label='Number of results')],
    outputs=[gr.outputs.File("PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
    # outputs=[gr.outputs.File("PDFs"), gr.outputs.Label(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
    examples=examples,
)
demo.launch()   