import gradio as gr
import numpy as np
from sentence_transformers import util, SentenceTransformer
import pandas as pd


embeddings = np.load('data/05_model_input/embeddings.npy')
abs_embeddings = np.load('data/05_model_input/abstract_embeddings.npy')
df = pd.read_parquet('data/04_feature/text.parquet')
abs_df = pd.read_parquet('data/04_feature/abstracts.parquet')
model = SentenceTransformer('msmarco-distilbert-base-v4')


# TODO upgrade to cross-encoder after semantic search
def semantic_search_abstracts(text, k):
    text_embed = model.encode(text)
    # NOTE using cosine similarity here, but can use other metrics
    hits = util.semantic_search(text_embed, abs_embeddings, top_k=k)
    scores = []
    titles = []
    texts = []
    for i, hit in enumerate(hits[0]):
        id = hit['corpus_id']
        scores.append(hit['score'])
        texts.append(abs_df.iloc[id]['abstract'])
        titles.append(abs_df.iloc[id]['title'])

    # print(type(text))
    if len(texts) == 1:
        return *titles, *scores, *texts
    return titles, scores, texts


def semantic_search(text, k):
    text_embed = model.encode(text)
    # NOTE using cosine similarity here, but can use other metrics
    hits = util.semantic_search(text_embed, embeddings, top_k=k)
    scores = []
    files = []
    texts = []
    for i, hit in enumerate(hits[0]):
        id = hit['corpus_id']
        scores.append(hit['score'])
        texts.append(df.iloc[id]['text'])
        files.append(df.iloc[id]['pdf_path'])

    if len(texts) == 1:
        return *files, *scores, *texts
    return files, scores, texts


examples = [
    ["third rock from the sun", 1],
    ["god of war", 2],
    ["mars gravity", 1],
    ["Zeus", 1],
    ["The Moon is a barren, rocky world without air or water. It has dark, mysterious craters and only one-eighth the gravity of Earth. It is the only place beyond Earth where humans have set foot. The Moon has been the goal of human exploration.", 1],
]

with gr.Blocks() as demo:
    gr.Markdown("Searching data from NASA STI Program")
    with gr.Tab("Subset of PDFs (500)"):
        gr.Markdown("The slow API makes it hard to get more PDFs")
        with gr.Row():
            with gr.Column():
                input = [gr.Textbox(label="Query", default="third rock from the sun"),
                        gr.Slider(1, 10, 1, step=1, label='Number of results', default=1)]
            with gr.Column():
                output = [gr.outputs.File(label="PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")]
        pdf_button = gr.Button("Search")

        pdf_button.click(semantic_search, input, output)
        gr.Examples(examples=examples, inputs=input)

    with gr.Tab("Titles and abstracts of all NASA STI Program"):
        with gr.Row():
            with gr.Column():
                input = [gr.Textbox(label="Query", default="third rock from the sun"),
                        gr.Slider(1, 10, 1, step=1, label='Number of results', default=1)]
            with gr.Column():
                # output = [gr.outputs.File(label="PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")]
                output = [gr.outputs.Textbox(label="Titles"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Abstracts")]
        abs_button = gr.Button("Search")

        abs_button.click(semantic_search_abstracts, input, output)
        gr.Examples(examples=examples, inputs=input)


# demo = gr.Interface(
#     fn=semantic_search, 
#     inputs=[gr.Text(label="Query"), gr.Slider(1, 10, 1, step=1, label='Number of results')],
#     outputs=[gr.outputs.File("PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
#     # outputs=[gr.outputs.File("PDFs"), gr.outputs.Label(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
#     examples=examples,
# )
demo.launch(share=True)   