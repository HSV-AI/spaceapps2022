import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb

chroma_client = chromadb.PersistentClient(path="../data/05_model_input/chromadb")
collection = chroma_client.get_or_create_collection(name="spaceapps")

model = SentenceTransformer('BAAI/bge-small-en-v1.5')


# TODO upgrade to cross-encoder after semantic search
def semantic_search_abstracts(text, k):
    text_embed = model.encode(text)
    results = collection.query(
        query_embeddings=text_embed.tolist(),
        n_results=k
    )
    scores = []
    titles = []
    texts = []

    for id, score, doc in zip(results['ids'], results['distances'], results['documents']):
        scores.append(score)
        texts.append(doc)
        titles.append("unknown")

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
    ["mars gravity", 5],
    ["moments of inertia", 5],
    ["effective heat material", 5],
    ["lunar moisture content", 5],
    ["god of war", 2],
    ["Zeus", 1],
]

with gr.Blocks() as demo:
    gr.Markdown("Searching data from NASA STI Program")

    with gr.Tab("Titles and abstracts of all NASA STI Program"):
        gr.Markdown("This search compares against the titles and abstracts of the NASA STI Repository.")
        gr.Markdown("This makes use of the easily available meta data for STI documents (~500k).")
        with gr.Row():
            with gr.Column():
                input = [gr.Textbox(label="Query", value="third rock from the sun"),
                        gr.Slider(1, 10, 1, step=1, label='Number of results', value=1)]
                abs_button = gr.Button("Explore")
            with gr.Column():
                # output = [gr.outputs.File(label="PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")]
                output = [gr.components.Textbox(label="Titles"), gr.components.Textbox(label="Scores"), gr.components.Textbox(label="Abstracts")]

        abs_button.click(semantic_search_abstracts, input, output)
        gr.Examples(examples=examples, inputs=input)


    with gr.Tab("Subset of PDFs"):
        gr.Markdown("This search looks at the text in PDFs of the STI database.")
        gr.Markdown("The slow API makes it hard to get more PDFs. We were only able to scrape ~500 PDFs.")
        gr.Markdown("If it errors, that means something went wrong with the PDF download.")
        with gr.Row():
            with gr.Column():
                input = [gr.Textbox(label="Query", value="third rock from the sun"),
                        gr.Slider(1, 10, 1, step=1, label='Number of results', value=1)]
                pdf_button = gr.Button("Explore")
            with gr.Column():
                output = [gr.components.File(label="PDFs"), gr.components.Textbox(label="Scores"), gr.components.Textbox(label="Raw Text")]

        pdf_button.click(semantic_search, input, output)
        gr.Examples(examples=examples, inputs=input)

# demo = gr.Interface(
#     fn=semantic_search, 
#     inputs=[gr.Text(label="Query"), gr.Slider(1, 10, 1, step=1, label='Number of results')],
#     outputs=[gr.outputs.File("PDFs"), gr.outputs.Textbox(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
#     # outputs=[gr.outputs.File("PDFs"), gr.outputs.Label(label="Scores"), gr.outputs.Textbox(label="Raw Text")],
#     examples=examples,
# )
demo.launch(server_name="0.0.0.0")   