import time
import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb
import json
import pandas as pd

with open("data/01_raw/ntrs-public-metadata.json", 'r') as f:
    ntrs = json.load(f)

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
abstract_collection = chroma_client.get_or_create_collection(name="spaceapps")
chunk_collection = chroma_client.get_or_create_collection(name="chunks")

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def semantic_search_abstracts(text, k):
    print(f'Creating embedding for {text}')
    start = time.perf_counter_ns()
    text_embed = model.encode(text)
    embed_stop = time.perf_counter_ns()
    print(f'Embedding time: {(embed_stop - start)/1e6:0.4f}ms')
    print('Running query')
    results = abstract_collection.query(
        query_embeddings=text_embed.tolist(),
        n_results=k
    )
    query_stop = time.perf_counter_ns()
    print(f'Query time: {(query_stop - embed_stop)/1e6:0.4f}ms')
    scores = []
    titles = []
    texts = []

    for doc_id, score, doc in zip(results['ids'], results['distances'], results['documents']):
        scores.append(score)
        texts.append(doc)
        titles.append("unknown")

    # print(type(text))
    if len(texts) == 1:
        return *titles, *scores, *texts
    return titles, scores, texts


def semantic_search(text, k):
    print(f'Creating embedding for {text}')
    start = time.perf_counter_ns()
    text_embed = model.encode(text)
    embed_stop = time.perf_counter_ns()
    print(f'Embedding time: {(embed_stop - start)/1e6:0.4f}ms')
    print('Running query')
    results = chunk_collection.query(
        query_embeddings=text_embed.tolist(),
        n_results=k
    )
    query_stop = time.perf_counter_ns()
    print(f'Query time: {(query_stop - embed_stop)/1e6:0.4f}ms')
    scores = []
    titles = []
    texts = []

    for distance, document, meta in zip(results['distances'][0], results['documents'][0], results['metadatas'][0]):
        key = meta['key']
        print(key)
        title = ntrs[key]['title']
        # title = df.loc[df['id'] == key].to_dict(orient='records')[0]['title']
        # print(f'{key} Document: {title}, page {meta["page"]} - "{document}"\n')
        scores.append(distance)
        texts.append(document)
        titles.append(f'{title}, page {meta["page"]}')

    df = pd.DataFrame(
        {'Title': titles,
        'Score': scores,
        'Text': texts
        })

    return df
    # if len(texts) == 1:
    #     return *titles, *scores, *texts
    # return titles, scores, texts




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
        # with gr.Row():
            # with gr.Column():
        input = [gr.Textbox(label="Query", value="third rock from the sun"),
                gr.Slider(1, 10, 1, step=1, label='Number of results', value=1)]
        pdf_button = gr.Button("Explore")
            # with gr.Column():
        output = [gr.Dataframe(row_count = (2, "dynamic"), col_count=(3, "fixed"), label="Matches")]
                # output = [gr.components.Textbox(label="Titles"), gr.components.Textbox(label="Scores"), gr.components.Textbox(label="Raw Text")]

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