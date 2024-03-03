
from pathlib import Path

import random
import time
import requests as r

import click
import pandas as pd

random.seed(4)
BASE_URL = "https://ntrs.nasa.gov"

def make_request(url):
    print(f'Retrieving {url}')
    res = r.request("GET", url, timeout=60)
    time.sleep(1)
    if res.status_code != 200:
        print(f"Request failed with status code {res.status_code}")
        print(url)
    while res.status_code == 429:
        print("Rate limited, waiting {i} seconds")
        time.sleep(i)
        res = r.request("GET", url)
        i+=1
    return res

def get_file(url, key, ext, output_path):
    res = make_request(url)
    if res.status_code == 200:
        with open(output_path + f"/{key}.{ext}", 'wb') as f:
            f.write(res.content)

def get_data(df, output_dir):
    texts = df['text'].tolist()
    pdfs = df['pdf'].tolist()
    ids = df['id'].tolist()


    pdf_urls = [f"{BASE_URL}{file}" for id, file in zip(ids, pdfs)]
    text_urls = [f"{BASE_URL}{file}" for id, file in zip(ids, texts)]
    pdfs_dir = Path(output_dir + "/pdfs")
    pdfs_dir.mkdir(exist_ok=True)
    texts_dir = Path(output_dir + "/texts")
    texts_dir.mkdir(exist_ok=True)

    [get_file(url, id, "pdf", str(pdfs_dir)) for url, id in zip(pdf_urls, ids)]
    [get_file(url, id, "txt", str(texts_dir)) for url, id in zip(text_urls, ids)]


@click.command()
@click.argument("csv_path", type=click.Path())
@click.argument("output_dir", type=click.Path())
def main(csv_path, output_dir):
    df = pd.read_csv(csv_path, dtype={
        'id': 'int64',
        'pdf': 'string',
        'text': 'string'},
        na_filter = False)
    df = df[(df["pdf"] != "N/A") & (df["text"] != "N/A")]
    get_data(df, output_dir)


if __name__ == '__main__':
    main()
