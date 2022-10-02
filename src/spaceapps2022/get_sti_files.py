import click
import pandas as pd
from pathlib import Path

import json
import os
import pandas as pd
from pathlib import Path

import requests as r
from icecream import ic
from collections import defaultdict
import random
random.seed(4)
import time
import numpy as np


def make_request(url):
    # try:
    res = r.request("GET", url)
    i = 1
    if res.status_code != 200:
        print(f"Request failed with status code {res.status_code}")
        print(url)
    while res.status_code == 429:
        print("Rate limited, waiting {i} seconds")
        time.sleep(i)
        res = r.request("GET", url)
        i+=1
    return res

def get_file(url, id, ext, output_path):
    res = make_request(url)
    with open(output_path + f"/{id}.{ext}", 'wb') as f:
        f.write(res.content)

def get_data(df, output_dir):
    texts = df['text'].tolist()
    pdfs = df['pdf'].tolist()
    ids = df['id'].tolist()

    BASE_URL = "https://ntrs.nasa.gov"

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
    df = pd.read_csv(csv_path)
    get_data(df, output_dir)


if __name__ == '__main__':
    main()
