
from pathlib import Path
import json
from collections import defaultdict
import random
import time
import requests as r
import click
import pandas as pd

random.seed(4)

def get_metadata():
    print("Loading metadata from data/01_raw/ntrs-public-metadata.json")
    metadata_path = Path("data/01_raw/ntrs-public-metadata.json")
    with open(metadata_path, 'r', encoding="utf-8") as f:
        metadata = json.load(f)
    print("Done loading metadata")
    return metadata

def get_metadata_df():
    metadata = get_metadata()
    df = pd.DataFrame(metadata)
    return df

def get_metadata_df_by_year(year):
    df = get_metadata_df()
    df = df[df['year'] == year]
    return df


def make_request(url):
    print(f'Retrieving {url}')
    # try:
    res = r.request("GET", url)
    i = 1
    while res.status_code != 200:
        print(f"Request failed with status code {res.status_code}")
        if res.status_code == 429:
            print("Rate limited, waiting {i} seconds")
            time.sleep(i)
        res = r.request("GET", url)
        i+=1
    return res


def get_data(limit):
    meta = get_metadata()
    BASE_URL = "https://ntrs.nasa.gov/api/citations/"
    aval = 0
    not_aval = 0
    i = 0
    counts = defaultdict(int)

    keys = list(meta.keys())
    random.shuffle(keys)
    keys = keys[:limit]
    keys = sorted(keys)
    urls = [BASE_URL + key for key in keys]
    # urls = [BASE_URL + key + '/download' for key in keys]
    # from multiprocessing import Pool
    # pool = Pool(4)
    # results = pool.map(make_request, urls)
    results = [make_request(url) for url in urls]
    print(len(results))

    texts = []
    pdfs = []
    fals = 0
    ids = []

    for key, result in zip(keys, results):
        # print(result)
        # next
        if result.status_code != 200:
            # print("429")
            # print("Falure")
            print(result.status_code)
            # print(result)
            fals += 1
            continue

        data = result.json()
        counts[len(data['downloads'])] += 1
        if data['downloadsAvailable']:
            # print(data.keys())
            if len(data['downloads']) > 0:
                links = data['downloads'][0]['links']
                ids.append(key)
                if 'pdf' in links:
                    pdfs.append(links['pdf'])
                else:
                    pdfs.append("N/A")

                if "fulltext" in links:
                    texts.append(links['fulltext'])
                else:
                    texts.append("N/A")

                aval += 1
            else:
                print('error')
                not_aval += 1
        else:
            not_aval += 1
        i+=1
    return ids, pdfs, texts


@click.command()
@click.argument("parquet_path", type=click.Path())
@click.argument("limit", type=int)
def main(parquet_path, limit):
    ids, pdfs, texts = get_data(limit)
    df = pd.DataFrame({'id': ids, 'pdf': pdfs, 'text': texts})
    df.to_csv(parquet_path, index=False)


if __name__ == '__main__':
    main()
