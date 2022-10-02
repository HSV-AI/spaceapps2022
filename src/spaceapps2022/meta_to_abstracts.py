import click
import pandas as pd
from pathlib import Path
import json


@click.command()
@click.argument('meta_json', type=click.Path(exists=True))
@click.argument('output_path', type=click.STRING)
def main(meta_json, output_path):
    output_path = Path(output_path)

    with open(meta_json, 'r') as f:
        meta = json.load(f)
    
    # filter to only include papers with abstracts
    meta = {key: value  for key, value in meta.items() if 'abstract' in value.keys()}

    # filter to only include papers with titles
    meta = {key: value  for key, value in meta.items() if 'title' in value.keys()}

    abstracts = []
    titles = []
    ids = []
    for id, value in meta.items():
        ids.append(id)
        abstracts.append(value['abstract'])
        titles.append(value['title'])
    
    data = {
        'id': ids,
        'title': titles,
        'abstract': abstracts
    }
    df = pd.DataFrame(data)
    df.to_parquet(str(output_path), index=False)


if __name__ == '__main__':
    main()