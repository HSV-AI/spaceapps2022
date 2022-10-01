import click
import pandas as pd
from pathlib import Path


@click.command()
@click.argument('text_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.STRING)
def main(text_dir, output_path):
    text_dir = Path(text_dir)
    output_path = Path(output_path)

    text_paths = list(text_dir.glob('*.txt'))
    text_paths.sort()

    data = []
    for text_path in text_paths:
        text = text_path.read_text()
        data.append({'text': text})

    df = pd.DataFrame(data)
    print("here")
    df.to_parquet(str(output_path), index=False)


if __name__ == '__main__':
    main()