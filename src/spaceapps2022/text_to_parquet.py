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
        pdf_path = "data/01_raw/pdfs/" + text_path.stem + ".pdf"
        data.append({
            'text': text,
            'text_path': str(text_path),
            'pdf_path': str(pdf_path)
        })

    df = pd.DataFrame(data)
    df.to_parquet(str(output_path), index=False)


if __name__ == '__main__':
    main()