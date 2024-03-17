import pandas as pd
from pathlib import Path
import click
from llmsherpa.readers import LayoutPDFReader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)


# def extract_images(pdf_path, output_path):
#     open_pdf_file = open(pdf_path, 'rb')
#     read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
#     num_pages = read_pdf.getNumPages()

#     # Probably a better way to do this to handle the errors for some if the images.
#     for page_num in range(num_pages):
#         try:
#             if read_pdf.isEncrypted:
#                 read_pdf.decrypt("")
#             for image in read_pdf.getPage(page_num).images:
#                     filepath = output_path / ".." / (pdf_path.stem + image.name)
#                     with open(filepath, "wb") as fp:
#                         fp.write(image.data)
#         except Exception as e:
#             print(e)

# @click.command()
# @click.argument('pdf_dir', type=click.Path(exists=True))
# @click.argument('output_dir', type=click.Path())
def main(pdf_dir, output_path):
    pdf_dir = Path(pdf_dir)

    ids = []
    pages = []
    chunks = []

    try:
        for pdf_path in pdf_dir.glob('*.pdf'):
            filepath = str(pdf_path)
            print(f'Parsing: {filepath}')
            key = pdf_path.stem
            doc = pdf_reader.read_pdf(filepath)
            for chunk in doc.chunks():
                text = chunk.to_text()
                chunks.append(text)
                ids.append(key)
                pages.append(chunk.page_idx)
            # Now extract the images from the PDF
            # output_dir = Path(output_dir)
            # output_dir.mkdir(exist_ok=True)

            # image_dir = Path(output_dir / "images")
            # image_dir.mkdir(exist_ok=True)

            # extract_images(pdf_path, image_dir)
    except Exception as exc:
        print(f'Stopping early because of: {exc}')

    data = {
        'id': ids,
        'page': pages,
        'chunk': chunks
    }
    df = pd.DataFrame(data)
    df.to_parquet(str(output_path), index=False)

if __name__ == '__main__':
    main("data/01_raw/pdfs", "data/04_feature/chunks.parquet")
