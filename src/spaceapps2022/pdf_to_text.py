import click
import PyPDF2
from PyPDF2 import PdfFileReader
from pathlib import Path

# def convert_pdf_to_text(pdf_path):
#     from pdfminer.high_level import extract_text
#     return extract_text(pdf_path)


def convert_pdf_to_text(pdf_path):
    open_pdf_file = open(pdf_path, 'rb')
    read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
    num_pages = read_pdf.getNumPages()


    ann_text = []
    for page_num in range(num_pages):
        if read_pdf.isEncrypted:
            read_pdf.decrypt("")
            # print(read_pdf.getPage(page_num).extractText())
        page_text = read_pdf.getPage(page_num).extractText()
        ann_text.append(page_text)
    return "\n".join(ann_text)

def extract_images(pdf_path, output_path):
    open_pdf_file = open(pdf_path, 'rb')
    read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
    num_pages = read_pdf.getNumPages()

    # Probably a better way to do this to handle the errors for some if the images.
    for page_num in range(num_pages):
        try:
            if read_pdf.isEncrypted:
                read_pdf.decrypt("")
            for image in read_pdf.getPage(page_num).images:
                    filepath = output_path / (pdf_path.stem + image.name)
                    with open(filepath, "wb") as fp:
                        fp.write(image.data)
        except Exception as e:
            print(e)

@click.command()
@click.argument('pdf_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(pdf_dir, output_dir):
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_dir = Path(output_dir / "images")
    image_dir.mkdir(exist_ok=True)

    for pdf_path in pdf_dir.glob('*.pdf'):
        text = convert_pdf_to_text(pdf_path)
        text_path = output_dir / (pdf_path.stem + '.txt')
        text_path.write_text(text)

        # Now extract the images from the PDF
        extract_images(pdf_path, image_dir)

if __name__ == '__main__':
    main()

