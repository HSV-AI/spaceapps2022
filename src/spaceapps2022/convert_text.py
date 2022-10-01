import click

@click.command()
@click.argument('pdf_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def main(pdf_dir, output_dir):
    pass

if __name__ == '__main__':
    main()

