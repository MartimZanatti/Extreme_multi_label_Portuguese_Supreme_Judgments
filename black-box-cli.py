import click
import json
from usage import descriptors_assignment


@click.command()
@click.argument('filename', 'section', type=click.Path(exists=True))
def black_box(filename, section):

    descriptors_json = descriptors_assignment(filename, section)
    print(descriptors_json)





if __name__ == "__main__":
    black_box()