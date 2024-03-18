import click
import json
from usage import descriptors_assignment
import sys
from labels import get_label_by_id


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.argument('file_extension')
@click.argument('section')
def black_box(filename, file_extension, section="1_seccao"):



    labels_reversed, y = descriptors_assignment(filename, section, file_extension)

    json_descritores = []

    for yy in labels_reversed:
        label = get_label_by_id(yy, section)
        if y[yy] != 0:
            json_descritores.append({"text": label, "score": y[yy]})
        else:
            json_descritores.append({"text": label, "score": 0})




    print(json.dumps(json_descritores))





if __name__ == "__main__":
    black_box()