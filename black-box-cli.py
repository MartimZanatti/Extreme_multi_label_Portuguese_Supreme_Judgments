import click
import json
from usage import descriptors_assignment
import sys
from labels import get_label_by_id


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.argument('file_extension')
@click.argument('area')
def black_box(filename, file_extension, area):



    labels = descriptors_assignment(filename, area, file_extension)

    pred_labels = {k: v for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)}




    json_descritores = []

    for key,value in pred_labels.items():
        json_descritores.append({"text": key, "score": value})





    print(json.dumps(json_descritores))





if __name__ == "__main__":
    black_box()