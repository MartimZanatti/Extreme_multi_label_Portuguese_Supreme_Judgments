import pickle
from matplotlib import pyplot as plt
import os
import pandas as pd

def histograma_descritores():
    x_train_file = open("data/criminal/criminal.pkl", "rb")
    x_train = pickle.load(x_train_file)

    print(x_train)

    des_dict = {}

    for i,row in enumerate(x_train.iloc):
        descritores = row["descritores"]

        for d in descritores:
            if d not in des_dict:
                des_dict[d] = 1
            else:
                des_dict[d] += 1


    contagem_descritores = {}

    for key,num_descritores in des_dict.items():
        if num_descritores in contagem_descritores:
            contagem_descritores[num_descritores] += 1
        else:
            contagem_descritores[num_descritores] = 1



    plt.bar(contagem_descritores.keys(), contagem_descritores.values(), color='blue')
    plt.xlabel('Descriptor Frequency')
    plt.ylabel('Total number of unique Descriptors')
    plt.title('Distribution of the Number of Descriptors Appearing a Number of Times in the Judgments.')
    plt.yscale('log')  # Aplicando escala logarítmica no eixo y
    plt.show()





