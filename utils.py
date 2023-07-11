import os
import json
from elasticsearch import Elasticsearch
import pandas as pd
import pickle
import xlsxwriter
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
import torch
from emb_model import load_bert_model
from labels import *
from scipy.sparse import issparse


client = Elasticsearch("http://127.0.0.1:9200")


# recebe json com a estrutura dos acordaos e retorna o texto do relatório - fundamentação - decisão
def get_judgment_nucleo_text(path, file_name):
    f = open(path + file_name)
    text_list = []
    ids = []
    data = json.load(f)
    denotations = data["denotations"]
    for d in denotations:
        if d["type"] in ["relatório", "fundamentação", "decisão"]:
            ids.append((d["start"], d["end"]))
    text = data["text"]

    for p in text:
        for i in ids:
            if int(i[0]) <= int(p[1]) <= int(i[1]):
                text_list.append(p[0])

    return text_list

# funcao que vai buscar os acordaos ao elastic search e cria um dataframe panda
def get_descritores_by_id():
    path = "../IrisDataset/new_json_elastic/"
    files = os.listdir(path)
    dic = {"1.ª Secção (Cível)": {"file_name": [], "text": [], "labels": []}, "2.ª Secção (Cível)": {"file_name": [], "text": [], "labels": []},
           "3.ª Secção (Criminal)": {"file_name": [], "text": [], "labels": []}, "4.ª Secção (Social)": {"file_name": [], "text": [], "labels": []},
           "5.ª Secção (Criminal)": {"file_name": [], "text": [], "labels": []}, "6.ª Secção (Cível)": {"file_name": [], "text": [], "labels": []},
           "7.ª Secção (Cível)": {"file_name": [], "text": [], "labels": []}, "Contencioso": {"file_name": [], "text": [], "labels": []},
           "«sem secção»": {"file_name": [], "text": [], "labels": []}}

    duplicate_des = []

    for i,file_name in enumerate(files):
        print(i)
        final_descritores = []
        id = file_name[:-5]
        r = client.get(index="jurisprudencia.9.4", id=id)
        descritores = r["_source"]["Descritores"]
        seccao = r["_source"]["Secção"]
        if "{}" not in descritores:
            new_descritores = set(descritores)
            if len(new_descritores) != len(descritores):
                duplicate_des.append(id)
            for d in new_descritores:
                if not d.startswith("`"):
                    final_descritores.append(d)

            dic[seccao]["file_name"].append(id)
            text = get_judgment_nucleo_text(path, file_name)
            dic[seccao]["text"].append(text)
            dic[seccao]["labels"].append(final_descritores)


    for key, values in dic.items():
        df = pd.DataFrame(values)
        print(df)
        with open("dataframe" + key + ".pkl", 'wb') as f:
            pickle.dump(df, f)


def stat_descritores():
    file = open("data/1_seccao/dataframe1noFew.pkl", "rb")
    df = pickle.load(file)
    stat_dic = {}

    workbook = xlsxwriter.Workbook("data/1_seccao/zones_stats_few_1_seccao.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', "Descritor")
    worksheet.write('B1', "Acordaos")
    worksheet.write("C1", "Quantidade")

    for i, row in enumerate(df.iloc):
        des = row["labels"]
        ac = row["file_name"]
        for d in des:
            if d not in stat_dic:
                stat_dic[d] = [ac]
            else:
                stat_dic[d].append(ac)

    row = 1
    for key,value in stat_dic.items():
        column = 0
        worksheet.write(row, column, key)
        column += 1
        worksheet.write(row, column, str(value))
        column += 1
        worksheet.write(row, column, len(value))
        row += 1

    workbook.close()


def delete_few_descritores():
    file = open("data/1_seccao/dataframe1.ª Secção (Cível).pkl", "rb")
    df = pickle.load(file)
    dic = {"file_name": [], "text": [], "labels": []}
    stat_dic = {}
    delete_descritores = []

    for i, row in enumerate(df.iloc):
        des = row["labels"]
        ac = row["file_name"]
        for d in des:
            if d not in stat_dic:
                stat_dic[d] = [ac]
            else:
                stat_dic[d].append(ac)

    for key, value in stat_dic.items():
        if len(value) < 3:
            delete_descritores.append(key)


    for i, row in enumerate(df.iloc):
        new_descritores = []
        des = row["labels"]
        text = row["text"]
        file_name = row["file_name"]
        for d in des:
            if d not in delete_descritores:
                new_descritores.append(d)
        if len(new_descritores) != 0:
            dic["file_name"].append(file_name)
            dic["text"].append(text)
            dic["labels"].append(new_descritores)

    df = pd.DataFrame(dic)
    print(df)
    with open("data/1_seccao/dataframe1noFew.pkl", 'wb') as f:
        pickle.dump(df, f)


def get_descritores_list():
    file = open("data/1_seccao/dataframe1noFew.pkl", "rb")
    df = pickle.load(file)
    descritores = []
    for i, row in enumerate(df.iloc):
        des = row["labels"]
        for d in des:
            if d not in descritores:
                descritores.append(d)

    sorted_list = sorted(descritores)

    print(sorted_list)

def create_panda_all_labels():
    file = open("data/Contencioso/dataframeContenciosonoFew.pkl", "rb")
    df = pickle.load(file)
    dic = {"file_name": [], "text": []}

    labels = return_labels("contencioso")

    print(labels)
    for d in labels:
        dic[d] = []

    for i, row in enumerate(df.iloc):
        dic["file_name"].append(row["file_name"])
        dic["text"].append(row["text"])

        des = row["labels"]

        for de in labels:
            if de not in des:
                dic[de].append(0)
            else:
                dic[de].append(1)


    df = pd.DataFrame(dic)
    print(df)
    with open("data/Contencioso/dataframe_contencioso_all_labels.pkl", 'wb') as f:
        pickle.dump(df, f)


def delete_acordaos_no_text():
    file = open("data/1_seccao/dataframe_1_seccao_all_labels.pkl", "rb")
    df = pickle.load(file)

    print("initial df", df)

    zero_indxs = []

    equal_indxs = []

    for i, row in enumerate(df.iloc):
        text_list = row["text"]
        if len(text_list) <= 0:
            zero_indxs.append(i)

    for id in zero_indxs:
        df.drop(index=id, inplace=True)

    df = df.reset_index(drop=True)

    print("df after no text", df)


    for i, row in enumerate(df.iloc):
        print("i", i)
        for j, row_2 in enumerate(df.iloc):
            if j > i:
                text = row["text"]
                text_2 = row_2["text"]
                labels = row[2:].tolist()
                labels_2 = row_2[2:].tolist()
                if text == text_2 and labels == labels_2:
                    equal_indxs.append(j)



    for ind in equal_indxs:
        df.drop(index=ind, inplace=True)

    df = df.reset_index(drop=True)

    print("df after equal text", df)


    with open("data/1_seccao/dataframe_1_seccao_all_labels.pkl", 'wb') as f:
        pickle.dump(df, f)


def divide_data_statified():
    file = open("data/1_seccao/dataframe_1_seccao_all_labels.pkl", "rb")
    df = pickle.load(file)


    y = df[df.columns[2:]].values

    x = df["text"].values.reshape(len(df), 1)
    print(x.shape)
    print(y.shape)


    x_train, y_train, x_test, y_test = iterative_train_test_split(x,y, test_size=0.2)

    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    with open("data/1_seccao/X_train.pkl", 'wb') as f:
        pickle.dump(x_train, f)

    with open("data/1_seccao/X_test.pkl", 'wb') as f:
        pickle.dump(x_test, f)

    with open("data/1_seccao/y_train.pkl", 'wb') as f:
        pickle.dump(y_train, f)

    with open("data/1_seccao/y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)


def create_pandas_stratified_divisions():
    file = open("data/1_seccao/dataframe_1_seccao_all_labels.pkl", "rb")
    df = pickle.load(file)
    train = {"file_name": [], "text": [], "labels": []}
    test = {"file_name": [], "text": [], "labels": []}

    print(df)

    x_train_file = open("data/1_seccao/X_train.pkl", "rb")
    x_train = pickle.load(x_train_file)


    y_train_file = open("data/1_seccao/y_train.pkl", "rb")
    y_train = pickle.load(y_train_file)


    x_test_file = open("data/1_seccao/X_test.pkl", "rb")
    x_test = pickle.load(x_test_file)


    y_test_file = open("data/1_seccao/y_test.pkl", "rb")
    y_test = pickle.load(y_test_file)

    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    for i, row in enumerate(df.iloc):
        found = False
        text = row["text"]
        file_name = row["file_name"]
        labels = row[2:].tolist()


        for j,t in enumerate(x_train[:]):
            y_labels = y_train[j].tolist()
            if text == t[0] and labels == y_labels:
                train["file_name"].append(file_name)
                train["text"].append(text)
                train["labels"].append(y_train[j])
                found = True
                break

        for w,tt in enumerate(x_test[:]):
            y_labels_test = y_test[w].tolist()
            if text == tt[0] and labels == y_labels_test:
                if found == False:
                    test["file_name"].append(file_name)
                    test["text"].append(text)
                    test["labels"].append(y_test[w])
                    break
                else:
                    print(row["file_name"])

    train_df = pd.DataFrame(train)
    print(train_df)
    with open("data/1_seccao/data_train.pkl", 'wb') as f:
        pickle.dump(train_df, f)

    test_df = pd.DataFrame(test)
    print(test_df)
    with open("data/1_seccao/data_test.pkl", 'wb') as f:
        pickle.dump(test_df, f)


def create_embeddings(args):
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    if device == 'cuda':
        torch.cuda.set_device(int(args["--device-number"]))
    file = open(args["ARGUMENTS"][0], "rb")
    df = pickle.load(file)

    y_file = open(args["ARGUMENTS"][1], "rb")
    y = pickle.load(y_file)

    indxs = df.index.tolist()

    assert len(indxs) == len(y)

    model_name_bert = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    model_emb = load_bert_model(model_name_bert, device)
    X = ''

    for i, row in enumerate(df.iloc):
        print("i", i)
        text_list = row["text"]
        X_aux = torch.zeros(1024)
        for j,p in enumerate(text_list):
            emb = torch.from_numpy(model_emb.encode(p)) #-> (1024) -> X[i,j]
            if emb.sum().data == 0:
                emb = torch.from_numpy(model_emb.encode("UNK"))
                X_aux = X_aux + emb
            else:
                X_aux = X_aux + emb

        assert len(text_list) > 0
        X_aux = torch.div(X_aux, len(text_list))
        X_aux = X_aux.to_sparse()
        X_aux = torch.unsqueeze(X_aux, 0)
        if i == 0:
            X = X_aux
        else:
            X = torch.cat((X, X_aux), 0)


    with open(args["ARGUMENTS"][2], 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)


def embedding_judgment(text, device):
    model_name_bert = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    model_emb = load_bert_model(model_name_bert, device)
    X = torch.zeros(1024)

    for p in text:
        emb = torch.from_numpy(model_emb.encode(p))
        if emb.sum().data == 0:
            emb = torch.from_numpy(model_emb.encode("UNK"))
            X = X + emb
        else:
            X = X + emb

    X = torch.div(X, len(text))
    X = X.to_sparse()

    return X


def project(V, x):
    """
    V: (embed dim, feature dim)
    x: (feature_dim, )

    return:
    vector of (embed dim, )
    """
    if issparse(x):
        x = x.toarray().T
    if len(x.shape) == 1:
        x = x[:, None]

    return (V @ x).flatten()


def precision_at_ks(true_Y, pred_Y, ks=[1, 2, 3, 4, 5, 10, 15]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / min(k, len(t))
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = np.mean(precs)
    return result




