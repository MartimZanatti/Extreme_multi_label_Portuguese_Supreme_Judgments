import bilstm_crf
from Judgment import Judgment
from bilstm_utils import id2word
from utils import embedding_judgment
import pickle
from ensemble import Ensemble, Model
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import sys
from labels import return_labels

import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def descriptors_assignment(doc_name, area, file_extension, device="cpu"):

    if file_extension == ".docx":
        type = "docx"
    elif file_extension == ".html":
        type = "html"
    elif file_extension == ".txt":
        type = "text"

    output = get_sections(doc_name, type=type)

    final_text = get_judgment_nucleo_text(output)



    emb_text = embedding_judgment(final_text, device)

    emb_text = emb_text.to_dense()
    emb_text = emb_text.numpy()
    emb_text_sparse = csr_matrix(emb_text)

    models = []
    ys = []

    X_labels = {}

    section_labels = []





    if area == "civel": #se a area for civel, vamos buscar os modelos da 1,2,6,7 seccao
        y_train_1_file = open("data/1_seccao/y_train_numpy.pkl", "rb")
        y_train_1 = pickle.load(y_train_1_file)
        ys.append(y_train_1)
        section_labels.append(return_labels("1_seccao"))


        with open("data/1_seccao/models.pkl", 'rb') as f:
            learners_1 = pickle.load(f)

        models.append(learners_1)

        y_train_2_file = open("data/2_seccao/y_train_numpy.pkl", "rb")
        y_train_2 = pickle.load(y_train_2_file)
        ys.append(y_train_2)
        section_labels.append(return_labels("2_seccao"))

        with open("data/2_seccao/models.pkl", 'rb') as f:
            learners_2 = pickle.load(f)

        models.append(learners_2)

        y_train_6_file = open("data/6_seccao/y_train_numpy.pkl", "rb")
        y_train_6 = pickle.load(y_train_6_file)
        ys.append(y_train_6)
        section_labels.append(return_labels("6_seccao"))

        with open("data/6_seccao/models.pkl", 'rb') as f:
            learners_6 = pickle.load(f)

        models.append(learners_6)

        y_train_7_file = open("data/7_seccao/y_train_numpy.pkl", "rb")
        y_train_7 = pickle.load(y_train_7_file)
        ys.append(y_train_7)
        section_labels.append(return_labels("7_seccao"))

        with open("data/7_seccao/models.pkl", 'rb') as f:
            learners_7 = pickle.load(f)

        models.append(learners_7)

        y_train_civel_file = open("data/civel/y_train_numpy.pkl", "rb")
        y_train_civel = pickle.load(y_train_civel_file)
        ys.append(y_train_civel)
        section_labels.append(return_labels("civel"))

        with open("data/civel/models.pkl", 'rb') as f:
            learners_civel = pickle.load(f)

        models.append(learners_civel)

    elif area == "criminal": #se a area for civel, vamos buscar os modelos da 3,5 seccao
        y_train_3_file = open("data/3_seccao/y_train_numpy.pkl", "rb")
        y_train_3 = pickle.load(y_train_3_file)
        ys.append(y_train_3)
        section_labels.append(return_labels("3_seccao"))


        with open("data/3_seccao/models.pkl", 'rb') as f:
            learners_3 = pickle.load(f)

        models.append(learners_3)

        y_train_5_file = open("data/5_seccao/y_train_numpy.pkl", "rb")
        y_train_5 = pickle.load(y_train_5_file)
        ys.append(y_train_5)
        section_labels.append(return_labels("5_seccao"))

        with open("data/5_seccao/models.pkl", 'rb') as f:
            learners_5 = pickle.load(f)

        models.append(learners_5)

        y_train_criminal_file = open("data/criminal/y_train_numpy.pkl", "rb")
        y_train_criminal = pickle.load(y_train_criminal_file)
        ys.append(y_train_criminal)
        section_labels.append(return_labels("criminal"))

        with open("data/criminal/models.pkl", 'rb') as f:
            learners_criminal = pickle.load(f)

        models.append(learners_criminal)

    elif area == "social": #se a area for civel, vamos buscar os modelos da 4 seccao
        y_train_4_file = open("data/4_seccao/y_train_numpy.pkl", "rb")
        y_train_4 = pickle.load(y_train_4_file)
        ys.append(y_train_4)
        section_labels.append(return_labels("4_seccao"))


        with open("data/4_seccao/models.pkl", 'rb') as f:
            learners_4 = pickle.load(f)

        models.append(learners_4)

    elif area == "contencioso": #se a area for civel, vamos buscar os modelos da 3,5 seccao
        y_train_cont_file = open("data/contencioso/y_train_numpy.pkl", "rb")
        y_train_cont = pickle.load(y_train_cont_file)
        ys.append(y_train_cont)
        section_labels.append(return_labels("contencioso"))


        with open("data/contencioso/models.pkl", 'rb') as f:
            learners_cont = pickle.load(f)

        models.append(learners_cont)


    for i,model in enumerate(models):
        models = [Model(learner, ys[i])
                  for learner in model]
        ensemble = Ensemble(models)

        pred_y = ensemble.predict_one(emb_text_sparse)

        for j,y in enumerate(pred_y):
            label = section_labels[i][j]
            if label not in X_labels:
                X_labels[label] = y
            else:
                X_labels[label] += y




        return X_labels















    """
    with open("data/" + section + "/models.pkl", 'rb') as f:
        learners = pickle.load(f)


    models = [Model(learner, y_train)
              for learner in learners]
    ensemble = Ensemble(models)


    y = ensemble.predict_one(emb_text_sparse)

    y_ids = np.argsort(y)

    reverse_y_ids = np.flip(y_ids)


    return reverse_y_ids, y
    """





def get_sections(doc_name, type="docx", device="cpu"):
    doc = Judgment(doc_name, type, False)

    model = bilstm_crf.BiLSTMCRF.load("descritores.pth", device)
    model.eval()
    all_text, ids, text_ids = doc.get_list_text()
    output = {"wrapper": "plaintext", "text": text_ids, "denotations": []}

    #secções pela ordem mais importante
    sections_doc = {"fundamentação de direito": [], "fundamentação de facto": [], "relatório": [], "decisão": [], "delimitação": [], "colectivo": [], "declaração": [], "cabeçalho": [], "foot-note": [], "título": []}
    sections = model.get_sections(all_text, device)
    #print(sections)
    sections = sections[0][1:-1]
    sections_names = []
    for tag in sections:
        sections_names.append(id2word(tag))

    for i, section in enumerate(sections_names):
        if section in ["B-cabeçalho", "I-cabeçalho"]:
            sections_doc["cabeçalho"].append((section, ids[i]))
        elif section in ["B-relatório", "I-relatório"]:
            sections_doc["relatório"].append((section, ids[i]))
        elif section in ["B-delimitação", "I-delimitação"]:
            sections_doc["delimitação"].append((section, ids[i]))
        elif section in ["B-fundamentação-facto", "I-fundamentação-facto"]:
            sections_doc["fundamentação de facto"].append((section, ids[i]))
        elif section in ["B-fundamentação-direito", "I-fundamentação-direito"]:
            sections_doc["fundamentação de direito"].append((section, ids[i]))
        elif section in ["B-decisão", "I-decisão"]:
            sections_doc["decisão"].append((section, ids[i]))
        elif section in ["B-colectivo", "I-colectivo"]:
            sections_doc["colectivo"].append((section, ids[i]))
        elif section in ["B-declaração", "I-declaração"]:
            sections_doc["declaração"].append((section, ids[i]))
        elif section in ["B-foot-note", "I-foot-note"]:
            sections_doc["foot-note"].append((section, ids[i]))
        elif section == "título":
            sections_doc["título"].append((section, ids[i]))

    id = 0
    for key, value in sections_doc.items():
        if len(value) != 0:
            if key in ["cabeçalho", "relatório", "delimitação", "fundamentação de facto", "fundamentação de direito", "decisão", "foot-note"]:
                output["denotations"].append(
                    {"id": id, "start": value[0][1][0], "end": value[-1][1][0], "start_char": value[0][1][1],
                     "end_char": value[-1][1][2], "type": key})
            else:
                zones = []
                for v in value:
                    zones.append(v[1])

                output["denotations"].append({"id": id, "zones": zones, "type": key})
            id += 1



    return output





def get_judgment_nucleo_text(output):
    text_list = []
    ids = []
    ids_used = []
    denotations = output["denotations"]
    for d in denotations:
        if d["type"] in ["relatório", "fundamentação", "decisão"]:
            ids.append((d["start"], d["end"]))
    text = output["text"]


    for p in text:
        for i in ids:
            if int(i[0]) <= int(p[1]) <= int(i[1]):
                if p[1] not in ids_used:
                    text_list.append(p[0])
                    ids_used.append(p[1])

    if text_list == []:
        sys.stderr.write("não encontrou nenhuma das seguintes seccões: relatório, fundamentação, decisão")
        sys.stderr.write("vamos retornar o texto todo")
        for t in text:
            text_list.append(t[0])


    return text_list


descriptors_assignment("../IrisDataset/test_examples/teste.txt", "civel", ".txt")

