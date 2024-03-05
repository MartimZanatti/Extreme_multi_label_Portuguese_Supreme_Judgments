import bilstm_crf
from Judgment import Judgment
from bilstm_utils import id2word
from utils import embedding_judgment
import pickle
from ensemble import Ensemble, Model
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from labels import get_label_by_id
import json

def usage(doc_name, section, device="cpu"):
    output = get_sections(doc_name)

    final_text = get_judgment_nucleo_text(output)

    labels_output = {"labels": [], "scores": []}

    emb_text = embedding_judgment(final_text, device)

    emb_text = emb_text.to_dense()
    emb_text = emb_text.numpy()
    emb_text_sparse = csr_matrix(emb_text)


    y_train_file = open("data/" + section + "/y_train_numpy.pkl", "rb")
    y_train = pickle.load(y_train_file)


    with open("data/" + section + "/models.pkl", 'rb') as f:
        learners = pickle.load(f)


    models = [Model(learner, y_train)
              for learner in learners]
    ensemble = Ensemble(models)


    y = ensemble.predict_one(emb_text_sparse)

    y_ids = np.argsort(y)

    reverse_y_ids = np.flip(y_ids)

    labels_list = []
    for yy in reverse_y_ids:
        label = get_label_by_id(yy, section)
        labels_list.append(label)
        labels_output["labels"].append(label)
        labels_output["scores"].append(y[yy])


    out = json.dumps(labels_output)

    return out






def get_sections(doc_name, type="docx", device="cpu"):
    doc = Judgment(doc_name, type, False)

    model = bilstm_crf.BiLSTMCRF.load("JudgmentModel/with_fundamentacao_separation/model.pth", device)
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
        print("não encontrou nenhuma das seguintes seccões: relatório, fundamentação, decisão")
        print("vamos retornar o texto todo")
        for t in text:
            text_list.append(t[0])


    return text_list


usage("../IrisDataset/test_examples/Ac 1249-16.0JAPRT.P1.S1.docx", "5_seccao")