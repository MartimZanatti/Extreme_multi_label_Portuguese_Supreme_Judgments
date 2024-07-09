from elasticsearch import Elasticsearch
import pickle
import pandas as pd
from Judgment import Judgment
import re
from labels import get_official_descritores, return_labels
import xlsxwriter
from skmultilearn.model_selection import iterative_train_test_split
from nltk import ngrams

def get_docs_elastic_search():
    client = Elasticsearch("http://127.0.0.1:9200", http_auth=('elastic', 'elasticsearch'))
    offset = 0
    size = 50
    acordaos_path = "../IrisDataset/elastic_acordaos/"
    elastic_dic = {"id": [], "section": [], "descritores": [], "text": []}

    while True:
        results = client.search(index="jurisprudencia.11.0", body={"from": offset, "size": size})
        hits = results["hits"]["hits"]
        print("offset", offset)
        if not hits:
            break
        offset += size
        for hit in hits:
            #print(hit)
            text = hit["_source"]["Texto"]
            #print(text)
            if text is None or len(text) < 700:
                continue
            descritores = hit["_source"]["Descritores"]["Index"]
            if descritores[0] == '{}':
                print("descritores", descritores)
                continue
            elastic_dic["id"].append(hit["_id"])
            elastic_dic["section"].append(hit["_source"]["Secção"]["Show"][0])
            elastic_dic["descritores"].append(hit["_source"]["Descritores"]["Index"])
            elastic_dic["text"].append(text)




    df = pd.DataFrame(elastic_dic)
    print(df)

    with open("data/elastic_data_descritores.pkl", 'wb') as f:
        pickle.dump(df, f)




def pandas_to_excel():
    file = open("data/elastic_data_descritores.pkl", 'rb')
    df = pickle.load(file)
    df.to_excel("data/elastic_data_descritores.xlsx", index=False)


def divide_elastic_data_in_sections():
    file = open("data/elastic_data_descritores.pkl", 'rb')
    df = pickle.load(file)

    official_descritores = get_official_descritores()

    ss_dict = {"id": [], "section": [], "descritores": [], "text": []}
    first_section = {"id": [], "section": [], "descritores": [], "text": []}
    second_section = {"id": [], "section": [], "descritores": [], "text": []}
    third_section = {"id": [], "section": [], "descritores": [], "text": []}
    fourth_section = {"id": [], "section": [], "descritores": [], "text": []}
    fifth_section = {"id": [], "section": [], "descritores": [], "text": []}
    sixth_section = {"id": [], "section": [], "descritores": [], "text": []}
    seventh_section = {"id": [], "section": [], "descritores": [], "text": []}
    cont_section = {"id": [], "section": [], "descritores": [], "text": []}

    for i, row in enumerate(df.iloc):
        new_descritores = []
        descritores = row["descritores"]
        for d in descritores:
            if d in official_descritores:
                new_descritores.append(d)
        if len(new_descritores) == 0:
            continue
        if row["section"] == "«sem secção registada»":
            ss_dict["id"].append(row["id"])
            ss_dict["section"].append(row["section"])
            ss_dict["descritores"].append(new_descritores)
            ss_dict["text"].append(row["text"])

        elif row["section"] == "1.ª Secção (Cível)":
            first_section["id"].append(row["id"])
            first_section["section"].append(row["section"])
            first_section["descritores"].append(new_descritores)
            first_section["text"].append(row["text"])

        elif row["section"] == "2.ª Secção (Cível)":
            second_section["id"].append(row["id"])
            second_section["section"].append(row["section"])
            second_section["descritores"].append(new_descritores)
            second_section["text"].append(row["text"])

        elif row["section"] == "3.ª Secção (Criminal)":
            third_section["id"].append(row["id"])
            third_section["section"].append(row["section"])
            third_section["descritores"].append(new_descritores)
            third_section["text"].append(row["text"])

        elif row["section"] == "4.ª Secção (Social)":
            fourth_section["id"].append(row["id"])
            fourth_section["section"].append(row["section"])
            fourth_section["descritores"].append(new_descritores)
            fourth_section["text"].append(row["text"])

        elif row["section"] == "5.ª Secção (Criminal)":
            fifth_section["id"].append(row["id"])
            fifth_section["section"].append(row["section"])
            fifth_section["descritores"].append(new_descritores)
            fifth_section["text"].append(row["text"])

        elif row["section"] == "6.ª Secção (Cível)":
            sixth_section["id"].append(row["id"])
            sixth_section["section"].append(row["section"])
            sixth_section["descritores"].append(new_descritores)
            sixth_section["text"].append(row["text"])

        elif row["section"] == "7.ª Secção (Cível)":
            seventh_section["id"].append(row["id"])
            seventh_section["section"].append(row["section"])
            seventh_section["descritores"].append(new_descritores)
            seventh_section["text"].append(row["text"])

        elif row["section"] == "Contencioso":
            cont_section["id"].append(row["id"])
            cont_section["section"].append(row["section"])
            cont_section["descritores"].append(new_descritores)
            cont_section["text"].append(row["text"])

    df_1 = pd.DataFrame(first_section)
    print(df_1)

    with open("data/first_section.pkl", 'wb') as f:
        pickle.dump(df_1, f)

    df_2 = pd.DataFrame(second_section)
    print(df_2)

    with open("data/second_section.pkl", 'wb') as f:
        pickle.dump(df_2, f)

    df_3 = pd.DataFrame(third_section)
    print(df_3)

    with open("data/third_section.pkl", 'wb') as f:
        pickle.dump(df_3, f)

    df_4 = pd.DataFrame(fourth_section)
    print(df_4)

    with open("data/forth_section.pkl", 'wb') as f:
        pickle.dump(df_4, f)

    df_5 = pd.DataFrame(fifth_section)
    print(df_5)

    with open("data/fifth_section.pkl", 'wb') as f:
        pickle.dump(df_5, f)

    df_6 = pd.DataFrame(sixth_section)
    print(df_6)

    with open("data/sixth_section.pkl", 'wb') as f:
        pickle.dump(df_6, f)

    df_7 = pd.DataFrame(seventh_section)
    print(df_7)

    with open("data/seventh_section.pkl", 'wb') as f:
        pickle.dump(df_7, f)

    df_cont = pd.DataFrame(cont_section)
    print(df_cont)

    with open("data/contencioso_section.pkl", 'wb') as f:
        pickle.dump(df_cont, f)



#so é preciso retreinar para as áreas (cível) - 1,2,6,7 secções e para a criminal 3,5 seccções
def divide_elastic_data_in_areas():
    file = open("data_area/elastic_data_descritores.pkl", 'rb')
    df = pickle.load(file)

    print(df)

    official_descritores = get_official_descritores()

    civel = {"id": [], "area": [], "descritores": [], "text": []}
    criminal = {"id": [], "area": [], "descritores": [], "text": []}

    for i, row in enumerate(df.iloc):
        new_descritores = []
        descritores = row["descritores"]
        for d in descritores:
            if d in official_descritores:
                new_descritores.append(d)
        if len(new_descritores) == 0:
            continue

        elif row["section"] == "1.ª Secção (Cível)":
            civel["id"].append(row["id"])
            civel["area"].append("cível")
            civel["descritores"].append(new_descritores)
            civel["text"].append(row["text"])

        elif row["section"] == "2.ª Secção (Cível)":
            civel["id"].append(row["id"])
            civel["area"].append("cível")
            civel["descritores"].append(new_descritores)
            civel["text"].append(row["text"])

        elif row["section"] == "3.ª Secção (Criminal)":
            criminal["id"].append(row["id"])
            criminal["area"].append("criminal")
            criminal["descritores"].append(new_descritores)
            criminal["text"].append(row["text"])

        elif row["section"] == "5.ª Secção (Criminal)":
            criminal["id"].append(row["id"])
            criminal["area"].append("criminal")
            criminal["descritores"].append(new_descritores)
            criminal["text"].append(row["text"])

        elif row["section"] == "6.ª Secção (Cível)":
            civel["id"].append(row["id"])
            civel["area"].append("cível")
            civel["descritores"].append(new_descritores)
            civel["text"].append(row["text"])

        elif row["section"] == "7.ª Secção (Cível)":
            civel["id"].append(row["id"])
            civel["area"].append("cível")
            civel["descritores"].append(new_descritores)
            civel["text"].append(row["text"])


    civel_1 = pd.DataFrame(civel)
    print(civel_1)

    with open("data_area/Civel/civel.pkl", 'wb') as f:
        pickle.dump(civel_1, f)

    criminal_2 = pd.DataFrame(criminal)
    print(criminal_2)

    with open("data_area/Criminal/criminal.pkl", 'wb') as f:
        pickle.dump(criminal_2, f)






def get_descritores_list():
    file = open("data_area/Criminal/criminal.pkl", "rb")
    df = pickle.load(file)
    descritores = []
    official_descritores = get_official_descritores()
    for i, row in enumerate(df.iloc):
        des = row["descritores"]
        for d in des:
            if d not in descritores and d in official_descritores:
                descritores.append(d)

    sorted_list = sorted(descritores)

    print(sorted_list)



def stat_descritores():
    file = open("data/civel/civel_without_few_descritores.pkl", "rb")
    df = pickle.load(file)
    stat_dic = {}

    workbook = xlsxwriter.Workbook("data/criminal/civel_stats_few_descritores.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', "Descritor")
    worksheet.write('B1', "Acordaos")
    worksheet.write("C1", "Quantidade")

    for i, row in enumerate(df.iloc):
        des = row["descritores"]
        ac = row["id"]
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
    file = open("data/criminal/criminal.pkl", "rb")
    df = pickle.load(file)
    print(df)
    dic = {"id": [], "area": [], "descritores": [], "text": []}
    stat_dic = {}
    delete_descritores = []

    for i, row in enumerate(df.iloc):
        des = row["descritores"]
        ac = row["id"]
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
        des = row["descritores"]
        text = row["text"]
        section = row["area"]
        file_name = row["id"]
        for d in des:
            if d not in delete_descritores:
                new_descritores.append(d)
        if len(new_descritores) != 0:
            dic["id"].append(file_name)
            dic["area"].append(section)
            dic["text"].append(text)
            dic["descritores"].append(new_descritores)

    df = pd.DataFrame(dic)
    print(df)
    #with open("data_area/Criminal/criminal_without_few_descritores.pkl", 'wb') as f:
    #    pickle.dump(df, f)


#delete_few_descritores()

def create_panda_all_labels(section):
    file = open("data/" + section + "/civil_without_few_descritores.pkl", "rb")
    df = pickle.load(file)
    dic = {"id": [], "area": [], "text": []}

    labels = return_labels("civel")

    print(labels)
    for d in labels:
        dic[d] = []

    for i, row in enumerate(df.iloc):
        dic["id"].append(row["id"])
        dic["text"].append(row["text"])
        dic["area"].append(row["area"])

        des = row["descritores"]

        for de in labels:
            if de not in des:
                dic[de].append(0)
            else:
                dic[de].append(1)


    df = pd.DataFrame(dic)
    print(df)

    df = df.loc[:, (df != 0).any(axis=0)]

    print(df)

    with open("data/" + section + "/civil_without_few_descritores_all_labels.pkl", 'wb') as f:
        pickle.dump(df, f)


#create_panda_all_labels("civel v2")



def divide_data_statified(section):
    file = open("data/" + section + "/civil_without_few_descritores_all_labels.pkl", "rb")
    df = pickle.load(file)

    y = df[df.columns[3:]].values




    x = df["text"].values.reshape(len(df), 1)
    print(x.shape)
    print(y.shape)


    x_train, y_train, x_test, y_test = iterative_train_test_split(x,y, test_size=0.2)

    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    with open("data/" + section + "/X_train.pkl", 'wb') as f:
        pickle.dump(x_train, f)

    with open("data/" + section + "/X_test.pkl", 'wb') as f:
        pickle.dump(x_test, f)

    with open("data/" + section + "/y_train.pkl", 'wb') as f:
        pickle.dump(y_train, f)

    with open("data/" + section + "/y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)

#divide_data_statified("civel v2")


def create_pandas_stratified_divisions(section_dir):
    file = open("data/" + section_dir + "/civil_without_few_descritores_all_labels.pkl", "rb")
    df = pickle.load(file)
    train = {"id": [], "area": [], "text": [], "descritores": []}
    test = {"id": [], "area": [], "text": [], "descritores": []}

    print(df)

    x_train_file = open("data/" + section_dir + "/X_train.pkl", "rb")
    x_train = pickle.load(x_train_file)


    y_train_file = open("data/" + section_dir + "/y_train.pkl", "rb")
    y_train = pickle.load(y_train_file)


    x_test_file = open("data/" + section_dir + "/X_test.pkl", "rb")
    x_test = pickle.load(x_test_file)


    y_test_file = open("data/" + section_dir + "/y_test.pkl", "rb")
    y_test = pickle.load(y_test_file)

    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    for i, row in enumerate(df.iloc):
        print("i", i)
        found = False
        text = row["text"]
        file_name = row["id"]
        section = row["area"]
        labels = row[3:].tolist()


        for j,t in enumerate(x_train[:]):
            y_labels = y_train[j].tolist()
            if text == t[0] and labels == y_labels:
                train["id"].append(file_name)
                train["text"].append(text)
                train["area"].append(section)
                train["descritores"].append(y_train[j])
                found = True
                break

        for w,tt in enumerate(x_test[:]):
            y_labels_test = y_test[w].tolist()
            if text == tt[0] and labels == y_labels_test:
                if found == False:
                    test["id"].append(file_name)
                    test["text"].append(text)
                    test["area"].append(section)
                    test["descritores"].append(y_test[w])
                    break
                else:
                    print(row["id"])

        assert len(train["id"]) + len(test["id"]) - 1 == i

    train_df = pd.DataFrame(train)
    print(train_df)
    with open("data/" + section_dir + "/data_train.pkl", 'wb') as f:
        pickle.dump(train_df, f)

    test_df = pd.DataFrame(test)
    print(test_df)
    with open("data/" + section_dir + "/data_test.pkl", 'wb') as f:
        pickle.dump(test_df, f)

#create_pandas_stratified_divisions("civel v2")

def check_descritores_in_text(section_dir):
    file = open("data/" + section_dir + "/first_section_without_few_descritores.pkl", "rb")
    df = pickle.load(file)

    print(df)
    l_dic = []

    for i, row in enumerate(df.iloc):
        print("i", i)
        text = row["text"]
        d = {"id": row["id"]}
        doc = Judgment(text, "html", False)
        descritores  = row["descritores"]
        for des_real in descritores:
            d[des_real] = 0
            des = des_real.lower()
            des = re.sub(r'ç', 'c', des)
            des = re.sub(r'ã', 'a', des)
            des = re.sub(r'á', 'a', des)
            des = re.sub(r'õ', 'o', des)
            des = re.sub(r'ó', 'o', des)
            des = re.sub(r'í', 'i', des)
            des = re.sub(r'é', 'e', des)
            des = re.sub(r'â', 'a', des)
            des = re.sub(r'à', 'a', des)
            des = re.sub(r'[^a-zA-Z0-9\s]+', '', des)
            print(des)
            for p in doc.paragraphs:
                paragraph = p.text.get_text().lower()
                paragraph = re.sub(r'ç', 'c', paragraph)
                paragraph = re.sub(r'ã', 'a', paragraph)
                paragraph = re.sub(r'á', 'a', paragraph)
                paragraph = re.sub(r'õ', 'o', paragraph)
                paragraph = re.sub(r'ó', 'o', paragraph)
                paragraph = re.sub(r'í', 'i', paragraph)
                paragraph = re.sub(r'é', 'e', paragraph)
                paragraph = re.sub(r'â', 'a', paragraph)
                paragraph = re.sub(r'à', 'a', paragraph)
                paragraph = re.sub(r'[^a-zA-Z0-9\s]+', '', paragraph)
                if len(des.split(' ')) == 1:
                    paragraph_grams = paragraph.split(' ')
                    for p_g in paragraph_grams:
                        if des == p_g:
                            d[des_real] += 1


                else:
                    n_grams = len(des.split(' '))
                    paragraph_grams = []
                    paragraph = ngrams(paragraph.split(), n_grams)
                    for p in paragraph:
                        s = ''
                        for w in p:
                            s += w + ' '
                        paragraph_grams.append(s)



                for p_g in paragraph_grams:
                    if des == p_g:
                        d[des_real] += 1



        l_dic.append(d)



    workbook = xlsxwriter.Workbook("data/" + section_dir + "/descritores_in_judgment.xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    colum = 0

    for l_d in l_dic:
        print(l_d)
        for key, value in l_d.items():
            worksheet.write(row, colum, key)
            row += 1
            worksheet.write(row, colum, value)
            row -= 1
            colum += 1

        row += 3
        colum = 0

    workbook.close()


def check_descritores_in_judgment(section_dir):
    file = open("data/" + section_dir + "/first_section_without_few_descritores.pkl", "rb")
    df = pickle.load(file)

    print(df)
    dic_descritores = {}
    found = False

    for i, row in enumerate(df.iloc):
        print("i", i)
        text = row["text"]
        doc = Judgment(text, "html", False)
        descritores = row["descritores"]
        for des_real in descritores:
            des = des_real.lower()
            des = re.sub(r'ç', 'c', des)
            des = re.sub(r'ã', 'a', des)
            des = re.sub(r'á', 'a', des)
            des = re.sub(r'õ', 'o', des)
            des = re.sub(r'ó', 'o', des)
            des = re.sub(r'í', 'i', des)
            des = re.sub(r'é', 'e', des)
            des = re.sub(r'â', 'a', des)
            des = re.sub(r'à', 'a', des)
            des = re.sub(r'[^a-zA-Z0-9\s]+', '', des)
            print(des)

            if des_real + "_total" not in dic_descritores:
                dic_descritores[des_real] = 0
                dic_descritores[des_real + "_total"] = 1
            else:
                dic_descritores[des_real + "_total"] += 1

            for p in doc.paragraphs:
                paragraph = p.text.get_text().lower()
                paragraph = re.sub(r'ç', 'c', paragraph)
                paragraph = re.sub(r'ã', 'a', paragraph)
                paragraph = re.sub(r'á', 'a', paragraph)
                paragraph = re.sub(r'õ', 'o', paragraph)
                paragraph = re.sub(r'ó', 'o', paragraph)
                paragraph = re.sub(r'í', 'i', paragraph)
                paragraph = re.sub(r'é', 'e', paragraph)
                paragraph = re.sub(r'â', 'a', paragraph)
                paragraph = re.sub(r'à', 'a', paragraph)
                paragraph = re.sub(r'[^a-zA-Z0-9\s]+', '', paragraph)
                if len(des.split(' ')) == 1:
                    paragraph_grams = paragraph.split(' ')
                    for p_g in paragraph_grams:
                        if des == p_g:
                            if des_real not in dic_descritores:
                                dic_descritores[des_real] = 1
                            else:
                                dic_descritores[des_real] += 1
                            found = True
                            break
                    if found:
                        found = False
                        break


                else:
                    n_grams = len(des.split(' '))
                    paragraph_grams = []
                    paragraph = ngrams(paragraph.split(), n_grams)
                    for p in paragraph:
                        s = ''
                        for w in p:
                            s += w + ' '
                        paragraph_grams.append(s)

                for p_g in paragraph_grams:
                    if des == p_g:
                        if des_real not in dic_descritores:
                            dic_descritores[des_real] = 1
                        else:
                            dic_descritores[des_real] += 1
                        break




    workbook = xlsxwriter.Workbook("data/" + section_dir + "/descritores_in_judgment.xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    colum = 0


    for key, value in dic_descritores.items():
        worksheet.write(row, colum, key)
        row += 1
        worksheet.write(row, colum, value)
        row -= 1
        colum += 1



    workbook.close()


