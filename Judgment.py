import pypandoc
from bs4 import BeautifulSoup
import re


class Judgment:

    def __init__(self, doc_name, type):
        if type == "docx":
            self.html = self.init_docx_format(doc_name)
        elif type == "html":
            self.html = doc_name
        self.paragraphs = []
        self.add_paragraphs()


    def init_docx_format(self, doc_name):
        html = pypandoc.convert_file(doc_name, "html", extra_args=["--wrap", "none"])
        return html


    def add_paragraphs(self):
        paragraph_id = 1 # id do paragrafo
        soup = BeautifulSoup(self.html, features="lxml")
        paragraph_division = soup.find_all("p") #divide o doc html por paragrafos
        for paragraph in paragraph_division:
            self.paragraphs.append(Paragraph(text=paragraph, id=paragraph_id))
            paragraph_id += 1

    def get_list_text(self):
        all_text = []
        ids = []
        text_ids = []
        i_char = 0
        for p in self.paragraphs:
            num_char = len(list(p.text.get_text()))
            end_char = i_char + num_char
            if not p.symbols:
                all_text.append(p.text.get_text())
                ids.append((p.id, i_char + 1, end_char))


            text_ids.append((p.text.get_text(), p.id, i_char + 1, end_char))
            i_char = end_char
        return all_text, ids, text_ids



class Paragraph:
    def __init__(self, text, id):
        self.text = text
        self.id = id
        if is_only_symbols(text):
            self.symbols = True
        else:
            self.symbols = False

    def __getattribute__(self, item):
        return super(Paragraph, self).__getattribute__(item)



def is_only_symbols(text):
    paragraph_text = text.get_text()
    if re.match(r'^[^a-zA-Z0-9]+$', paragraph_text): # se um paragrafo so contiver simbolos
        return True