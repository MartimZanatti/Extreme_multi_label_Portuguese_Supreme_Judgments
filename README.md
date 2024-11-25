# Extreme_multi_label_Portuguese_Supreme_Judgments

### Introduction:
Welcome to the Extreme Multi-Label System for Portuguese Supreme Court Judgments! This system is designed to return legal index terms (labels) given a judgment. 

### Key Features:
1. The system comprises separate models for each of the nine sections of Supreme Court judgments, namely "Contencioso", "Sem secção", "1.ª Secção (Cível)", "2.ª Secção (Cível)", "3.ª Secção (Criminal)", "4.ª Secção (Social)", "5.ª Secção (Criminal)", "6.ª Secção (Cível)", and "7.ª Secção (Cível)". Each model is trained to identify and assign relevant labels (descriptors) specific to its respective section.
2. Paragraph labelling: We have trained a model (Bilstm with CRF) that gives for each paragraph of the judgment a label (zone of the judgment). We considered nine zones: "cabeçalho", "relatório", "delimitação", "fundamentação", "decisão", "colectivo", "declaração", "foot-notes" and "títulos. 

### Usage:
To use this Extreme Multi-Label System for Portuguese Supreme Court Judgments, follow these steps:

1. Clone the repository to your local machine.
2. In the usage.py file, use the usage function that receives a judgments and to which section it belongs. The judgment is divided into zones, where only the "relatório", "fundamentação" and "decisão" are considered to the Extreme-multi-label model. The y_train is loaded every time the model is used as well. This function returns the chosen labels for the respective document. The descriptors and documents used are from the Portuguese Supreme Court of Justice.
3. It is possible to train new models with new data. The train function is in the run.py and the run.sh has the script. The dataset used is available in here (https://huggingface.co/datasets/MartimZanatti/Descriptors_STJ).

Models at: https://gitlab.com/diogoalmiro/iris-lfs-storage/

Running with python venv:

(install once:)
  - python -m venv env
  - source env/bin/activate
  - pip install -U pip
  - pip install -r requirements.txt

(running:)
  - python server.py

Running with docker:
  - (build once:)
    - docker build .
  - (running:)
    - docker run -it -p 8999:8999


  - After running server.py, go to http://127.0.0.1:8999 and insert a judgment (docx, txt or html) and the legal division it belongs to.
  - The judgment is divided into section (see this paper https://link.springer.com/chapter/10.1007/978-3-031-73497-7_20).
  - The sections report, facts, law and decisions are used as the all judgment and then the descriptors are returned 
  - See descriptors_assignment function in the usage.py file for more details



### Contributing:
We welcome contributions to enhance the functionality and performance of this system. If you have any ideas, bug fixes, or improvements, please submit a pull request, and we will review it promptly.

### Contact:
For any inquiries or feedback regarding this Extreme Multi-Label System, please contact martim.zanatti@tecnico.ulisboa.pt

We hope you find this system useful in efficiently navigating and exploring the extensive collection of Portuguese Supreme Court judgments. Happy labeling and categorizing!
