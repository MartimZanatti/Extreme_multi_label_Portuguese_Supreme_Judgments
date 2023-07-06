# Extreme_multi_label_Portuguese_Supreme_Judgments

### Introduction:
Welcome to the Extreme Multi-Label System for Portuguese Supreme Court Judgments! This system is designed to return legal index terms (labels) given a judgment. 

### Key Features:
1. The system comprises separate models for each of the nine sections of Supreme Court judgments, namely "Contencioso", "Sem secção", "1.ª Secção (Cível)", "2.ª Secção (Cível)", "3.ª Secção (Criminal)", "4.ª Secção (Social)", "5.ª Secção (Criminal)", "6.ª Secção (Cível)", and "7.ª Secção (Cível)". Each model is trained to identify and assign relevant labels (descriptors) specific to its respective section.
2. Paragraph labelling: We have trained a model (Bilstm with CRF) that gives for each paragraph of the judgment a label (zone of the judgment). We considered nine zones: "cabeçalho", "relatório", "delimitação", "fundamentação", "decisão", "colectivo", "declaração", "foot-notes" and "títulos. 

### Usage:
To use this Extreme Multi-Label System for Portuguese Supreme Court Judgments, follow these steps:

1. Clone the repository to your local machine.
2. In the usage.py file, use the usage function that receives a judgments and to which section it belongs. The judgment is divided into zones, where only the "relatório", "fundamentação" and "decisão" are considered to the Extreme-multi-label model. The y_train is loaded every the model is used as well. 

### Contributing:
We welcome contributions to enhance the functionality and performance of this system. If you have any ideas, bug fixes, or improvements, please submit a pull request, and we will review it promptly.

### Contact:
For any inquiries or feedback regarding this Extreme Multi-Label System, please contact martim_zanatti_5@hotmail.com.

We hope you find this system useful in efficiently navigating and exploring the extensive collection of Portuguese Supreme Court judgments. Happy labeling and categorizing!
