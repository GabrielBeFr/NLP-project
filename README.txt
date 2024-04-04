Involved students:
- BEN ZENOU Gabriel
- IAGARU David
- MACE Quentin
- ROVERATO Chiara

Details on the classifier:
    The classifier used is Yang Heng's ABSA classifier "deberta-v3-base-absa-v1.1". 
    It's a finetuned BERT for ABSA classification. With no finetuning on our data,
    we get 86% of accuracy on the devdata.csv which is already a good score but we 
    succeed in improving it to 91% with a finetuning on the traindata.csv.

Reached accuracy:
    0.91