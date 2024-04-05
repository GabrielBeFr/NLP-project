Involved students:
- BEN ZENOU Gabriel
- IAGARU David
- MACE Quentin
- ROVERATO Chiara

Details on the classifier:
    The classifier used is based on a Roberta-base model trained for sentiment analysis on a big corpus of tweets available <a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest">here</> \\
    We dropped the original classification head from the model and use the backbone as a text embedding model (using the embedding of the \<CLS\> token).
    At inference, we use this backbone to infer embeddings for both the sentence and the given word. We then apply a simple dense head to the concatenation of those embedding in order to yield the sentiment.

    Here is a picture of the used architecture : \\
    ![Architecture Diagram](architecture.png)

    We trained this model for 5 epochs with a very low learning rate (5e-6) to avoid catastrophic forgetting and using the AdamW optimizer from pytorch. \\
    We also used CrossEntropyLoss for training.
Reached accuracy:
    86.38 on the dev set (averaged over 5 runs)