from typing import List
import pandas as pd
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification, AdamW


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        base_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.embedder = base_model.roberta
        self.classifier = nn.Sequential(nn.Linear(768*2, 768), nn.GELU(), nn.Linear(768, 3))
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # for param in self.embedder.parameters():
        #     param.requires_grad = False
    
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        corresp_inv = {'neutral':1, 'positive':2, 'negative': 0}
        data_raw = load_file(train_filename)
        sentences = [row[4] for row in data_raw]
        words = [row[2] for row in data_raw]
        texts = [f'[CLS] {row[4]}  [SEP] {row[2]} [SEP]'    for row in data_raw]
        labels = [corresp_inv[row[0]] for row in data_raw]

        texts_train, texts_val, words_train, words_test, labels_train, labels_val = train_test_split(sentences, words, labels, test_size=0.3, random_state=42)

        
        # Tokenize input texts
        inputs_train = self.tokenizer(texts_train, padding=True, truncation=True, return_tensors='pt')
        inputs_val = self.tokenizer(texts_val, padding=True, truncation=True, return_tensors='pt')
        # Tokenize input words
        inputs_train_words = self.tokenizer(words_train, padding=True, truncation=True, return_tensors='pt')
        inputs_val_words = self.tokenizer(words_test, padding=True, truncation=True, return_tensors='pt')
        # Convert labels to tensors
        labels_train = torch.tensor(labels_train)
        labels_val = torch.tensor(labels_val)


        # Create PyTorch datasets and data loaders
        train_dataset = TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], inputs_train_words["input_ids"], inputs_train_words["attention_mask"], torch.tensor(labels_train))
        val_dataset = TensorDataset(inputs_val['input_ids'], inputs_val['attention_mask'], inputs_val_words["input_ids"], inputs_val_words["attention_mask"], torch.tensor(labels_val))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Define optimizer and training settings
        params = list(self.embedder.parameters()) + list(self.classifier.parameters())
        optimizer = AdamW(params, lr=5e-6) # low lr to avoid catastrophic forgetting
        self.embedder.to(device)
        self.classifier.to(device)
        # Training loop
        num_epochs = 5

        loss_history = []
        for epoch in range(num_epochs):
            self.embedder.train()
            self.classifier.train()
            for batch in train_loader:
                input_ids, attention_mask, input_word_ids, attention_word_mask, labels = batch
                input_ids, attention_mask, input_word_ids, attention_word_mask, labels = input_ids.to(device), attention_mask.to(device), input_word_ids.to(device), attention_word_mask.to(device), labels.to(device)

                optimizer.zero_grad()
                # forward pass
                embeddings = self.embedder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
                embeddings_words = self.embedder(input_ids=input_word_ids, attention_mask=attention_word_mask)["last_hidden_state"][:, 0, :]

                concat_embeddings = torch.cat([embeddings, embeddings_words], dim=1)
                outputs = self.classifier(concat_embeddings)

                # grad computation
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                loss_history.append(loss.item())
                optimizer.step()

            # Validation
            self.embedder.eval()
            self.classifier.eval()
            total_correct = 0
            total_examples = 0
            for batch in val_loader:
                input_ids, attention_mask, input_word_ids, attention_word_mask, labels = batch
                input_ids, attention_mask, input_word_ids, attention_word_mask, labels = input_ids.to(device), attention_mask.to(device), input_word_ids.to(device), attention_word_mask.to(device), labels.to(device)

                with torch.no_grad():
                    embeddings = self.embedder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
                    embeddings_words = self.embedder(input_ids=input_word_ids, attention_mask=attention_word_mask)["last_hidden_state"][:, 0, :]

                    concat_embeddings = torch.cat([embeddings, embeddings_words], dim=1)
                    outputs = self.classifier(concat_embeddings)
                    logits = self.softmax(outputs)
                    preds = torch.argmax(logits, dim=1)

                total_correct += (preds == labels).sum().item()
                total_examples += len(labels)

            accuracy = total_correct / total_examples
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        test_data = load_file(data_filename)
        sentences = [row[4] for row in test_data]
        words = [row[2] for row in test_data]
        inputs_test = [f'[CLS] {row[4]}  [SEP] {row[2]} [SEP]'   for row in test_data]
        labels_test = [row[0]   for row in test_data]

        corresp = {1: 'neutral', 2: 'positive', 0:'negative'}

        outputs_all = []
        self.embedder.to(device)
        self.classifier.to(device)
        for s,w in zip(sentences, words):
            # Input text
            input_text = s
            input_word = w

            # Tokenize input text
            input_s = self.tokenizer(input_text, return_tensors="pt")
            input_words = self.tokenizer(input_word, return_tensors="pt")

            # Move input tensors to appropriate device
            input_s = {key: value.to(device) for key, value in input_s.items()}
            input_words = {key: value.to(device) for key, value in input_words.items()}

            # Perform inference
            with torch.no_grad():
                embeddings = self.embedder(input_ids=input_s['input_ids'], attention_mask=input_s["attention_mask"])["last_hidden_state"][:, 0, :]
                embeddings_words = self.embedder(input_ids=input_words['input_ids'], attention_mask=input_words["attention_mask"])["last_hidden_state"][:, 0, :]

                concat_embeddings = torch.cat([embeddings, embeddings_words], dim=1)
                outputs = self.classifier(concat_embeddings)
                logits = self.softmax(outputs)

            # Get predicted label
            predicted_class = torch.argmax(logits, dim=1).item()

            # Print predicted class
            outputs_all.append(corresp[predicted_class])
        return outputs_all





