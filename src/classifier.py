from typing import List
from utils import *

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW




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
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    
    
    
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
        texts = [f'[CLS] {row[4]}  [SEP] {row[2]} [SEP]'    for row in data_raw]
        labels = [corresp_inv[row[0]] for row in data_raw]
        texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.3, random_state=42)

        
        # Tokenize input texts
        inputs_train = self.tokenizer(texts_train, padding=True, truncation=True, return_tensors='pt')
        inputs_val = self.tokenizer(texts_val, padding=True, truncation=True, return_tensors='pt')
        # Convert labels to tensors
        labels_train = torch.tensor(labels_train)
        labels_val = torch.tensor(labels_val)


        # Create PyTorch datasets and data loaders
        train_dataset = TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], torch.tensor(labels_train))
        val_dataset = TensorDataset(inputs_val['input_ids'], inputs_val['attention_mask'], torch.tensor(labels_val))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Define optimizer and training settings
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.to(device)
        # Training loop
        num_epochs = 15
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            total_correct = 0
            total_examples = 0
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
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
        inputs_test = [f'[CLS] {row[4]}  [SEP] {row[2]} [SEP]'   for row in test_data]
        labels_test = [row[0]   for row in test_data]

        corresp = {1: 'neutral', 2: 'positive', 0:'negative'}

        outputs_all = []
        for x in inputs_test:
            # Input text
            input_text = x

            # Tokenize input text
            inputsi = self.tokenizer(input_text, return_tensors="pt")

            # Move input tensors to appropriate device
            inputsi = {key: value.to(device) for key, value in inputsi.items()}

            # Perform inference
            outputs = self.model(**inputsi)

            # Get predicted label
            predicted_class = outputs.logits.argmax().item()

            # Print predicted class
            outputs_all.append(corresp[predicted_class])
        return outputs_all





