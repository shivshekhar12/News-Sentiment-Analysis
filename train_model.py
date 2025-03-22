import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

df = pd.read_csv("data/all-data.csv", encoding="ISO-8859-1", header=None)
df.columns = ["label", "text"]  # Rename columns for clarity

# Load datasets
train_df = pd.read_csv('data/news_train_from_all.csv')
val_df = pd.read_csv('data/news_val_from_all.csv')

# Encode labels
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
val_df['label_encoded'] = label_encoder.transform(val_df['label'])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=3)


def tokenize_data(df):
    return tokenizer(
        df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

train_encodings = tokenize_data(train_df)
val_encodings = tokenize_data(val_df)

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_df['label_encoded'].tolist())
val_dataset = NewsDataset(val_encodings, val_df['label_encoded'].tolist())

# Load Roberta model
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6, 
    learning_rate=2e-5, 
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    no_cuda=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Function to convert logits to compound sentiment score
def get_compound_score(logits):
    probs = F.softmax(torch.tensor(logits), dim=-1)
    score = probs[2] - probs[0]  # positive - negative
    return score.item()  # Between -1 and 1