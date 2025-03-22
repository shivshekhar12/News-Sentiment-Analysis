import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
def load_model(model_path="./results"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Predict sentiment from text
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        compound_score = (probs[0][2] - probs[0][0]).item()
    return {
        "label_id": pred_class,
        "compound_score": round(compound_score, 4),
        "probs": probs[0].tolist()
    }
