News Headline Sentiment Classification using Fine-Tuned RoBERTa

This project develops a custom sentiment analysis model trained specifically on news headlines. By fine-tuning a RoBERTa transformer, we classify headline sentiment into positive, neutral, and negative categories—achieving significantly better performance than traditional rule-based tools like VADER.

Key Features
Fine-tunes FacebookAI/roberta-base using Hugging Face Transformers on labeled news headline data.
Predicts sentiment with:
- Sentiment label (positive / neutral / negative)
- Class probabilities (softmax output)
- Custom compound sentiment score derived from model logits
- Runtime comparison (RoBERTa vs VADER)
Highlights how a deep learning model built for headlines outperforms VADER on accuracy, nuance handling, and reliability.

Model Overview
RoBERTa Model
- Fine-tuned on a labeled dataset of news headlines, enabling it to understand context, tone, and subtle sentiment cues missed by traditional analyzers.
VADER
- A lexicon-based rule engine designed for general sentiment detection. While fast, it struggles with headline nuance and domain-specific context.
Our model shows consistent improvements in:
- Prediction accuracy
- Confidence scores
- Context-sensitive sentiment interpretation

Dependencies
- torch
- transformers
- pandas
- scikit-learn
- vaderSentiment

