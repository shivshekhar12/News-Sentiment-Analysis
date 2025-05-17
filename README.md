This project compares the sentiment classification of financial/news headlines using two different approaches:
1. RoBERTa-based Transformer model, fine-tuned on labeled headline data.
2. VADER Sentiment Analyzer, a rule-based NLP tool.
It aims to evaluate how a modern deep learning model performs against a traditional sentiment analyzer, providing insights into accuracy, confidence, and runtime.

Features
Fine-tunes FacebookAI/roberta-base using HuggingFace Transformers
Predicts sentiment across three categories: positive, neutral, and negative
- Outputs include:
- Predicted label
- Class probabilities
- Compound sentiment score (RoBERTa)
- Runtime comparison (RoBERTa vs VADER)

Project Structure
news-sentiment-analyzer/
├── data/ # Training/validation CSVs (not included)
├── results/ # Saved model checkpoints (after training)
├── train_model.py # Fine-tunes RoBERTa on labeled data
├── predict_sentiment.py # Loads model and compares VADER vs RoBERTa
├── utils.py # Utility functions for loading and predicting
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Getting Started
1. Clone this repo:
git clone https://github.com/your-username/news-sentiment-analyzer.git
cd news-sentiment-analyzer
2. Install dependencies:
pip install -r requirements.txt
3. Train the model (optional):
python train_model.py
4. Run sentiment comparisons:
python predict_sentiment.py

Model Details
= RoBERTa is fine-tuned on labeled news headlines across 3 sentiment classes.
- VADER uses lexicon-based scoring to assign a sentiment label.
- Results are compared for multiple real-world headlines.

Dependencies
- torch
- transformers
- pandas
- scikit-learn
- vaderSentiment

Notes
- Datasets are not included. Add your own under the data/ folder.
- Checkpoints are saved in ./results and loaded during evaluation.
