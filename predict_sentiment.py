from utils import load_model, predict_sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time  

# Load RoBERTa model
tokenizer, model = load_model("./results/checkpoint-2910")

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

import time  # Add this at the top with your other imports

def compare_sentiment(text):
    # ----- RoBERTa Prediction -----
    start_roberta = time.time()
    roberta_result = predict_sentiment(text, tokenizer, model)
    roberta_time = time.time() - start_roberta

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    roberta_label = label_map[roberta_result['label_id']]

    # ----- VADER Prediction -----
    start_vader = time.time()
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_time = time.time() - start_vader

    vader_compound = vader_scores['compound']
    vader_label = "positive" if vader_compound > 0.05 else "negative" if vader_compound < -0.05 else "neutral"

    # ----- Print Results -----
    print(f"Headline: {text}")
    print(f"RoBERTa Sentiment: {roberta_label} (Confidence: {max(roberta_result['probs']):.2f})")
    print(f"RoBERTa Probabilities: {roberta_result['probs']}")
    print(f"RoBERTa Compound Score: {roberta_result['compound_score']:.4f}")
    print(f"RoBERTa Runtime: {roberta_time:.4f} seconds")

    print(f"VADER Sentiment: {vader_label}")
    print(f"VADER Compound Score: {vader_compound:.4f}")
    print(f"VADER Runtime: {vader_time:.6f} seconds")
    print("-" * 60)


# Test with sample headlines
headlines = [
    "Apple stock surges after strong earnings report",
    "Economic downturn leads to mass layoffs in tech",
    "Federal Reserve raises interest rates, markets react negatively",
    "Climate change policies receive bipartisan support",
    "Unemployment rate drops to historic low, experts optimistic",
]

for headline in headlines:
    compare_sentiment(headline)
