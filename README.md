# SentimentScope: Transformer-Based Sentiment Analysis

SentimentScope is a transformer-based binary sentiment classification system built with PyTorch. The model analyzes IMDB movie reviews and predicts whether the expressed sentiment is **positive** or **negative**.

Unlike applications that simply fine-tune an existing pretrained classifier, this project implements the core transformer architecture directly, including token embeddings, positional embeddings, multi-head self-attention, feed-forward networks, residual connections, layer normalization, pooling, and a classification head.

The final model achieved **76.78% test accuracy** on the IMDB movie review dataset.

---

## Results

| Metric | Result |
|---|---:|
| Epoch 1 Validation Accuracy | 69.04% |
| Epoch 2 Validation Accuracy | 72.96% |
| Epoch 3 Validation Accuracy | **77.48%** |
| Final Test Accuracy | **76.78%** |

The project requirement was to achieve more than 75% test accuracy, which the trained model successfully exceeded.

---

## Features

- Custom transformer architecture implemented in PyTorch
- Multi-head self-attention
- Token and positional embeddings
- Feed-forward transformer layers
- Residual connections and layer normalization
- Mean pooling for text classification
- Binary sentiment classification
- Custom PyTorch `Dataset` and `DataLoader`
- Training and validation pipeline
- Model checkpoint saving and loading
- CPU-compatible inference interface
- Hugging Face BERT tokenizer
- IMDB movie review dataset

---

## Model Architecture

The model uses a lightweight transformer configuration:

| Parameter | Value |
|---|---:|
| Vocabulary | BERT Base Uncased vocabulary |
| Maximum Sequence Length | 128 tokens |
| Embedding Dimension | 128 |
| Transformer Blocks | 4 |
| Attention Heads | 4 |
| Head Dimension | 32 |
| Dropout | 0.1 |
| Output Classes | 2 |

### Architecture Flow

```text
Movie Review
     ↓
BERT Tokenizer
     ↓
Token Embeddings
     +
Positional Embeddings
     ↓
4 × Transformer Blocks
     ↓
Layer Normalization
     ↓
Mean Pooling
     ↓
Linear Classification Head
     ↓
Positive / Negative
```

Each transformer block contains:

```text
LayerNorm
   ↓
Multi-Head Self-Attention
   ↓
Residual Connection
   ↓
LayerNorm
   ↓
Feed-Forward Network
   ↓
Residual Connection
```

---

## Repository Structure

```text
SentimentScope-Transformer-Sentiment-Analysis/
│
├── SentimentScope.ipynb
├── sentimentscope_model.pth
├── model.py
├── inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Files

**`SentimentScope.ipynb`**  
Contains the complete project workflow including data loading, exploratory analysis, dataset preparation, transformer architecture, training, validation, and testing.

**`model.py`**  
Contains the reusable PyTorch implementation of the transformer architecture.

**`inference.py`**  
Loads the trained model checkpoint and performs sentiment prediction on new movie reviews.

**`sentimentscope_model.pth`**  
Saved checkpoint of the trained transformer model.

**`requirements.txt`**  
Lists the Python dependencies required to run the project.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MUHAMMAD-HUZAIFA-RIZWAN/SentimentScope-Transformer-Sentiment-Analysis.git
cd SentimentScope-Transformer-Sentiment-Analysis
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Inference

The trained checkpoint can be used directly without retraining the model.

Run:

```bash
python inference.py
```

Example output:

```text
Using device: cpu

Review: This movie was fantastic. I really enjoyed the story and performances.
Sentiment: Positive
Confidence: 96.56%

Review: The film was boring, predictable, and a complete waste of time.
Sentiment: Negative
Confidence: 99.90%
```

The inference script automatically uses CUDA when available and otherwise runs on CPU.

---

## Using the Model in Python

```python
from inference import load_model, predict_sentiment

model, tokenizer, device = load_model()

reviews = [
    "An excellent movie with brilliant performances.",
    "The story was boring and the acting was disappointing."
]

results = predict_sentiment(
    reviews,
    model,
    tokenizer,
    device
)

for result in results:
    print(result)
```

Example result:

```python
{
    "text": "An excellent movie with brilliant performances.",
    "sentiment": "Positive",
    "confidence": 0.95
}
```

---

## Dataset

The project uses the **Large Movie Review Dataset (IMDB)** for binary sentiment classification.

The dataset contains:

- 25,000 training reviews
- 25,000 test reviews
- Positive and negative sentiment labels

The training dataset was further divided into:

- 22,500 training samples
- 2,500 validation samples

The dataset itself is not included in this repository.

---

## Training

The model was trained using:

- **Optimizer:** AdamW
- **Learning Rate:** `3e-4`
- **Loss Function:** Cross-Entropy Loss
- **Batch Size:** 32
- **Epochs:** 3
- **Maximum Sequence Length:** 128

Training progression:

```text
Epoch 1 → Validation Accuracy: 69.04%
Epoch 2 → Validation Accuracy: 72.96%
Epoch 3 → Validation Accuracy: 77.48%
```

Final evaluation:

```text
Test Accuracy: 76.78%
```

Training was performed in a CUDA-enabled GPU environment. The provided checkpoint can be used for inference on a CPU-only machine without retraining.

---

## Local CPU Inference

The model has also been tested locally on a CPU-only Windows machine.

Example predictions:

```text
Positive review → Positive (96.56% confidence)
Negative review → Negative (99.90% confidence)
```

This demonstrates that GPU hardware is not required for normal inference using the provided checkpoint.

---

## Key Learnings

This project provided practical experience with:

- Transformer architecture implementation
- Self-attention mechanisms
- PyTorch neural network development
- Custom datasets and data loaders
- NLP tokenization
- Transformer training and optimization
- Validation and test-set evaluation
- Model checkpoint management
- Separating training and inference workflows
- Deployable model inference on CPU hardware

---

## Potential Improvements

Future improvements could include:

- Increasing embedding dimensions and transformer depth
- Hyperparameter optimization
- Learning-rate scheduling
- Early stopping
- Attention-mask support for padded tokens
- Precision, recall, F1-score, ROC-AUC, and confusion-matrix evaluation
- Comparison with pretrained transformer models such as BERT
- Interactive Streamlit or Gradio interface
- REST API deployment using FastAPI
- Cloud deployment
- Improved accuracy through pretrained model fine-tuning

---

## Project Background

This project was developed as part of the **Udacity Future AWS AI Programmer Nanodegree Program** and successfully passed the Udacity project review.

The portfolio version additionally includes a reusable model module and standalone inference interface for loading the trained checkpoint and making predictions on new reviews.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook
- Git / GitHub

---

## Author

**Muhammad Huzaifa Rizwan**

AI / Generative AI Developer

---

## License

This repository is intended for educational, research, and portfolio purposes.