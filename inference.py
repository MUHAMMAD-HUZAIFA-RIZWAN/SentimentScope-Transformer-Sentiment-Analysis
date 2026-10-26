import torch
from transformers import AutoTokenizer

from model import DemoGPT


CHECKPOINT_PATH = "sentimentscope_model.pth"
TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    """
    Load the trained SentimentScope model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        device (torch.device, optional): Device to load the model on.

    Returns:
        tuple: Loaded model, tokenizer, and device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    config = checkpoint["config"]

    model = DemoGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return model, tokenizer, device


def predict_sentiment(texts, model, tokenizer, device):
    """
    Predict sentiment for a batch of text inputs.

    Args:
        texts (list[str]): List of movie reviews.
        model (torch.nn.Module): Loaded SentimentScope model.
        tokenizer: Hugging Face tokenizer.
        device (torch.device): Device used for inference.

    Returns:
        list[dict]: Predicted sentiment and confidence for each input.
    """
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    results = []

    for text, prediction, probability in zip(
        texts,
        predictions,
        probabilities
    ):
        label = "Positive" if prediction.item() == 1 else "Negative"
        confidence = probability[prediction].item()

        results.append({
            "text": text,
            "sentiment": label,
            "confidence": confidence
        })

    return results


if __name__ == "__main__":
    model, tokenizer, device = load_model()

    sample_reviews = [
        "This movie was fantastic. I really enjoyed the story and performances.",
        "The film was boring, predictable, and a complete waste of time."
    ]

    results = predict_sentiment(
        sample_reviews,
        model,
        tokenizer,
        device
    )

    print(f"Using device: {device}")

    for result in results:
        print()
        print(f"Review: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")