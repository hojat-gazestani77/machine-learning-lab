from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

## Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
text = "I love you"
print("\n1. Original text")
print(text)

### Tokenization
tokens = tokenizer.tokenize(text)
print("\n2. Tokens")
print(tokens)

### Token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("\n3. Token IDs")
print(token_ids)

### Encode for model
inputs = tokenizer(text, return_tensors="pt")  # ?
print("\n4. Model Input")
print(inputs)

### Embedding vectors
with torch.no_grad():
    embeddings = model.distilbert.embeddings.word_embeddings(inputs["input_ids"])
print("\n5. Embedding shape")
print(embeddings.shape)

print("\nExample embedding vector for first token:")
print(embeddings[0][0][:10])  # first 10 numbers

## Full transformer forward pass
with torch.no_grad():
    outputs = model(**inputs)
print("\n6. RAW model output")
print(outputs)

logits = outputs.logits
print("\n7. logits")
print(logits)

### Probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print("\n8. Probabilities")
print(probabilities)

### Prediction
predicted_class = torch.argmax(probabilities).item()
labels = model.config.id2label
print("\n9. Final prediction")
print(labels[predicted_class])
