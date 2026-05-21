from datasets import load_dataset

imdb = load_dataset("imdb")

# Smaller dataset to enable faster training
small_train_dataset = (
    imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
)
small_test_dataset = (
    imdb["test"].shuffle(seed=42).select([i for i in list(range(3000))])
)

# Use DistilBERT tokenizer to preprocess the data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


## Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# To speed up training, use data_collector to convert the training smaple
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3 Training the model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

## define the metrics will be use to evaluate
import numpy as np

# from datasets import load_metric
import evaluate


load_accuracy = evaluate.load("accuracy")
load_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, exis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, refrences=labels)[
        "accuracy"
    ]

    accuracy = f1_metrics.compute(predictions=predictions, refrences=labels)
    ["f1"]

    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


## Login to Hugging face to manage the you model repository
# from huggingface_hub import notebook_login
# notebook_login()

## Define the training arguments

from transformers import TrainingArguments, Trainer

repo_name = "finetuning-sentiment-model-3000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    # tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

