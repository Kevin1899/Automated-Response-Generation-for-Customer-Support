from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Load dataset
dataset = load_dataset("Kaludi/Customer-Support-Responses")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s\']', '', text)  # Keep apostrophes and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing to the dataset
dataset = dataset.map(lambda x: {'query': preprocess_text(x['query']), 'response': preprocess_text(x['response'])})

# Convert dataset to DataFrame
df = pd.DataFrame(dataset['train'])

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert DataFrames back to Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Model and tokenizer
model_name = "t5-base"  # Change to a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Check if GPU is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['query'], text_target=examples['response'], padding='max_length', truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    predict_with_generate=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Function to generate response
def generate_response(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to device
    outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model with some queries
sample_queries = [
    "How can I reset my password?",
    "What is your return policy?",
    "My order hasn't arrived yet."
]

for query in sample_queries:
    response = generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print()
