from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import torch

# Step 1: Model & tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Step 2: Load dataset (IMDb for sentiment classification)
dataset = load_dataset("imdb")

# Step 3: Tokenization function
def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = tokens["input_ids"]  # required for causal LM loss
    return tokens


tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",   # use CUDA if available, else CPU
    trust_remote_code=True
)

# Step 5: Prepare model for 8-bit training
model = prepare_model_for_kbit_training(model)

# Step 6: Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["query_key_value"]  # Falcon-specific
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 7: TrainingArguments
training_args = TrainingArguments(
    output_dir="./falcon_lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# Step 8: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(1000))  # small subset for testing
)

# Step 9: Train
trainer.train()
