from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch

# ✅ Correct adapter folder path
adapter_path = "falcon_lora_output/checkpoint-125"

# Load adapter config
peft_config = PeftConfig.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True,
    device_map="auto"
)

# Load adapter on top of base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)


# Test input
prompt = "The movie was absolutely wonderful because"
result = pipe(prompt)

# Output
print("\nGenerated Output:\n", result[0]["generated_text"])
