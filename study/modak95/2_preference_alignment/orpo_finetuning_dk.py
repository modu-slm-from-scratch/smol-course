
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
from huggingface_hub import login

f = open("/home/user/smol/smol-course/key.txt", 'r')
key = f.read()
login(token = key)

dataset = load_dataset(path="maywell/ko_Ultrafeedback_binarized")

dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)


model_name = "HuggingFaceTB/SmolLM2-135M"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float32,
).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = setup_chat_format(model, tokenizer)

finetune_name = "SmolLM2-ORPO-ko-ultrafeedback_binarized"
finetune_tags = ["smol-course", "module_1"]


orpo_args = ORPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
    num_train_epochs=30,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="none",
    output_dir="/home/user/smol/smol-course/dk_dir/models/SmolLM2-ORPO-ko-ultrafeedback_binarized",
    hub_model_id=finetune_name,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train() 

trainer.save_model("/home/user/smol/smol-course/dk_dir/models/SmolLM2-ORPO-ko-ultrafeedback_binarized")

if os.getenv("HF_TOKEN"):
    trainer.push_to_hub(tags=finetune_tags)