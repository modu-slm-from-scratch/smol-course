from huggingface_hub import login
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

f = open("/home/user/smol/smol-course/key.txt", 'r')
key = f.read()
login(token = key)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = load_dataset(path="maywell/ko_Ultrafeedback_binarized", split="train")

model_name = "/home/user/smol/smol-course/dk_dir/models/SmolLM2-FT-ko-instruction-dataset"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float32,
).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

finetune_name = "SmolLM2-FT-ko-DPO-ko_Ultrafeedback"
finetune_tags = ["smol-course", "module_1"]

# Training arguments
training_args = DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir="smol_dpo_output",
    warmup_steps=100,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    hub_model_id=finetune_name,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)


trainer = DPOTrainer(
    # The model to be trained
    model=model,
    # Training configuration from above
    args=training_args,
    # Dataset containing preferred/rejected response pairs
    train_dataset=dataset,
    # Tokenizer for processing inputs
    processing_class=tokenizer,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    # beta=0.1,
    # Maximum length of the input prompt in tokens
    # max_prompt_length=1024,
    # Maximum combined length of prompt + response in tokens
    # max_length=1536,
)



# Train the model
trainer.train()

# Save the model
trainer.save_model("/home/user/smol/smol-course/dk_dir/models/SmolLM2-FT-ko-DPO-ko_Ultrafeedback")

# Save to the huggingface hub if login (HF_TOKEN is set)
if os.getenv("HF_TOKEN"):
    trainer.push_to_hub(tags=finetune_tags)
