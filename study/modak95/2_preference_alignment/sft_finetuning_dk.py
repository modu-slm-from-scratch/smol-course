from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "HuggingFaceTB/SmolLM2-135M"
# model_name = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float32,
).to(device)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

def process_dataset(sample):
    messages = [{"role": "user", "content": sample['instruction']},
                {"role": "assistant", "content": sample['output']}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": input_text}


ds = load_dataset(path="CarrotAI/ko-instruction-dataset")
ds = ds["train"].train_test_split(test_size=0.2, seed=42)

ds["train"] = ds["train"].map(process_dataset)
ds["test"] = ds["test"].map(process_dataset)


output_dir = "/home/user/smol/smol-course/dk_dir/models/SmolLM2-FT-ko-instruction-dataset"


sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=10,  
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=20,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=50,
    hub_model_id="SmolLM2-135M-dk-test",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
    eval_dataset=ds["test"],
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
trainer.push_to_hub()
