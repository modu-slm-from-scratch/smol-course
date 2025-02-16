import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from huggingface_hub import login
f = open("/home/user/smol/smol-course/key.txt", 'r')
key = f.read()
login(token = key)


from datasets import load_dataset

# TODO: define your dataset and config using the path and name parameters
# dataset = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
# dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
# model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

def process_dataset(sample):
    messages = [{"role": "user", "content": sample['instruction']},
                {"role": "assistant", "content": sample['output']}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": input_text}


dataset = load_dataset(path="CarrotAI/ko-instruction-dataset")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

dataset["train"] = dataset["train"].map(process_dataset)
dataset["test"] = dataset["test"].map(process_dataset)


# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-MyDataset"
finetune_tags = ["smol-course", "module_1"]

from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 6
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Training configuration
# Hyperparameters based on QLoRA paper recommendations
args = SFTConfig(
    # Output settings
    output_dir="/home/user/smol/smol-course/dk_dir/models/llama_1b-lora-ko-instruction-dataset",  # Directory to save model checkpoints
    # Training duration
    num_train_epochs=5,  # Number of training epochs
    # Batch size settings
    per_device_train_batch_size=4,  # Batch size per GPU
    gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch
    # Memory optimization
    gradient_checkpointing=True,  # Trade compute for memory savings
    # Optimizer settings
    optim="adamw_torch_fused",  # Use fused AdamW for efficiency
    learning_rate=2e-4,  # Learning rate (QLoRA paper)
    max_grad_norm=0.3,  # Gradient clipping threshold
    # Learning rate schedule
    warmup_ratio=0.03,  # Portion of steps for warmup
    lr_scheduler_type="constant",  # Keep learning rate constant after warmup
    # Logging and saving
    logging_steps=10,  # Log metrics every N steps
    save_strategy="epoch",  # Save checkpoint every epoch
    # Precision settings
    bf16=True,  # Use bfloat16 precision
    # Integration settings
    push_to_hub=False,  # Don't push to HuggingFace Hub
    report_to="none",  # Disable external logging
)

max_seq_length = 1512  # max sequence length for model and packing of the dataset

# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # LoRA configuration
    max_seq_length=max_seq_length,  # Maximum sequence length
    tokenizer=tokenizer,
    packing=True,  # Enable input packing for efficiency
    dataset_kwargs={
        "add_special_tokens": False,  # Special tokens handled by template
        "append_concat_token": False,  # No additional separator needed
    },
)

trainer.train()

# save model
trainer.save_model()

from peft import AutoPeftModelForCausalLM


# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(
    args.output_dir, safe_serialization=True, max_shard_size="2GB"
)


del model
del trainer
torch.cuda.empty_cache()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Load Model with PEFT adapter
tokenizer = AutoTokenizer.from_pretrained(finetune_name)
model = AutoPeftModelForCausalLM.from_pretrained(
    finetune_name, device_map="auto", torch_dtype=torch.float16
)
pipe = pipeline(
    "text-generation", model=merged_model, tokenizer=tokenizer, device=device
)

ko_prompts = [
    "1 + 2 * 6 은 뭐야?",
    "5줄 이내로 자기소개를 작성해줘.",
    "이 문장을 10단어 이내로 요약해줘: ‘최근 인공지능 기술이 급속도로 발전하면서 다양한 산업에서 혁신적인 변화를 이끌고 있다.",
    "한국에서 인기 있는 전통 음식 3가지를 설명해줘."
]
# prompts = [
#     "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
#     "Write a Python function to calculate the factorial of a number.",
#     "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
#     "What is the difference between a fruit and a vegetable? Give examples of each.",
# ]


def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipe(
        prompt,
    )
    return outputs[0]["generated_text"][len(prompt) :].strip()


for prompt in ko_prompts:
    print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(prompt)}")
    print("-" * 50)


