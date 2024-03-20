from transformers import Trainer, TrainingArguments, AutoTokenizer, MambaForCausalLM, MambaConfig
import torch
from datasets import load_dataset
import accelerate
import re

tokenizer = AutoTokenizer.from_pretrained("sonoisa/byt5-small-japanese")

mamba_config = MambaConfig(
    hidden_size = 768,
    num_hidden_layers = 24,
    vocab_size = len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False
)

model = MambaForCausalLM(mamba_config)

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids)[0]
        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss

from transformers import DataCollatorForLanguageModeling

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, max_length=1000, truncation=True, padding="max_length")

def preprocess_function(examples):
    text = examples["text"]
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if para.startswith("_START_ARTICLE_") or para.startswith("_START_SECTION_"):
            continue
        if para.startswith("_START_PARAGRAPH_"):
            para = para[len("_START_PARAGRAPH_"):]
        if len(para) > 0:
            for line in para.split("_NEWLINE_"):
                line = re.sub(r'(概要|略歴・人物|生涯|経歴)', '', line)
                if len(line) > 0:
                    lines.append(line)
    return {"text": "".join(lines)}

wiki_dataset = load_dataset("range3/wiki40b-ja", split="train")

# データセットをプリプロセス
processed_dataset = wiki_dataset.map(preprocess_function, remove_columns=["wikidata_id", "version_id"])

# データセットをトークン化
tokenized_datasets = processed_dataset.map(tokenize_function, batched=True)

# データコレーターの準備
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    bf16=True,
    output_dir="mambabyte-130m_checkpoints",
    report_to="wandb",
    save_strategy='steps',
    save_steps=500,
    save_total_limit=5,
    num_train_epochs=3,
    per_device_train_batch_size=40,
    lr_scheduler_type='cosine',
    learning_rate=6e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
)

trainer = MambaTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()
trainer.save_model('mambabyte-130m')
