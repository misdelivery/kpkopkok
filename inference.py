from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig
import torch

tokenizer = AutoTokenizer.from_pretrained('mambabyte-130m_checkpoints/checkpoint-17000')
if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

mamba_config = MambaConfig(
    hidden_size = 768,
    num_hidden_layers = 24,
    vocab_size = len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True
)

model = MambaForCausalLM.from_pretrained('mambabyte-130m_checkpoints/checkpoint-17000', config=mamba_config, torch_dtype=torch.bfloat16,)

import torch
model.to('cuda:0')
with torch.no_grad():
    input_ids = tokenizer.encode("夏目漱石は、", add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
    input_ids.to(model.device),
    max_length=800,
)

print(tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True))

with torch.no_grad():
    input_ids = tokenizer.encode("夏目漱石は、", add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
    input_ids.to(model.device),
    max_length=800,
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.1
)

print(tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True))

with torch.no_grad():
    input_ids = tokenizer.encode("夏目漱石は、", add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
    input_ids.to(model.device),
    max_length=800,
    do_sample=True,
    temperature=0.6,
    repetition_penalty=1.1
)
