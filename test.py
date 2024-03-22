from transformers import MambaForCausalLM, AutoTokenizer
import transformers
import torch

model = MambaForCausalLM.from_pretrained("misdelivery/frankenmamba-5.6B",torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('misdelivery/frankenmamba-5.6B')

model.to('cuda:0')
with torch.no_grad():
    input_ids = tokenizer.encode("Shakespeare is", add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
    input_ids.to(model.device),
    max_length=256,
    do_sample=True,
    temperature=0.6,
    repetition_penalty=1.1
)

print(tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True))

with torch.no_grad():
    input_ids = tokenizer.encode("夏目漱石は、", add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
    input_ids.to(model.device),
    max_length=256,
    do_sample=True,
    temperature=0.6,
    repetition_penalty=1.1
)

print(tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True))
