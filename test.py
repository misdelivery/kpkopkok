from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "jovyan/Swallow-MS-7b-v0.1-ChatVector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<s>[INST]まど☆マギで一番かわいいのは誰ですか？[/INST]"

generator_params = dict(
    max_new_tokens = 256,
    do_sample = True,
    temperature = 0.99,
    top_p = 0.95,
    pad_token_id = tokenizer.eos_token_id,
)

output = generator(
    prompt,
    **generator_params,
)

print(output[0]["generated_text"])