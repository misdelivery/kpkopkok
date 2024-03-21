from transformers import MambaForCausalLM, MambaConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

mamba_config = MambaConfig(vocab_size = 50280, hidden_size = 2560, state_size = 16, num_hidden_layers = 64, layer_norm_epsilon = 1e-05, pad_token_id = tokenizer.pad_token_id, bos_token_id = 0, eos_token_id = tokenizer.eos_token_id, expand = 2, conv_kernel = 4, use_bias = False, use_conv_bias = True, hidden_act = 'silu', initializer_range = 0.1, residual_in_fp32 = True, time_step_rank = 'auto', time_step_scale = 1.0, time_step_min = 0.001, time_step_max = 0.1, time_step_init_scheme = 'random', time_step_floor = 0.0001, rescale_prenorm_residual = False, use_cache = False)
model = MambaForCausalLM.from_pretrained('kotoba-tech/kotomamba-2.8B-CL-v1.0', config=mamba_config, low_cpu_mem_usage=True).to('cuda:0')

model.push_to_hub('misdelivery/kotomamba-2.8B-CL-v1.0-hf')
tokenizer.push_to_hub("misdelivery/kotomamba-2.8B-CL-v1.0-hf")
