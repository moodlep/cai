Script 1 Jan 2025

0. Go to /home/perusha/git_repos/alignment-handbook

1. conda activate handbook

2. Changes to yaml files: 
NB - any changes made to setup.py or any of the yaml files requires a new build of the path: 
(handbook) perusha@pkaiju:~/git_repos/alignment-handbook$ python -m pip install .

* Created a qlora profile for cai anthropic sft
recipes/constitutional-ai/sft/config_anthropic_qlora.yaml
from /home/perusha/git_repos/alignment-handbook/recipes/zephyr-7b-beta/sft/config_qlora.yaml

3. Before running a test, update the HF token with a write token!

huggingface-cli logout
huggingface-cli login

The current active token is: `CAI_token`


4. Do we want to switch to the zephyr model - I may have access to this!! Or test Mistral first? 

Test Mistral first with new HF token

5. Run script: 
OLD: 
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/fsdp_qlora.yaml scripts/run_sft.py recipes/constitutional-ai/sft/config_anthropic.yaml
NEW: 
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/fsdp_qlora.yaml scripts/run_sft.py recipes/constitutional-ai/sft/config_anthropic_qlora.yaml

6. Error message


Looking into how Zephyr was trained for qlora: 
(see Readme.md here: https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta)

# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml


TRY: 
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/constitutional-ai/sft/config_anthropic_qlora.yaml --load_in_4bit=true

Success!! But expected to take 83 hours!!

[INFO|trainer.py:734] 2025-01-01 16:12:18,785 >> Using auto half precision backend
2025-01-01 16:12:18 - INFO - __main__ - *** Train ***
[INFO|trainer.py:2362] 2025-01-01 16:12:19,074 >> ***** Running training *****
[INFO|trainer.py:2363] 2025-01-01 16:12:19,075 >>   Num examples = 140,168
[INFO|trainer.py:2364] 2025-01-01 16:12:19,075 >>   Num Epochs = 1
[INFO|trainer.py:2365] 2025-01-01 16:12:19,075 >>   Instantaneous batch size per device = 8
[INFO|trainer.py:2368] 2025-01-01 16:12:19,075 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:2369] 2025-01-01 16:12:19,075 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2370] 2025-01-01 16:12:19,075 >>   Total optimization steps = 2,190
[INFO|trainer.py:2371] 2025-01-01 16:12:19,078 >>   Number of trainable parameters = 41,943,040
  0%|                                                                                            | 0/2190 [00:00<?, ?it/s][WARNING|logging.py:328] 2025-01-01 16:12:19,685 >> The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in torch.bfloat16.
[rank0]:[W101 16:12:25.508892781 reducer.cpp:1400] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
{'loss': 9.323, 'grad_norm': 2.925379753112793, 'learning_rate': 9.132420091324201e-08, 'epoch': 0.0}                     
{'loss': 9.4034, 'grad_norm': 2.7770936489105225, 'learning_rate': 4.5662100456621004e-07, 'epoch': 0.0}                  
  0%|▎                                                                               | 8/2190 [19:21<87:45:03, 144.78s/it]

7. Asked copilot what can speed this up and make more use of the CPU cores: 
With the current config in this file, the SFT training is estimated to be 83 hours. how can I modify the config in this file to make more use of the 80 cores and 200GB of RAM on the server to speed things up? I only have one GPU with 24GB VRAM. What types of settings should I try to change?

To make better use of the available CPU cores and RAM to speed up the SFT training, you can adjust several settings in your configuration file. Here are some suggestions:

Increase preprocessing_num_workers: Utilize more CPU cores for data preprocessing.
Increase gradient_accumulation_steps: This will reduce the memory load on the GPU by accumulating gradients over more steps.
Enable mixed precision training: This can help reduce the memory usage on the GPU.
Adjust batch size: Increase the batch size if the GPU memory allows it.
Use CPU for certain operations: Offload some operations to the CPU to balance the load.
Here is an updated version of your configuration file with these changes:

# Model arguments
model_name_or_path: mistralai/Mistral-7B-v0.1
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# LoRA arguments
load_in_4bit: true
use_peft: true
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

# Data training arguments
chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
dataset_mixer:
  HuggingFaceH4/cai-conversation-harmless: 1.0
  HuggingFaceH4/ultrachat_200k: 1.0
dataset_splits:
- train_sft
- test_sft
preprocessing_num_workers: 40  # Increase to utilize more CPU cores

# SFT trainer config
bf16: true
do_eval: true
do_train: true
eval_strategy: epoch # One of ["no", "steps", "epoch"]
gradient_accumulation_steps: 8  # Increase to reduce GPU memory usage
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
hub_model_id: mistral-7b-sft-constitutional-ai
hub_strategy: every_save
learning_rate: 2.0e-05
log_level: info
logging_steps: 5  
logging_strategy: steps
lr_scheduler_type: cosine
max_seq_length: 2048
max_steps: -1
num_train_epochs: 1
output_dir: data/mistral-7b-sft-constitutional-ai
overwrite_output_dir: true
per_device_train_batch_size: 2  # Adjust based on GPU memory
per_device_eval_batch_size: 2  # Adjust based on GPU memory


2 Jan 2025

1. Try smaller subsets of the datasets
dataset_mixer:
  HuggingFaceH4/cai-conversation-harmless: 0.15
  HuggingFaceH4/ultrachat_200k: 0.25

This can be done in 1 day!! 

[INFO|trainer.py:734] 2025-01-02 11:18:28,452 >> Using auto half precision backend
2025-01-02 11:18:28 - INFO - __main__ - *** Train ***
[INFO|trainer.py:2362] 2025-01-02 11:18:28,824 >> ***** Running training *****
[INFO|trainer.py:2363] 2025-01-02 11:18:28,824 >>   Num examples = 34,882
[INFO|trainer.py:2364] 2025-01-02 11:18:28,825 >>   Num Epochs = 1
[INFO|trainer.py:2365] 2025-01-02 11:18:28,825 >>   Instantaneous batch size per device = 8
[INFO|trainer.py:2368] 2025-01-02 11:18:28,825 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:2369] 2025-01-02 11:18:28,825 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2370] 2025-01-02 11:18:28,825 >>   Total optimization steps = 545
[INFO|trainer.py:2371] 2025-01-02 11:18:28,828 >>   Number of trainable parameters = 41,943,040
  0%|                                                                                           | 0/545 [00:00<?, ?it/s][WARNING|logging.py:328] 2025-01-02 11:18:30,071 >> The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in torch.bfloat16.
[rank0]:[W102 11:18:35.134450544 reducer.cpp:1400] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
{'loss': 9.0061, 'grad_norm': 2.602250099182129, 'learning_rate': 3.6363636363636366e-07, 'epoch': 0.0}                 
  0%|▏                                                                              | 1/545 [02:24<21:52:52, 144.80s/it]


Even smalller run: 0.05 ultra + 0.05 CAI = 7000 records  @ 4 hours!!

[INFO|trainer.py:734] 2025-01-02 11:24:29,734 >> Using auto half precision backend
2025-01-02 11:24:29 - INFO - __main__ - *** Train ***
[INFO|trainer.py:2362] 2025-01-02 11:24:30,062 >> ***** Running training *****
[INFO|trainer.py:2363] 2025-01-02 11:24:30,062 >>   Num examples = 6,995
[INFO|trainer.py:2364] 2025-01-02 11:24:30,062 >>   Num Epochs = 1
[INFO|trainer.py:2365] 2025-01-02 11:24:30,062 >>   Instantaneous batch size per device = 8
[INFO|trainer.py:2368] 2025-01-02 11:24:30,062 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:2369] 2025-01-02 11:24:30,062 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2370] 2025-01-02 11:24:30,062 >>   Total optimization steps = 109
[INFO|trainer.py:2371] 2025-01-02 11:24:30,065 >>   Number of trainable parameters = 41,943,040

{'loss': 9.7005, 'grad_norm': 2.635891914367676, 'learning_rate': 1.8181818181818183e-06, 'epoch': 0.01}                
  1%|▋                                                                               | 1/109 [02:25<4:21:37, 145.35s/it]
