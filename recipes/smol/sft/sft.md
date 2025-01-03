The command line for running the script is below. It is set up to fine-tune a smollm2 instruct model.  

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/constitutional-ai/sft/config_anthropic_smollm.yaml --load_in_4bit=true

This QLORA profile was run successfully on a machine with a 24GB GPU (80 cores and 200GB RAM). 
With a very limited dataset (7k records of mixed data) it ran for 1+ hours. 

It requires an HF token with write abilities. 
