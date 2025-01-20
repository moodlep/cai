**SFT Training**

The command line for running the script is below. It is set up to fine-tune a smollm2 instruct model.  

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/cai/smol/sft/config_anthropic_smollm.yaml --load_in_4bit=true
```

This QLORA profile was run successfully on a machine with a 24GB GPU (80 cores and 200GB RAM). 
With a very limited dataset (7k records of mixed data) it ran for 1+ hours. 

It requires an HF token with write abilities. 

**DPO Training**

The command line for running the script is below. It is set up to fine-tune a SFT model from the hub.  

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/cai/smol/dpo/config_anthropic_smol_qlora.yaml --load_in_4bit=true
```

Sample Training notes: 

This QLORA profile was run successfully on a machine with a 24GB GPU (80 cores and 200GB RAM). 
The config uses a paged optimiser that runs training in 1h17 min for 2 epochs with eval runs taking 16 min/ 


```
[INFO|trainer.py:2243] 2025-01-09 13:34:16,719 >> ***** Running training *****
[INFO|trainer.py:2244] 2025-01-09 13:34:16,720 >>   Num examples = 4,119
[INFO|trainer.py:2245] 2025-01-09 13:34:16,720 >>   Num Epochs = 1
[INFO|trainer.py:2246] 2025-01-09 13:34:16,720 >>   Instantaneous batch size per device = 4
[INFO|trainer.py:2249] 2025-01-09 13:34:16,720 >>   Total train batch size (w. parallel, distributed & accumulation) = 16
[INFO|trainer.py:2250] 2025-01-09 13:34:16,720 >>   Gradient Accumulation steps = 4
[INFO|trainer.py:2251] 2025-01-09 13:34:16,720 >>   Total optimization steps = 257
[INFO|trainer.py:2252] 2025-01-09 13:34:16,723 >>   Number of trainable parameters = 18,087,936
```
