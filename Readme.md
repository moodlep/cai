Repository for the Blue Dot AI Safety Fundamentals alignment course project on CAI (Constitutional AI) 

The repo contains recipes for training smollm2 and mistral on a single GPU server using PEFT (QLoRA). 

The recipes are modified versions of the constitutional-AI recipes provide by the HuggingFace team within their alignment-handbook. 
To run these recipes, install the alignment-handbook as suggested, then access these recipes. 

Some notes: 
* some library conflicts were noted. The setup.py files used for SFT and DPO are in recipes/hf_setup
* no changes were made to the multi-gpu.yaml file with accelerate config. 
* an HF token with write access is required as we write models back to the hub. 

