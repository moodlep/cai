Some setup.py changes were required for SFT and later for DPO. 
I am listing both as an fyi

**SFT**: 

There was an error from TRL:  SFTTrainer complained about  init_model_qwargs being an invalid parameter.  
A search revealed this was an old problme and the recommendation was to change the >= to == for torch and trl. 

   "torch==2.1.2",
    "trl==0.9.6",

**DPO**: 

Error: trl dpo AttributeError: 'generator' object has no attribute 'generate'

This is a documented issue: https://github.com/huggingface/trl/issues/2292
Followed the suggestions and found upgrading trl did not work but limiting transformers did: 

    "torch==2.1.2",
    "transformers<=4.45",  # "transformers>=4.39.3",
    "trl==0.9.6",

