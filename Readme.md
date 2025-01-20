## Directory structure
```
cai/
├── data/    
│   ├── scripts/             
│   │   ├── generate_principles.py # Optional if we get to do this programatically
│   │   ├── generate_eval_data.py # Optional
│   │   ├── generate_sft_data.py # Generate SFT data
│   │   ├── generate_rt_data.py # Generate red-teaming data
│   │   ├── generate_preference_data.py # Generate preference data
│   │   ├── mix_data.py       # TBD
│   │   └── utils.py
│   ├── config/               # YAML config files used for running data generation scripts
│   │   ├── ro_sft_config.yaml # Config for generating SFT data based on RO principles
│   │   └── sa_preference_config.py # Config for generating preference data based on SA principles
│   └── datasets/             # Finalized datasets for training/evaluation
│       ├── ro_rt_prompts.json
│       ├── ro_principles.json
│       ├── ro_violations.json
│       ├── sa_rt_prompts.json
│       ├── sa_principles.json
│       ├── sa_violations.json
│       ├── ro_preference_data.json
│       ├── sa_preference_data.json
│       ├── ro_sft_data.json
│       ├── sa_sft_data.json
│       ├── ro_eval_data.json
│       ├── sa_eval_data.json
│       └── sft_data.json
├── models/                   # Model-related code and checkpoints
│   ├── scripts/
│   │   ├── model_utils.py
│   │   └── model_config.py
│   ├── configs/
│   └── checkpoints/             
├── train/                    # TBD
│   ├── scripts/                  
│   │   ├── ppo.py
│   │   ├── dpo.py     
│   │   ├── sft.py
│   │   └── train_utils.py    
│   ├── configs/             
│   │   ├── ppo_config.yaml          
│   │   └── dpo_config.yaml        
│   └── results/
│       └── logs.json                      
├── eval/                     # TBD
├── experiments/              # TBD             
├── tests/                    # TBD
├── utils/                    # TBD
│   ├── logging.py         
│   └── storage.py            
├── docs/                     
│   ├── README.md             # Overview of the project
│   ├── CONTRIBUTING.md       
│   ├── machine_ethics.md       
│   └── experiments.md        # Experiment design and guidelines
└── README.md                 # Top-level project README
```


