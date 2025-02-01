# Data Generation for SFT, Preference/DPO, and Red Teaming

This folder contains scripts and resources for generating datasets required for Supervised Fine-Tuning (SFT), Preference/DPO, and Red Teaming. Follow the instructions below to generate each type of dataset.

## Prerequisites

A more concise `requirements.txt` file will be added shortly however the [ARENA repo](https://github.com/callummcdougall/ARENA_3.0) requirements file should be sufficient for running the dataset scripts. 

## Dataset Generation 

### 1. Generating Red Teaming Datasets

To generate datasets for Red Teaming, use the `generate_rt_data.py` script. This script processes raw data and formats it for red teaming exercises. For additional args, review the `generate_rt_data.py` file.

#### Usage
```sh
python generate_rt_data.py --config data/config/rt_config.yaml --output-file data/datasets/red_team/red_team_dataset.json
```
#### Arguments
- `--config`: Path to the configuration file for preference data generation.
- `--output-file`: Path to the output file where the generated dataset will be saved.


### 2. Generating SFT Datasets

To generate datasets for Supervised Fine-Tuning (SFT), use the `generate_sft_data.py` script. This script processes raw data and formats it for SFT. Review the additional args in `generate_sft_data.py`.  

#### Usage
```sh
python generate_sft_data.py --config data/config/sft_config.yaml --output-file data/datasets/sft/sft_dataset.json
```

#### Arguments
- `--config`: Path to the configuration file for preference data generation.
- `--output-file`: Path to the output file where the generated dataset will be saved.

### 3. Generating Preference/DPO Datasets

To generate datasets for Preference/DPO, use the `generate_preference_data.py` script. This script processes raw data and formats it for preference training and Direct Preference Optimization (DPO). Additional args are in the `generate_preference_data.py` file. 

#### Usage
```sh
python generate_preference_data.py --config data/config/preference_config.yaml --output-file data/datasets/preference/preference_dataset.json
```

#### Arguments
- `--config`: Path to the configuration file for preference data generation.
- `--output-file`: Path to the output file where the generated dataset will be saved.

### 4. Configuration Files
The configuration files for each type of data generation are located in the config folder. 
There is a single yaml file per country.  
Each file contains the prompts required to generate the red teaming, SFT and preference datasets. 
Modify these files as needed to suit your requirements.