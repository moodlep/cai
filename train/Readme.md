# Training Recipes for smollm2 and mistral using PEFT (QLoRA)

## Prerequisites

These recipes require the installation of the `alignment-handbook` repository from Hugging Face. The `alignment-handbook` provides many useful scripts and resources for supervised fine-tuning and preference training models, including a recipe for training a Constitutional AI (CAI) model. 

## Installation

To get started, follow these steps to set up your environment:

1. **Install the `alignment-handbook` repository, following instructions from the repo**:
    ```sh
    https://github.com/huggingface/alignment-handbook.git
    ```

2. **Clone this repository**:
    ```sh
    git clone https://github.com/moodlep/cai.git
    ```

## Overview

This repository contains recipes for training `smollm2` and `mistral` models on a single GPU (16/24GB VRAM) server  using Parameter-Efficient Fine-Tuning (PEFT) with QLoRA. These recipes are modified versions of the Constitutional AI recipes provided by the Hugging Face team within their `alignment-handbook`, fitted with QLoRA setup.

## Running the Recipes

### Pre-requisites:

- To run these recipes, ensure you have installed the `alignment-handbook` as suggested above. Then, access and execute the recipes provided in this repository.

- For convenience, datasets are expected to available in the HuggingFace hub. More details for how the datasets should be structured will follow.   

### Notes

- **Library Conflicts**: Some library conflicts were noted during the setup. The `setup.py` files used for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) are located in `recipes/hf_setup`.

- **Single-GPU Configuration**: For training on a single GPU with QLoRA, the `multi-gpu.yaml` file with the `accelerate` configuration was passed `num_processes=1` and the cai training config script was passed `--load_in_4bit=true` to activate the QLoRA setup:
    ```sh
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/cai/mistral/sft/config_anthropic_qlora.yaml --load_in_4bit=true
    ```
- **Hugging Face Token**: An HF token with write access is required as we write models back to the Hugging Face hub. Ensure you have your token set up in your environment:
    ```sh
    export HF_TOKEN=your_huggingface_token
    ```

- **Datasets**:  These scripts can produce a baseline CAI training run using the [HuggingFaceH4/cai-conversation-harmless](https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless?row=1) pre-generated SFT dataset from HuggingFace and the [Ultrachat_200k dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), also available on the HuggingFace hub.  

- **Models**: To train on other models than Mistral or smolLM2, copy the recipes and adjust as needed. 

## Contact

For any questions or inquiries, please contact us.

## Acknowledgements

We would like to thank the Hugging Face team for their support and the resources provided in the `alignment-handbook`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.