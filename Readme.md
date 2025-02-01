<img src="assets/bluedotlogo.jpeg" alt="Blue Dot Alignment Course Logo" width="50"/> 

## Blue Dot AI Safety Fundamentals Alignment Course Project on CAI (Constitutional AI)

Welcome to our Blue Dot AI Safety Fundamentals alignment course project on Constitutional AI (CAI). 

### Abstract
Within the broader Machine Ethics (ME) discussion, most value alignment work focuses on identifying the universal set of values that should reflect everyone’s preferences. We believe that the centralisation/standardisation of knowledge and values, accelerated by generative AI, presents a challenge for the preservation of traditional cultural, ethical, and intellectual diversity. 

At the same time, if carefully developed and deployed, AI systems could provide interactive repositories of underrepresented perspectives. We sought to evaluate how LLMs post-trained on principles reflecting different countries and groups compared to existing frontier models. We produced a set of synthetic datasets for aligning models with principles inspired from the South African and Romanian cultures respectively. 

After analysing the datasets generated, we observe that in some cases GPT-4o demonstrates a nuanced understanding of the principles used while other principles reveal a clear knowledge gap. Our writeup discusses whether different countries should fine-tune their own models versus educating their citizens about the behaviour of frontier models and best practices for using them. The link will be provided shortly. 

## Repository Contents

### Training Recipes
- **Train Folder**: Contains recipes for training `smollm2` and `mistral` models on a single GPU server using PEFT (QLoRA).

### Data
- **Constitutions**: Various constitutions used for training and evaluation.
- **Datasets**: Datasets required for training and testing the models.
- **Scripts**: Scripts for generating and processing datasets.


### Usage
1. **Training Models**: Follow the instructions in the [train](https://github.com/moodlep/cai/tree/main/train) folder to train `smollm2` and `mistral` models.
2. **Generating Datasets**: Use the scripts in the [data](https://github.com/moodlep/cai/tree/main/data) folder to generate and preprocess datasets.

### Configuration
- Configuration files for training and data generation are located in the `config` folder. Modify these files as needed to suit your requirements.

## Contributing
We welcome contributions! Please read our Contributing Guidelines for more details.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact us.

## Acknowledgements
We would like to thank the [Blue Dot AI Safety Fundamentals](https://aisafetyfundamentals.com/) team for their support and guidance.
We would also like to acknowledge [HuggingFace](https://huggingface.co/) for their tireless production of libraries, courses and documentation and 
[ARENA](https://github.com/callummcdougall/ARENA_3.0) for their excellent tutorials
