<img src="assets/bluedotlogo.jpeg" alt="Blue Dot Alignment Course Logo" width="50"/> 

## Blue Dot AI Safety Fundamentals Alignment Course Project on CAI (Constitutional AI)

Welcome to our Blue Dot AI Safety Fundamentals alignment course project on Constitutional AI (CAI). 

### Abstract
Within the broader Machine Ethics (ME) discussion, most value alignment work focuses on identifying the universal set of values that should reflect everyone’s preferences. We believe that the centralisation/standardisation of knowledge and values, accelerated by generative AI, presents a challenge for the preservation of traditional cultural, ethical, and intellectual diversity. 

At the same time, if carefully developed and deployed, AI systems could provide interactive repositories of underrepresented perspectives. We sought to evaluate how LLMs post-trained on principles reflecting different countries and groups compared to existing frontier models. We produced a set of synthetic datasets for aligning models with principles inspired from the South African and Romanian cultures respectively. 

After analysing the datasets generated, we observe that in some cases GPT-4o demonstrates a nuanced understanding of the principles used while other principles reveal a clear knowledge gap. Our writeup discusses whether different countries should fine-tune their own models versus educating their citizens about the behaviour of frontier models and best practices for using them. 

Here is a link to our [project writeup](https://docs.google.com/document/d/1441OXG_tZdbLY8ZNpsNSJsv9iDx2Nql77ZnryVleCL0/edit?usp=sharing); please note this is a living document as we continue to work on this project. 

## Repository Contents

### Dataset Generation Process

**Generating Datasets**: Use the scripts in the [data](https://github.com/moodlep/cai/tree/main/data) folder to generate and preprocess your own datasets. Upload datasets to the Hugging Face hub for training custom CAI models. 

In the `data` folder you can find the files relevant to this project including: 

- **Constitutions**: Various constitutions (sets of principles) used for training and evaluation.
- **Datasets**: Datasets required for training and testing the models, including red teaming, SFT and preference datasets.
- **Scripts**: Python scripts for generating and processing datasets.

### Training Recipes

**Training Models**: Follow the instructions in the [train](https://github.com/moodlep/cai/tree/main/train) folder to train `smollm2` and `mistral` models using pre-existing datasets on the HuggingFace hub or datasets using the datasets generated above. 

- **Train Folder**: Contains recipes for training `smollm2` and `mistral` models on a single GPU server using PEFT (QLoRA) for the Constitutional AI process, that includes multiple training steps including a Supervised Fine-Tuning (SFT) and a preference training (DPO) step.

For a more details on how to get started using Hugging Face's alignment-handbook refer to our [blog post](https://emergent-behaviour.blogspot.com/2025/02/training-constitutional-ai-model-using.html).  

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact us.

## Acknowledgements
We would like to thank the [Blue Dot AI Safety Fundamentals](https://aisafetyfundamentals.com/) team for their exceptional course, support and guidance.

We would also like to acknowledge [HuggingFace](https://huggingface.co/) for their tireless production of libraries, courses and documentation and the 
[ARENA](https://github.com/callummcdougall/ARENA_3.0) project for their excellent tutorials. 
