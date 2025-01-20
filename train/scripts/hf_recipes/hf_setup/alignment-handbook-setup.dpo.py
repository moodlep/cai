# just the dependencies for the alignment handbook setup for training DPO
# clashes between trl and transformers versions. 
deps = [
    "accelerate>=0.29.2",
    "bitsandbytes>=0.43.0",
    "black>=24.4.2",
    "datasets>=2.18.0",
    "deepspeed>=0.14.4",
    "einops>=0.6.1",
    "evaluate==0.4.0",
    "flake8>=6.0.0",
    "hf-doc-builder>=0.4.0",
    "hf_transfer>=0.1.4",
    "huggingface-hub>=0.19.2,<1.0",
    "isort>=5.12.0",
    "ninja>=1.11.1",
    "numpy>=1.24.2",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft>=0.9.0",
    "protobuf<=3.20.2",  # Needed to avoid conflicts with `transformers`
    "pytest",
    "safetensors>=0.3.3",
    "sentencepiece>=0.1.99",
    "scipy",
    "tensorboard",
    "torch==2.1.2",
    "transformers<=4.45",  # "transformers>=4.39.3",
    "trl==0.9.6",
    # "trl>=0.12", # recommended by git issue but did not work so changed transformers as above. https://github.com/huggingface/trl/issues/2292
    "jinja2>=3.0.0",
    "tqdm>=4.64.1",
]
