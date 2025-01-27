import json
import os
from datetime import datetime

import openai
from openai import OpenAI
import json

from data_utils import get_principles_from_constitution, generate_formatted_response, get_config, get_red_team_prompts
from structured_response_formats import SFTData, SFTDataset
import dotenv
import argparse


# add args
parser = argparse.ArgumentParser(description='Generate SFT Dataset')
parser.add_argument('--config', type=str, default='data/config/sa_config.yaml', help='path to the config file')
parser.add_argument('--config-category', type=str, default='sft', help='type of data generation: red_team, sft, dpo')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model to use for generation: gpt-4o-mini, gpt-4o')
parser.add_argument('--verbose', type=str, default=True, help='verbosity setting for openAI model: True/False')
parser.add_argument('--output-file', type=str, default="data/datasets/sft_dataset", help='output file name')
parser.add_argument('--include-few-shot', type=str, default=False, help='include few shot examples')
parser.add_argument('--src-red-team-prompts', type=str, default='data/datasets/red_team_prompts_20250127-113940.json', help='latest red team prompts  file')
parser.add_argument('--debug', type=str, default=False, help='debug')

args = parser.parse_args()


# Configure your OpenAI API key
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()

# Load config
config = get_config(args.config)
sft_config = config[args.config_category]

# Load principles
principles = get_principles_from_constitution(config['src_principles'])

# read red team prompts from json
rt_prompts = get_red_team_prompts(args.src_red_team_prompts)

# Run to see the questions that the model generated
# gen_prompts = GenPrompts(system_prompt=sft_config['SFT_SYSTEM_PROMPT'], user_prompt=sft_config['SFT_USER_PROMPT'])
system_prompt = sft_config['SFT_SYSTEM_PROMPT']
# user_prompt = sft_config['SFT_USER_PROMPT']

# call generate_formatted_response

results = {}

for id, principle in enumerate(principles):
    print(f"Generating SFT dataset for principle id {id}-{principle}: ")
    
    # there should be a list of red team prompts for each principle. 
    prompts = rt_prompts[str(id)]
    if args.debug: print(f"Number of prompts: {len(prompts)}")
    
    for rt_prompt in prompts:

        # messages = gen_prompts.get_message(principle=principle, num_to_gen=sft_config['SFT_NUM_RTP_PER_CALL'])
        user_prompt = sft_config['SFT_USER_PROMPT']
        user_prompt= user_prompt.format(principle=principle, rt_prompt=rt_prompt['adversarial_prompt'], initial_response=rt_prompt['incorrect_model_response_to_adversarial_prompt'])
        if args.debug: print(user_prompt)

        response = generate_formatted_response(client=client, model=args.model, system=system_prompt, user=user_prompt, messages=None, verbose=args.verbose, response_format=SFTDataset)
        
        results[id] = json.loads(response)["sft_data"]
        if args.debug: print(f"Model Response: {results[id]}")

# save results to json file
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f"{args.output_file}_{args.model}_{timestr}.json", 'w') as f:
    json.dump(results, f)


