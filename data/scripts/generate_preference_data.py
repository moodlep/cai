# Generate preference data:
# 1. Iterate over principles and red team prompts
# 2. Ask the model to produce two responses (the user prompt and variance matters here)
#
# Guidelines:
# 1. Generate meaningfully different responses some obvious violations, some near-violations some aligned. Achieved with variance?
# 2. Decide pairwise vs fine-grained ranking of multiple responses.
# 3. Consider samples that address multiple principles.
# 4. Ensure enough positive vs. negative examples. You want both “safe vs. unsafe” and “unsafe vs. extremely unsafe” comparisons.
# 
import json
import os
from datetime import datetime

import openai
from openai import OpenAI
import json

from data_utils import get_principles_from_constitution, generate_formatted_response, get_config, get_red_team_prompts
from structured_response_formats import PreferenceDataset
import dotenv
import argparse

# Configure and parse args
parser = argparse.ArgumentParser(description='Generate Preference Dataset')
parser.add_argument('--config', type=str, default='/home/mariak/cai/data/config/sa_config.yaml', help='path to the config file')
parser.add_argument('--config-category', type=str, default='pref', help='type of data generation: red_team, sft, pref')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model to use for generation: gpt-4o-mini, gpt-4o')
parser.add_argument('--verbose', type=str, default=True, help='verbosity setting for openAI model: True/False')
parser.add_argument('--output-file', type=str, default="/home/mariak/cai/data/datasets/sa/pref_dataset", help='output file base name')
parser.add_argument('--few-shot-samples', type=str, default=None, help='include few shot examples')
parser.add_argument('--src-red-team-prompts', type=str, default='/home/mariak/cai/data/datasets/sa/red_team_prompts_20250127-113940.json', help='latest red team prompts file')
parser.add_argument('--debug', type=str, default=False, help='debug')
args = parser.parse_args()


# Configure your OpenAI API key
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()

# Load config
config = get_config(args.config)
pref_config = config[args.config_category]

# Load principles, prompts, system prompt
principles = get_principles_from_constitution(config['src_principles'])
rt_prompts = get_red_team_prompts(args.src_red_team_prompts)
system_prompt = pref_config['PREF_SYSTEM_PROMPT']

# Few shot examples
if args.few_shot_samples:
    with open(args.few_shot_samples) as f:
        few_shot_samples = json.load(f)
        few_shot_samples = few_shot_samples['few-shot']
        if args.debug: print(few_shot_samples)


results = {}
for principle_id, principle in enumerate(principles):
    print(f"Generating preference dataset for principle id {principle_id}-{principle}: ")
    
    prompts = rt_prompts[str(principle_id)]
    if args.debug: print(f"Number of prompts: {len(prompts)}")

    if (principle_id == 2): break
    
    for rt_prompt in prompts:
        if args.few_shot_samples:
            user_prompt = pref_config['PREF_USER_PROMPT_FEW_SHOT']
            user_prompt = user_prompt.format(
                principle=principle, 
                rt_prompt=rt_prompt['adversarial_prompt'], 
                few_shot_samples=few_shot_samples,
                num_pref_per_call=pref_config['NUM_PREF_PER_CALL']
            )
        else:
            user_prompt = pref_config['PREF_USER_PROMPT']
            user_prompt= user_prompt.format(
                principle=principle, 
                rt_prompt=rt_prompt['adversarial_prompt'],
                num_pref_per_call=pref_config['NUM_PREF_PER_CALL']
            )
        if args.debug: print(user_prompt)

        response = generate_formatted_response(
            client=client, 
            model=args.model, 
            system=system_prompt, 
            user=user_prompt,
            messages=None,
            verbose=args.verbose, 
            response_format=PreferenceDataset
        )
        print(response)
        results[principle_id] = json.loads(response)["preference_data"]
        if args.debug: print(f"Model Response: {results[principle_id]}")

# Save results to json file
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f"{args.output_file}_{args.model}_{timestr}.json", 'w') as f:
    json.dump(results, f)



