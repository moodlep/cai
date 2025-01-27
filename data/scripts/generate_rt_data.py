import json
import os
from datetime import datetime

import openai
from openai import OpenAI
import json

from data_utils import get_principles_from_constitution, generate_formatted_response, get_config, GenPrompts
from structured_response_formats import RTResponse, RedTeamPrompts
import dotenv
import argparse

# add args
parser = argparse.ArgumentParser(description='Generate red team prompts')
parser.add_argument('--config', type=str, default='data/config/sa_config.yaml', help='path to the config file')
parser.add_argument('--config-category', type=str, default='red_team', help='type of data generation: red_team, sft, dpo')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model to use for generation: gpt-4o-mini, gpt-4o')
parser.add_argument('--verbose', type=str, default=True, help='verbosity setting for openAI model: True/False')
parser.add_argument('--output-file', type=str, default="data/datasets/red_team_prompts", help='output file name')
parser.add_argument('--include-few-shot', type=str, default=False, help='include few shot examples')

args = parser.parse_args()


# Configure your OpenAI API key
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()

# Load config
config = get_config(args.config)
rt_config = config[args.config_category]

# Load principles
principles = get_principles_from_constitution(config['src_principles'])

# Run to see the questions that the model generated
gen_prompts = GenPrompts(system_prompt=rt_config['RT_SYSTEM_PROMPT'], user_prompt=rt_config['RT_USER_PROMPT'])

# call generate_formatted_response

results = {}

for id, principle in enumerate(principles):
  print(f"Generating red-teaming prompts for principle id {id}-{principle}: ")
  
  messages = gen_prompts.get_message(principle=principle, num_to_gen=rt_config['RT_NUM_RTP_PER_CALL'])
  
  response = generate_formatted_response(client=client, model=args.model, messages=messages, verbose=args.verbose, response_format=RedTeamPrompts)
  print("MODEL RESPONSES:\n")
  results[id] = json.loads(response)["rt_responses"]
  print(results[id])

# save results to json file
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f"{args.output_file}_{timestr}.json", 'w') as f:
    json.dump(results, f)


