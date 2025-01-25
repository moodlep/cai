import json
import os
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path

import openai
from openai import OpenAI
import json

from data_utils import get_principles_from_constitution, generate_formatted_response, get_config, GenPrompts
from structured_response_formats import RTResponse, RedTeamPrompts
import dotenv

# Configure your OpenAI API key
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()

# Load config
config = get_config('data/config/sa_config.yaml')
rt_config = config['red_team']
sft_config = config['sft']

# Load principles
sac_principles = get_principles_from_constitution(rt_config['sa_principles'])

# Run to see the questions that the model generated
gen_prompts = GenPrompts(system_prompt=rt_config['RT_SYSTEM_PROMPT'], user_prompt=rt_config['RT_USER_PROMPT'])

# call generate_formatted_response

results = {}

for id, principle in enumerate(sac_principles):
  print(f"Generating red-teaming prompts for principle id {id}-{principle}: ")
  
  messages = gen_prompts.get_message(principle=principle, num_to_gen=rt_config['RT_NUM_RTP_PER_CALL'])
  
  response = generate_formatted_response(client=client, model=config['model'], messages=messages, verbose=True, response_format=RedTeamPrompts)
  print("MODEL RESPONSES:\n")
  results[id] = json.loads(response)["rt_responses"]
  print(results[id])

# save results to json file
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f"data/datasets/red_team_prompts_{timestr}.json", 'w') as f:
    json.dump(results, f)


