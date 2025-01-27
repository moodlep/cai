from pydantic import BaseModel
import types
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple, ClassVar



class RTResponse(BaseModel):
  # - adversarial_prompt: str
  # - `answer_matching_behavior`: `(List[str])` The answer that reflects or matches the target property (i.e. the "correct" answer label).
  # - `answer_not_matching_behavior`: `(List[str])` The answer(s) that do(es) not reflect the target property (i.e. the "incorrect" answer label).
  # - REMOVED `principle`: `(str)` Principle targetted

  adversarial_prompt: str
  correct_model_response_to_adversarial_prompt: str
  incorrect_model_response_to_adversarial_prompt: str
  # TODO: how do we control the number of list items if we want to use it?
  # answer_matching_behavior: List[str]
  # answer_not_matching_behavior: List[str]
  # principle: str

class RedTeamPrompts(BaseModel):
  rt_responses: List[RTResponse]


class SFTData(BaseModel):
  # Aligned with the HuggingFace cai-conversation-harmless dataset here: https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless
  # - `initial_prompt`: `(str)` The initial prompt that the user sees.
  # - `initial_response`: `(str)` The initial response that the user sees.
  # - `critique_prompt`: `(str)` The prompt that incorporates the relevant principle and requests a critique from the model
  # - `critique_response`: `(str)` The response that the model provides to the critic prompt
  # - `revision_prompt`: `(str)` The prompt that incorporates the relevant principle and requests a revision from the model
  # - `revision_response`: `(str)` The response that the model provides to the revision prompt
  # - `prompt`: `(str)` Same as the initial prompt
  # - `messages`: `(List[str])` TBD
  # - `chosen`: `(List[str])` The revised and correct response
  # - `rejected`: `(List[str])` The incorrect response
  # - `principle`: `(str)` The principle that the model is being evaluated against
  initial_prompt: str
  initial_response: str
  critique_prompt: str
  critique_response: str
  revision_prompt: str
  revision_response: str
  # Let's limit output tokens!
  # prompt: str
  # messages: List[str]
  # chosen: List[str]
  # rejected: List[str]
  # principle: str
  
class SFTDataset(BaseModel):
  sft_data: List[SFTData]
  
class Principle(BaseModel):
  principle: str
  critique: str
  revision: str
  id: int
  category: str
  
class Principles(BaseModel):
  principles: List[Principle]