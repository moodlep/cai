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

