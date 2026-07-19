"""
Base class for agents
"""

import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import json_repair
from google.genai import types

from utils import generation_utils
from utils.config import ExpConfig


class BaseAgent(ABC):
    """Base class for agents"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        system_prompt: str = "",
        exp_config: "ExpConfig" = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.exp_config = exp_config

    def _parse_response(
        self, response: str, key_list: List[str], prefix: str
    ) -> Dict[str, Any]:
        """
        Parse the response from the model into a structured format.
        """

        def extract_final_answer(response: str) -> str:
            pattern = (
                r"(?i)(?:prediction:|(?:\*{2}prediction\*{2}:)|the final answer is)"
            )
            parts = re.split(pattern, response)
            if len(parts) > 1:
                return parts[-1].strip()
            else:
                return response.strip()

        try:
            response_dict = json_repair.loads(response)
        except Exception as e:
            print(f"JSON repair failed: {e}")
            response_dict = {}
        if not isinstance(response_dict, dict):
            response_dict = {}
        result_dict = {f"{prefix}_response": response}
        for key in key_list:
            place_holder = "" if "located_pages" not in key else "[]"
            content = str(response_dict.get(key, place_holder))
            # for gemini-2.5-flash, it does not follow the instruction so well, we need to do some post-processing
            if key == "prediction" and content.strip() == "":
                content = extract_final_answer(response)
            result_dict[f"{prefix}_{key}"] = content
        return result_dict

    async def call_llm_with_retry_async(
        self,
        content_list: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        candidate_num: int,
    ) -> List[str]:
        """
        Call the LLM with retry mechanism.
        """
        if self.model_name.startswith("gemini"):
            response_text_list, total_consumed_token_num = (
                await generation_utils.call_gemini_with_retry_async(
                    model_name=self.model_name,
                    contents=content_list,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        candidate_count=candidate_num,
                        max_output_tokens=10000,
                    ),
                    max_attempts=5,
                    retry_delay=30,
                )
            )
        elif self.model_name.startswith("claude"):
            config = {
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_output_tokens": 10000,
            }
            response_text_list = await generation_utils.call_claude_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=config,
                max_attempts=5,
                retry_delay=30,
            )
        elif self.model_name.startswith("qwen"):
            config = {
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_output_tokens": 10000,
            }

            response_text_list, total_consumed_token_num = (
                await generation_utils.call_qwen_with_retry_async(
                    model_name=self.model_name,
                    contents=content_list,
                    config=config,
                    max_attempts=5,
                    retry_delay=30,
                )
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return response_text_list, total_consumed_token_num

    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return the result"""
