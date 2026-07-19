"""
AnswerSampler Agent
- Perform detailed analysis on retrieved evidence, generate several candidate answers
"""

from typing import Dict, Any

from utils import generation_utils
from .base_agent import BaseAgent


class AnswerSampler(BaseAgent):
    """
    AnswerSampler Agent
    - Perform detailed analysis on retrieved evidence, generate several candidate answers
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed analysis on retrieved evidence and generate multiple candidate answers
        """
        question = data.get("preprocessed_question", data.get("question", ""))
        user_prompt = f"""\n\n**Question**: {question}\n\n**Your Output**:"""
        content_list = await generation_utils.get_doc_content_list_async(
            data=data,
            input_pages=self.exp_config.phase2_input_pages,
            input_mode=self.exp_config.phase2_input_mode,
            exp_config=self.exp_config,
        )
        content_list.append({"type": "text", "text": user_prompt})

        response_text_list, total_consumed_token_num = (
            await self.call_llm_with_retry_async(
                content_list=content_list,
                system_prompt=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_num=self.exp_config.phase2_candidate_num,
            )
        )
        for i in range(self.exp_config.phase2_candidate_num):
            key_list = ["analysis", "prediction"]
            response_dict = self._parse_response(
                response_text_list[i], key_list, f"cand{i+1}"
            )
            data.update(response_dict)

        return data, total_consumed_token_num
