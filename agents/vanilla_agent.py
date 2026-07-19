"""
Vanilla Agent - Aggregate multiple candidate answers to generate final response
"""

from typing import Dict, Any

from utils import generation_utils
from .base_agent import BaseAgent


class VanillaAgent(BaseAgent):
    """Vanilla Agent - Aggregate multiple candidate answers to generate final response"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate candidate answers to generate final response
        """
        question = data.get("preprocessed_question", data.get("question", ""))
        user_prompt = f"""\n\n**Question**: {question}\n\n**Your Output**:"""
        content_list = await generation_utils.get_doc_content_list_async(
            data=data,
            input_pages="all",
            input_mode="use_ocr",
            exp_config=self.exp_config,
        )
        content_list.append({"type": "text", "text": user_prompt})

        response_text_list, total_consumed_token_num = (
            await self.call_llm_with_retry_async(
                content_list=content_list,
                system_prompt=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_num=1,
            )
        )
        key_list = ["analysis", "prediction"]
        response_dict = self._parse_response(response_text_list[0], key_list, "cand1")
        data.update(response_dict)

        return data, total_consumed_token_num
