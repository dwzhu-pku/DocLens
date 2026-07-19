"""
Adjudicator Agent - Aggregate multiple candidate answers to generate final response
"""

from typing import Dict, Any

from .base_agent import BaseAgent


class Adjudicator(BaseAgent):
    """Adjudicator Agent - Aggregate multiple candidate answers to generate final response"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate candidate answers to generate final response
        """
        question = data.get("preprocessed_question", data.get("question", ""))

        agent_answers = ""
        for idx in range(1, self.exp_config.phase2_candidate_num + 1):
            cand_idx_analysis = data.get(f"cand{idx}_analysis", "")
            cand_idx_prediction = data.get(f"cand{idx}_prediction", "")
            agent_answers += f"Agent {idx}\nAnalysis: {cand_idx_analysis}\nAnswer: {cand_idx_prediction}\n"
        user_prompt = f"""\n\n**Question**: {question}\n\n**List of Agent Analyses and Answers**:\n {agent_answers}\n\n**Your Output**:"""
        content_list = [{"type": "text", "text": user_prompt}]

        response_text_list, total_consumed_token_num = (
            await self.call_llm_with_retry_async(
                content_list=content_list,
                system_prompt=self.system_prompt,
                temperature=0,
                candidate_num=1,
            )
        )
        key_list = ["analysis", "prediction"]
        response_dict = self._parse_response(
            response_text_list[0], key_list, "adjudicator"
        )
        data.update(response_dict)

        return data, total_consumed_token_num
