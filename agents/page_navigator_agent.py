"""
Page Navigator Agent - Locate relevant pages from long documents
"""

import ast
import re
from typing import List, Dict, Any

from utils import generation_utils
from .base_agent import BaseAgent


class PageNavigator(BaseAgent):
    """Page Navigator - Locate relevant pages from long documents"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_with_chunking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the document in chunks to handle long documents with many pages.
        This method splits the document into smaller chunks, processes each chunk,
        and then merges the results.
        """
        question = data.get("preprocessed_question", data.get("question", ""))
        user_prompt = f""""\n\n**Question**: {question}\n\n**Your Output**:"""

        content_list = await generation_utils.get_doc_content_list_async(
            data=data,
            input_pages="all",  # For page navigator, we always use all pages as input
            input_mode=self.exp_config.phase1_input_mode,
            exp_config=self.exp_config,
        )
        content_list.append({"type": "text", "text": user_prompt})

        chunk_size = 50
        chunk_results, prompt_parts, current_chunk, chunks = [], [], [], []
        in_pages = False

        for part in content_list[:-1]:  # exclude the user prompt part
            # Check if it is a page marker
            if part["type"] == "text" and "Screenshot of page" in part["text"]:
                try:
                    page_num = int(part["text"].split("page")[1].split()[0])
                    # If it is the first page, it means the previous ones are all prompts
                    if page_num == 1:
                        in_pages = True
                        current_chunk.append(part)
                    # If it crosses the batch boundary (1->51, 51->101, etc.)
                    elif page_num % chunk_size == 1 and page_num > 1:
                        chunks.append(current_chunk)
                        current_chunk = [part]
                    else:
                        current_chunk.append(part)
                except Exception:
                    if in_pages:
                        current_chunk.append(part)
                    else:
                        prompt_parts.append(part)
            else:
                if in_pages:
                    current_chunk.append(part)
                else:
                    prompt_parts.append(part)

        if current_chunk:
            chunks.append(current_chunk)

        # Process each chunk
        user_prompt_part = content_list[-1]
        for chunk_pages in chunks:
            chunk_content_list = prompt_parts + chunk_pages + [user_prompt_part]

            # Call LLM to process the current chunk
            response_text_list = await self.call_llm_with_retry_async(
                content_list=chunk_content_list,
                system_prompt=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_num=self.exp_config.phase1_candidate_num,
            )

            # Parse the results of the current batch
            for i in range(self.exp_config.phase1_candidate_num):
                key_list = ["analysis", "located_pages", "prediction"]
                response_dict = self._parse_response(
                    response_text_list[i], key_list, f"pgnav_cand{i+1}"
                )
                chunk_results.append(response_dict)

        # Merge all chunk results
        merged_results = self._merge_chunk_results(
            chunk_results, self.exp_config.phase1_candidate_num
        )

        # Update data
        for candidate_result in merged_results:
            data.update(candidate_result)

        return data

    def _merge_chunk_results(
        self, chunk_results: List[Dict], candidate_num: int
    ) -> List[Dict]:
        """
        Merge multiple chunk results.
        Keep only the analysis and prediction from the first chunk, and merge all located_pages.
        """
        # Calculate the number of chunks for each candidate based on the candidate count
        num_chunks = len(chunk_results) // candidate_num
        merged = []

        for cand_idx in range(candidate_num):
            cand_key = f"pgnav_cand{cand_idx + 1}"
            all_located_pages = []
            # Collect all located_pages for this candidate across all chunks
            for chunk_idx in range(num_chunks):
                result_idx = chunk_idx * candidate_num + cand_idx
                chunk_result = chunk_results[result_idx]

                located_key = f"{cand_key}_located_pages"
                if located_key in chunk_result:
                    try:
                        located_list = ast.literal_eval(chunk_result[located_key])
                        all_located_pages.extend(located_list)
                    except Exception:
                        pass

            # Get the result of the first chunk as the base
            first_chunk_result = chunk_results[cand_idx].copy()

            # Only update located_pages with the merged result
            if all_located_pages:
                unique_pages = sorted(list(set(all_located_pages)))
                first_chunk_result[f"{cand_key}_located_pages"] = str(unique_pages)

            merged.append(first_chunk_result)

        return merged

    def union_located_pages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the union of all located pages from different candidates
        """
        located_pages = []
        for i in range(self.exp_config.phase1_candidate_num):
            cand_located_pages = str(data.get(f"pgnav_cand{i+1}_located_pages", ""))
            numbers_as_strings = re.findall(r"\d+", cand_located_pages)
            new_located_pages = [int(num) for num in numbers_as_strings]
            located_pages.extend(new_located_pages)
        located_pages = sorted(list(set(located_pages)))
        data["pgnav_all_located_pages"] = str(located_pages)
        return data

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed analysis on retrieved pages and generate multiple candidate answers
        """

        if "claude" in self.model_name:
            # For models with context window <= 128k, use chunking
            data_with_pred = await self.process_with_chunking(data)
            data_with_all_located_pages = self.union_located_pages(data_with_pred)
            return data_with_all_located_pages

        question = data.get("preprocessed_question", data.get("question", ""))
        user_prompt = f"""\n\n**Question**: {question}\n\n**Your Output**:"""
        content_list = await generation_utils.get_doc_content_list_async(
            data=data,
            input_pages="all",  # For page navigator, we always use all pages as input
            input_mode=self.exp_config.phase1_input_mode,
            exp_config=self.exp_config,
        )
        content_list.append({"type": "text", "text": user_prompt})

        response_text_list, total_consumed_token_num = (
            await self.call_llm_with_retry_async(
                content_list=content_list,
                system_prompt=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_num=self.exp_config.phase1_candidate_num,
            )
        )
        for i in range(self.exp_config.phase1_candidate_num):
            key_list = ["analysis", "located_pages", "prediction"]
            response_dict = self._parse_response(
                response_text_list[i], key_list, f"pgnav_cand{i+1}"
            )
            data.update(response_dict)
        data_with_all_located_pages = self.union_located_pages(data)

        return data_with_all_located_pages, total_consumed_token_num
