"""
Processing pipeline of DocLens
"""

import re
import asyncio
import time
from typing import List, Dict, Any
from ast import literal_eval

import numpy as np
from tqdm.asyncio import tqdm


from agents import (
    adjudicator_agent,
    answer_sampler_agent,
    page_navigator_agent,
    vanilla_agent,
)

from .config import ExpConfig
from .eval_toolkits import (
    get_score_for_response_mmlongbenchdoc,
    get_score_for_response_longdocurl,
    get_score_for_response_papertab,
    get_score_for_response,
    calculate_page_level_recall_precision_f1,
)


class DocLensProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: "ExpConfig",
        page_navigator: page_navigator_agent.PageNavigator,
        answer_sampler: answer_sampler_agent.AnswerSampler,
        adjudicator: adjudicator_agent.Adjudicator,
        vanilla_reader: vanilla_agent.VanillaAgent,
    ):
        self.exp_config = exp_config
        self.page_navigator = page_navigator
        self.answer_sampler = answer_sampler
        self.adjudicator = adjudicator
        self.vanilla_reader = vanilla_reader

    async def process_single_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single document
        """
        # Preprocess Question
        question = data["question"].strip()
        preprocessed_question = re.sub(
            r"page\s+(\d+)",
            r"the page with printed page number \1",
            question,
            flags=re.IGNORECASE,
        )
        data["preprocessed_question"] = preprocessed_question

        # Phase 1: Located Relevant Pages
        if self.exp_config.phase_name == "phase1":
            data_with_pred = await self.page_navigator.process(data)
        # Phase 2: Answer Generation
        elif self.exp_config.phase_name == "phase2":
            data_with_pred = await self.answer_sampler.process(data)
            data_with_pred = await self.adjudicator.process(data_with_pred)
        # End-to-End: From Page Navigation to Answer Generation
        elif self.exp_config.phase_name == "end2end":
            total_consumed_token_num = 0

            start_time = time.time()
            data_with_pred, pgnav_consumed_token_num = (
                await self.page_navigator.process(data)
            )
            pgnav_time = time.time() - start_time
            data_with_pred["pgnav_time_sec"] = pgnav_time
            data_with_pred["pgnav_consumed_token_num"] = pgnav_consumed_token_num
            total_consumed_token_num += pgnav_consumed_token_num

            start_time = time.time()
            data_with_pred, ans_sampler_consumed_token_num = (
                await self.answer_sampler.process(data_with_pred)
            )
            ans_sampler_time = time.time() - start_time
            data_with_pred["ans_sampler_time_sec"] = ans_sampler_time
            data_with_pred["ans_sampler_consumed_token_num"] = (
                ans_sampler_consumed_token_num
            )
            total_consumed_token_num += ans_sampler_consumed_token_num

            start_time = time.time()
            data_with_pred, adjudicator_consumed_token_num = (
                await self.adjudicator.process(data_with_pred)
            )
            adjudicator_time = time.time() - start_time
            data_with_pred["adjudicator_time_sec"] = adjudicator_time
            data_with_pred["adjudicator_consumed_token_num"] = (
                adjudicator_consumed_token_num
            )
            total_consumed_token_num += adjudicator_consumed_token_num

            data_with_pred["total_consumed_token_num"] = total_consumed_token_num

        elif self.exp_config.phase_name == "baseline":
            data_with_pred, total_consumed_token_num = (
                await self.vanilla_reader.process(data)
            )
            data_with_pred["total_consumed_token_num"] = total_consumed_token_num

        else:
            print(f"Unknown phase: {self.exp_config.phase_name}")

        # Evaluation
        if "answer" in data:
            data_with_eval = await self.evaluation_function(
                data_with_pred, self.exp_config
            )
            return data_with_eval
        return data_with_pred

    async def process_documents_batch(
        self, data_list: List[Dict[str, Any]], max_concurrent: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Batch process documents with concurrency support
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(doc):
            async with semaphore:
                return await self.process_single_document(doc)

        # Create all tasks
        tasks = []
        for data in data_list:
            task = asyncio.create_task(process_with_semaphore(data))
            tasks.append(task)

        all_result_list = []
        with tqdm(total=len(tasks), desc="Processing concurrently") as pbar:
            # Iterate through completed tasks returned by as_completed
            for future in asyncio.as_completed(tasks):
                result_data = await future
                all_result_list.append(result_data)
                postfix_dict = {}

                if "cand1_score" in result_data:
                    first_cand_avg_score = np.mean(
                        [data.get("cand1_score", 0) for data in all_result_list]
                    )
                    postfix_dict = {"cand1_avg": f"{first_cand_avg_score:.4f}"}

                if self.exp_config.phase2_candidate_num > 1:
                    cand_avg_score_list = [
                        np.mean(
                            [
                                data.get(f"cand{i}_score", 0)
                                for i in range(
                                    1, self.exp_config.phase2_candidate_num + 1
                                )
                            ]
                        )
                        for data in all_result_list
                    ]
                    cand_best_score_list = [
                        max(
                            data.get(f"cand{i}_score", 0)
                            for i in range(1, self.exp_config.phase2_candidate_num + 1)
                        )
                        for data in all_result_list
                    ]
                    postfix_dict["avg_of_n"] = f"{np.mean(cand_avg_score_list):.4f}"
                    postfix_dict["best_of_n"] = f"{np.mean(cand_best_score_list):.4f}"

                if "adjudicator_score" in result_data:
                    adjudicator_avg_score = np.mean(
                        [data.get("adjudicator_score", 0) for data in all_result_list]
                    )
                    postfix_dict["adjudicated"] = f"{adjudicator_avg_score:.4f}"

                if (
                    "pgnav_all_located_pages" in result_data
                    and "evidence_pages" in result_data
                ):
                    located_pages_recall_list = []
                    for data in all_result_list:
                        located_pages = literal_eval(
                            data.get("pgnav_all_located_pages", [])
                        )
                        evidence_pages = literal_eval(data["evidence_pages"])
                        recall_score, _, _ = calculate_page_level_recall_precision_f1(
                            located_pages, evidence_pages
                        )
                        located_pages_recall_list.append(recall_score)
                        data["located_pages_recall"] = recall_score
                    postfix_dict["page_recall"] = (
                        f"{np.mean(located_pages_recall_list):.4f}"
                    )

                pbar.set_postfix(**postfix_dict)
                pbar.update(1)

        return all_result_list

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: "ExpConfig"
    ) -> Dict[str, Any]:
        """
        the evaluation function we use
        """
        prefix_list = [
            f"cand{i}_" for i in range(1, exp_config.phase2_candidate_num + 1)
        ]
        if "adjudicator_prediction" in data:
            prefix_list.append("adjudicator_")

        for prefix in prefix_list:
            if exp_config.dataset_name in ["MMLongBenchDoc"]:
                score, reasoning, extracted_pred = (
                    await get_score_for_response_mmlongbenchdoc(data, prefix)
                )
                data[f"{prefix}extracted_pred"] = extracted_pred
                data[f"{prefix}score"] = score
                data[f"{prefix}score_reasoning"] = reasoning

                # --- Optional: Use llm-as-a-judge to remedy some evaluation errors ---
                # Note: The default evaluation follows the original MMLongBench-Doc protocol,
                # which is strictly rule-based. We've observed that this can incorrectly
                # penalize some semantically correct answers.
                #
                # The following code is provided as an optional remedy.
                # To enable LLM-based re-evaluation for failed cases, uncomment the lines below.

                # if score > 0:
                #     continue
                # # for failed cases, use llm-as-a-judge to remedy semantically correct ones
                # judge_score, judge_reasoning, _ = await llm_reevaluate(data, prefix)
                # data[f"{prefix}score"] = judge_score
                # data[f"{prefix}score_reasoning"] = judge_reasoning

            elif exp_config.dataset_name in ["LongDocURL"]:
                score, reasoning, extracted_pred = (
                    await get_score_for_response_longdocurl(data, prefix)
                )
                data[f"{prefix}extracted_pred"] = extracted_pred
                data[f"{prefix}score"] = score
                data[f"{prefix}score_reasoning"] = reasoning

            elif exp_config.dataset_name in ["PaperTab"]:
                score, reasoning = await get_score_for_response_papertab(data, prefix)
                data[f"{prefix}extracted_pred"] = ""
                data[f"{prefix}score"] = score
                data[f"{prefix}score_reasoning"] = reasoning

            else:
                score, reasoning = await get_score_for_response(data, prefix)
                data[f"{prefix}extracted_pred"] = ""
                data[f"{prefix}score"] = score
                data[f"{prefix}score_reasoning"] = reasoning

        return data
