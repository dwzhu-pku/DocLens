"""
Main script to run the document processing with agents
"""

import asyncio
import json
import argparse
from pathlib import Path
import aiofiles

from agents import (
    adjudicator_agent,
    answer_sampler_agent,
    page_navigator_agent,
    vanilla_agent,
)
from prompts.agent_prompts import (
    PAGE_NAVIGATOR_SYSTEM_PROMPT,
    ANSWER_SAMPLER_SYSTEM_PROMPT,
    ANSWER_SAMPLER_ALL_ANSWERABLE_SYSTEM_PROMPT,
    ADJUDICATOR_SYSTEM_PROMPT,
    VANILLA_READER_SYSTEM_PROMPT,
    ADJUDICATOR_ALL_ANSWERABLE_SYSTEM_PROMPT,
)
from utils import doclens_processor, generation_utils, config


async def main():
    """Main function example"""
    # add command line args
    parser = argparse.ArgumentParser(description="DocAgent processing script")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MMLongBenchDoc",
        help="name of the dataset to use (default: MMLongBenchDoc, FinRAGBench-V, LongDocURL, PaperTab)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="samples",
        help="split of the dataset to use (default: samples)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-pro",
        help="name of the model to use (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="dev",
        help="name of the experiment to use (default: dev)",
    )

    parser.add_argument(
        "--phase_name",
        type=str,
        default="end2end",
        help="name of the phase to use (default: end2end)",
    )

    parser.add_argument(
        "--phase1_input_pages",
        type=str,
        default="all",
        help="",
    )
    parser.add_argument(
        "--phase1_input_mode",
        type=str,
        default="use_ocr",
        help="",
    )
    parser.add_argument(
        "--phase1_candidate_num",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--phase2_input_pages",
        type=str,
        default="pgnav_all_located_pages",
        help="",
    )
    parser.add_argument(
        "--phase2_input_mode",
        type=str,
        default="use_element_localizer",
        help="",
    )
    parser.add_argument(
        "--phase2_candidate_num",
        type=int,
        default=1,
        help="",
    )
    args = parser.parse_args()

    exp_config = config.ExpConfig(
        phase1_input_pages=args.phase1_input_pages,
        phase1_candidate_num=args.phase1_candidate_num,
        phase1_input_mode=args.phase1_input_mode,
        phase2_input_pages=args.phase2_input_pages,
        phase2_input_mode=args.phase2_input_mode,
        phase2_candidate_num=args.phase2_candidate_num,
        phase_name=args.phase_name,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        model_name=args.model_name,
        exp_name=args.exp_name,
        work_dir=Path(__file__).parent,
    )
    print(exp_config.exp_name)

    input_filename = (
        Path(__file__).parent
        / "data"
        / exp_config.dataset_name
        / f"{exp_config.split_name}.json"
    )
    output_filename = exp_config.result_dir / f"{exp_config.exp_name}.json"
    print(f"Input file: {input_filename}", f"Output file: {output_filename}")
    with open(input_filename, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Create agents
    # For some datasets, we need to consider the unanswerable questions
    dataset_list_including_not_answerable = ["MMLongBenchDoc"]
    if exp_config.dataset_name in dataset_list_including_not_answerable:
        print(
            "For datasets including unanswerable questions, we apply some additional rules to reduce hallucination."
        )
        page_navigator_system_prompt = PAGE_NAVIGATOR_SYSTEM_PROMPT
        answer_sampler_system_prompt = ANSWER_SAMPLER_SYSTEM_PROMPT
        adjudicator_system_prompt = ADJUDICATOR_SYSTEM_PROMPT
    else:
        page_navigator_system_prompt = PAGE_NAVIGATOR_SYSTEM_PROMPT
        answer_sampler_system_prompt = ANSWER_SAMPLER_ALL_ANSWERABLE_SYSTEM_PROMPT
        adjudicator_system_prompt = ADJUDICATOR_ALL_ANSWERABLE_SYSTEM_PROMPT

    page_navigator = page_navigator_agent.PageNavigator(
        # model_name=exp_config.model_name,
        model_name="gemini-2.5-flash-lite",
        system_prompt=page_navigator_system_prompt,
        exp_config=exp_config,
    )
    answer_sampler = answer_sampler_agent.AnswerSampler(
        model_name=exp_config.model_name,
        system_prompt=answer_sampler_system_prompt,
        exp_config=exp_config,
    )
    adjudicator = adjudicator_agent.Adjudicator(
        model_name=exp_config.model_name,
        system_prompt=adjudicator_system_prompt,
        exp_config=exp_config,
    )
    vanilla_reader = vanilla_agent.VanillaAgent(
        model_name=exp_config.model_name,
        system_prompt=VANILLA_READER_SYSTEM_PROMPT,
        exp_config=exp_config,
    )

    # Create processor
    processor = doclens_processor.DocLensProcessor(
        exp_config=exp_config,
        page_navigator=page_navigator,
        answer_sampler=answer_sampler,
        adjudicator=adjudicator,
        vanilla_reader=vanilla_reader,
    )
    # Batch process documents
    concurrent_num = 50
    if "flash" in exp_config.model_name:
        concurrent_num = 100
    elif "claude" in exp_config.model_name:
        concurrent_num = 10
    elif "qwen" in exp_config.model_name:
        concurrent_num = 10
    print(f"Using max concurrency: {concurrent_num}")
    all_result_list = await processor.process_documents_batch(
        data_list, max_concurrent=concurrent_num
    )
    print(f"Saving results to {output_filename}")
    async with aiofiles.open(
        output_filename, "w", encoding="utf-8", errors="surrogateescape"
    ) as f:
        json_string = json.dumps(all_result_list, ensure_ascii=False, indent=4)
        await f.write(json_string)

    if exp_config.phase_name == "phase1":
        print("Writing located pages to input file for phase 2...")
        generation_utils.write_located_pages_to_input_file(
            all_result_list, input_filename
        )


if __name__ == "__main__":
    asyncio.run(main())
