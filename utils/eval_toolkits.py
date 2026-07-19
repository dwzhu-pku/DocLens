"""
Evaluation toolkits for DocLens
"""

import re
import math
from ast import literal_eval
from typing import List, Tuple

import json_repair
from google.genai import types

from prompts.eval_prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_PAPERTAB,
    EXTRACTION_SYSTEM_PROMPT_MMLONG,
    EXTRACTION_SYSTEM_PROMPT_LONGDOCURL,
    REEVALUATE_SYSTEM_PROMPT,
)
from utils.generation_utils import call_gemini_with_retry_async


def calculate_page_level_recall_precision_f1(
    predicted_pages: List[int], ground_truth_pages: List[int]
) -> Tuple[float, float, float]:
    """
    Calculate page-level recall, precision, and F1 score.
    """
    if not ground_truth_pages:
        # if ground truth is []
        return 1.0, 0.0, 0.0

    predicted_set = set(predicted_pages)
    ground_truth_set = set(ground_truth_pages)

    true_positives = len(predicted_set.intersection(ground_truth_set))
    false_positives = len(predicted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - predicted_set)

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return recall, precision, f1_score


### Helper functions from MMLongBench-Doc evaluation code Begins ###


def isfloat(num):
    """
    Check if a string can be converted to a float.
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    """
    Compute the ANLS (Average Normalized Levenshtein Similarity) score between groundtruth and prediction.
    """
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def get_clean_string(s):
    """Cleans the input string by removing unwanted characters and formatting."""
    s = str(s).lower().strip()
    s = s.replace(",", "")
    if s.endswith("kg"):
        s = s.rstrip("kg").strip()
    if s.endswith("mm"):
        s = s.rstrip("mm").strip()
    if s.endswith("m"):
        s = s.rstrip("m").strip()
    if s.endswith("meters"):
        s = s.rstrip("meters").strip()
    if s.endswith("acres"):
        s = s.rstrip("acres").strip()
    if s.endswith("minutes"):
        s = s.rstrip("minutes").strip()
    if s.endswith("mile"):
        s = s.rstrip("mile").strip()
    if s.endswith("miles"):
        s = s.rstrip("miles").strip()
    if s.endswith("million"):
        s = s.rstrip("million").strip()
    if s.endswith("thousand"):
        s = s.rstrip("thousand").strip()
    if s.endswith("billion"):
        s = s.rstrip("billion").strip()
    # remove parenthesis
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().lstrip("£").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_float_equal(
    reference, prediction, include_percentage: bool = False, is_close: float = False
) -> bool:
    """
    Check if two float numbers are equal within a certain precision.
    If include_percentage is True, consider the reference in percentage format as well.
    If is_close is True, use math.isclose for comparison with a relative tolerance of 1%.
    """

    def get_precision(gt_ans: float) -> int:
        precision = 3
        if "." in str(gt_ans):
            precision = len(str(gt_ans).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if math.isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def is_exact_match(s):
    """
    Check if the string is an exact match (e.g., contains URLs, code files, phone numbers, dates, emails).
    """
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):
        flag = True
    # Email address
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):
        flag = True
    return flag


def rule_based_eval_score_mmlongbenchdoc(gt, pred, answer_type):
    """
    Rule-based evaluation function for MMLongBench-Doc
    """
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except Exception:
            pred = ""
        score = gt == pred
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except Exception:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = gt == pred
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            gt = literal_eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            try:
                pred = literal_eval(pred)
            except Exception as e:
                print(e, pred)
                pred = []
        if not isinstance(pred, list):
            pred = [pred]
        # print(len(gt), len(pred))
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            # print(gt, pred)

            if gt == [] and pred == []:
                return 1.0

            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = "-".join(gt) == "-".join(pred)
            else:
                score = min(
                    [anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)]
                )

    return float(score)


def rule_based_eval_score_longdocurl(gt, pred, answer_type):
    """
    Rule-based evaluation function for LongDocURL
    """
    if answer_type == "Integer":
        try:
            gt = get_clean_string(str(gt))
            if (
                len(re.findall(r"\d+,\s*\d+", gt, re.DOTALL)) > 0
            ):  # deal with Integer value formatted as "96,395"
                gt = "".join([_.strip() for _ in gt.split(",")])
            gt = int(gt)
        except:
            gt = gt
        try:
            pred = get_clean_string(str(pred))
            if (
                len(re.findall(r"\d+,\s*\d+", pred, re.DOTALL)) > 0
            ):  # deal with Integer value formatted as "96,395"
                pred = "".join([_.strip() for _ in pred.split(",")])
            pred = int(pred)
        except:
            pred = ""
        score = gt == pred
    elif answer_type == "Float":
        gt = get_clean_string(str(gt))
        pred = get_clean_string(str(pred))

        if (
            len(re.findall(r"\d+,\s*\d+", gt, re.DOTALL)) > 0
        ):  # deal with Integer value formatted as "96,395"
            gt = "".join([_.strip() for _ in gt.split(",")])
        try:
            gt = float(gt)
        except:
            gt = gt

        if (
            len(re.findall(r"\d+,\s*\d+", pred, re.DOTALL)) > 0
        ):  # deal with Integer value formatted as "96,395"
            pred = "".join([_.strip() for _ in pred.split(",")])
        try:
            pred = float(pred)
        except:
            pred = str(pred)

        try:
            score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
        except:
            score = 0

    elif answer_type in ["String", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = gt == pred
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            try:
                gt = eval(gt)
            except:
                gt = gt
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            try:
                pred = eval(pred)
            except:
                pred = pred
        if not isinstance(pred, list):
            pred = [pred]
        if len(pred) == 0:
            pred = [""]
        if isinstance(gt[0], dict):
            gt = ["-".join([str(value) for key, value in _.items()]) for _ in gt]
        if isinstance(pred[0], dict):
            pred = ["-".join([str(value) for key, value in _.items()]) for _ in pred]

        # print(len(gt), len(pred))
        # print(gt, pred)
        def cal_score_v3(gt, pred):
            gt = [get_clean_string(a) for a in gt]
            pred = [get_clean_string(a) for a in pred]
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score_v3 = "-".join(gt) == "-".join(pred)
            else:
                greedy_scores = [
                    max([anls_compute(str(gt_v), str(pred_v)) for pred_v in pred])
                    for gt_v in gt
                ]
                score_v3 = (
                    sum(greedy_scores) / len(gt) * min(1, len(gt) / len(pred)) ** 0.5
                )
            return score_v3

        score_v3 = cal_score_v3(gt, pred)

    score_v3 = (
        score if answer_type in ["Integer", "Float", "String", "None"] else score_v3
    )

    return float(score_v3)


### Helper functions from MMLongBench-Doc evaluation code Ends ###


async def get_score_for_response(sample_data: dict, cand_prefix: str) -> float:
    """
    Directly using llm-as-a-judge
    """
    question, ground_truth, prediction = (
        sample_data["question"],
        sample_data["answer"],
        sample_data[f"{cand_prefix}prediction"],
    )
    user_prompt = f"\n\nQuestion: {question}\nGround Truth: {ground_truth}\nPrediction: {prediction}\n\nYour Output:"
    response_text_list, _ = await call_gemini_with_retry_async(
        model_name="gemini-2.5-flash",
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            temperature=0,
            candidate_count=1,
            max_output_tokens=10000,
        ),
    )
    cleaned_response = (
        response_text_list[0].replace("```json", "").replace("```", "").strip()
    )
    try:
        eval_result = json_repair.loads(cleaned_response)
        if not isinstance(eval_result, dict):
            eval_result = {}
    except Exception as e:
        eval_result = {}
        print(e, cleaned_response)

    score = eval_result.get("score", 0.0)
    # make sure between 0 and 1
    score = min(max(score, 0.0), 1.0)
    score_reasoning = eval_result.get("reasoning", cleaned_response)

    return score, score_reasoning


async def get_score_for_response_papertab(sample_data: dict, cand_prefix: str) -> float:
    """
    Directly using llm-as-a-judge
    """
    question, ground_truth_1, ground_truth_2, prediction = (
        sample_data["question"],
        sample_data["answer"],
        sample_data["answer_2"],
        sample_data[f"{cand_prefix}prediction"],
    )
    user_prompt = f"\n\nQuestion: {question}\nGround Truth 1: {ground_truth_1}\nGround Truth 2: {ground_truth_2}\nPrediction: {prediction}\n\nYour Output:"
    response_text_list, _ = await call_gemini_with_retry_async(
        model_name="gemini-2.5-flash",
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT_PAPERTAB,
            temperature=0,
            candidate_count=1,
            max_output_tokens=10000,
        ),
    )
    cleaned_response = (
        response_text_list[0].replace("```json", "").replace("```", "").strip()
    )
    try:
        eval_result = json_repair.loads(cleaned_response)
        if not isinstance(eval_result, dict):
            eval_result = {}
    except Exception as e:
        eval_result = {}
        print(e, cleaned_response)

    score = eval_result.get("score", 0.0)
    # make sure between 0 and 1
    score = min(max(score, 0.0), 1.0)
    score_reasoning = eval_result.get("reasoning", cleaned_response)

    return score, score_reasoning


async def get_score_for_response_mmlongbenchdoc(
    sample_data: dict, cand_prefix: str
) -> float:
    """
    The original evaluation pipeline used in MMLongBench-Doc
    First parse the concise answer from model response, then use heuristic rules for scoring.
    """

    question, ground_truth, prediction = (
        sample_data["question"],
        sample_data["answer"],
        sample_data[f"{cand_prefix}prediction"],
    )

    user_prompt = f"\n\nQuestion:{question}\nAnalysis:{prediction}\n"
    response_text_list, _ = await call_gemini_with_retry_async(
        model_name="gemini-2.5-flash",
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT_MMLONG,
            temperature=0,
            candidate_count=1,
            max_output_tokens=10000,
        ),
    )
    cleaned_response = (
        response_text_list[0].replace("```json", "").replace("```", "").strip()
    )
    pred_ans = (
        cleaned_response.split("Answer format:")[0]
        .split("Extracted answer:")[1]
        .strip()
    )
    score = rule_based_eval_score_mmlongbenchdoc(
        ground_truth, pred_ans, sample_data["answer_format"]
    )
    score_reasoning = ""

    return score, score_reasoning, pred_ans


async def get_score_for_response_longdocurl(
    sample_data: dict, cand_prefix: str
) -> float:
    """
    The original evaluation pipeline used in LongDocURL
    First parse the concise answer from model response, then use heuristic rules for scoring.
    """

    question, ground_truth, prediction = (
        sample_data["question"],
        sample_data["answer"],
        sample_data[f"{cand_prefix}prediction"],
    )

    user_prompt = f"\n\nQuestion:{question}\nAnalysis:{prediction}\n"
    response_text_list = await call_gemini_with_retry_async(
        model_name="gemini-2.5-flash",
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT_LONGDOCURL,
            temperature=0,
            candidate_count=1,
            max_output_tokens=10000,
        ),
    )
    try:
        cleaned_response = (
            str(response_text_list[0]).replace("```json", "").replace("```", "").strip()
        )
    except Exception as e:
        print(e, response_text_list[0])
    # pred_ans = cleaned_response.split("Answer format:")[0].split("Extracted answer:")[1].strip()
    try:
        pred_ans = re.findall(
            r"<concise_answer>(.*?)</concise_answer>", cleaned_response, re.DOTALL
        )[0]
    except:
        pred_ans = "Fail to extract"

    score = rule_based_eval_score_longdocurl(
        ground_truth, pred_ans, sample_data["answer_format"]
    )
    score_reasoning = ""

    return score, score_reasoning, pred_ans


async def llm_reevaluate(sample_data: dict, cand_prefix: str) -> float:
    """
    Remedy function for the original evaluation pipeline in MMLongBench-Doc (get_score_for_response_xxx)
    When cases are assessed as incorrect by rule-based metrics, this function will be triggered to redeem semantically correct ones.
    """
    question, ground_truth, prediction = (
        sample_data["question"],
        sample_data["answer"],
        sample_data[f"{cand_prefix}prediction"],
    )
    user_prompt = (
        f"Question: {question}\nGround Truth: {ground_truth}\nPrediction: {prediction}"
    )
    response_text_list, _ = await call_gemini_with_retry_async(
        model_name="gemini-2.5-flash",
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=REEVALUATE_SYSTEM_PROMPT,
            temperature=0,
            candidate_count=1,
            max_output_tokens=10000,
        ),
    )
    try:
        cleaned_response = (
            response_text_list[0].replace("```json", "").replace("```", "").strip()
        )
        eval_result = json_repair.loads(cleaned_response)
        if not isinstance(eval_result, dict):
            eval_result = {}
    except Exception as e:
        eval_result = {}
        print(e)

    score = max(eval_result.get("fixed_score", 0.0), sample_data[f"{cand_prefix}score"])
    score = min(max(score, 0.0), 1.0)
    score_reasoning = eval_result.get("fixing_reasoning", cleaned_response)

    return score, score_reasoning, ""
