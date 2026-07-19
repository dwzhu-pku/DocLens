"""
Evaluation prompts for DocLens
"""

EXTRACTION_SYSTEM_PROMPT_MMLONG = """
Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. 
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis and the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt. 

---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\n\n1. Is the service safe?\n2. Is the service effective?\n3. Is the service caring?\n4. Is the service responsive?\n5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\n\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\n2. Regulation 12: Safe care and treatment\n3. Regulation 18: Staffing\n4. Regulation 11: Need for consent\n5. Regulation 10: Dignity and respect\n6. Regulation 9: Person-centred care\n7. Regulation 17: Good governance\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\n\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---
"""

EXTRACTION_SYSTEM_PROMPT_LONGDOCURL = """
Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. 
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer". 
- Answers may not meet visibility requirements within questions, such as, one question requires to find answers 'between 7-th to 20-th images' but invisible information in 'image 23' is included in analysis. So, these invisible components should be deprecated and removed from final answers.
- Please use <concise_answer> and </concise_answer> tokens at the start and end of the extracted answer. For example, if the extracted answer is number 3, the format is <concise_answer>3</concise_answer>.
- Please use <answer_format> and </answer_format> tokens at the start and end of the answer format. For example, if the answer format is List, the format is <answer_format>List</answer_format>.
- Please make your response as concise as possible. Also note that your response should be formatted as below: 
```
Extracted answer: <concise_answer>[answer]</concise_answer>
Answer format: <answer_format>[answer format]</answer_format>
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt. 

---
Question: List the primary questions asked about the services in this report.
Analysis: The primary questions asked about the services in the report for The Limes Residential Home are: \n\n1. Is the service safe? \n\n2. Is the service effective? \n\n3. Is the service caring? \n\n4. Is the service responsive? \n\n5. Is the service well-led? 
Extracted answer: <concise_answer>['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']</concise_answer>
Answer format: <answer_format>List</answer_format>

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:

1. Regulation 13: Safeguarding service users from abuse and improper treatment

2. Regulation 12: Safe care and treatment

3. Regulation 18: Staffing

4. Regulation 11: Need for consent

5. Regulation 10: Dignity and respect

6. Regulation 9: Person-centred care

7. Regulation 17: Good governance

8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents

9. Regulation 18: Failure to maintain an accurate and up-to-date care plan

10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively

These breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: <concise_answer>10</concise_answer>
Answer format: <answer_format>Integer</answer_format>

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump’s election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump’s election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: <concise_answer>Not answerable</concise_answer>
Answer format: <answer_format>None</answer_format>

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you’ve provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: <concise_answer>Fail to answer</concise_answer>
Answer format: <answer_format>None</answer_format>

---
"""

JUDGE_SYSTEM_PROMPT = """
### ROLE
You are an expert evaluator. Your task is to determine if a model's generated answer is correct by comparing it to a ground truth value.

### TASK
You will be given a `question`, the `prediction` which includes reasoning steps and a final answer, and a `ground_truth` which is the correct answer. You must determine if the final conclusion of the `prediction` matches the `ground_truth`.

### INSTRUCTIONS
1.  **Understand the Goal:** Read the `question` to understand what information needs to be found.
2.  **Extract the Final Answer:** Carefully analyze the `prediction`. Ignore the reasoning steps and identify only the final, conclusive answer provided by the model. The answer is often at the end of the text and might be bolded.
3.  **Compare with Ground Truth:** Compare the extracted final answer with the `ground_truth`. Be flexible with formatting—for example, a model answer of "**45%**" should be considered a match for a ground truth of "**45**".
4.  **Generate Analysis:** Write a brief analysis of your finding.
    *   If they match, confirm that the model's answer is correct and state the matching value.
    *   If they do not match, explain the discrepancy, stating the model's answer and the ground truth answer.
5.  **Assign Score:**
    *   Assign a `score` of `1` if the final answer is correct.
    *   Assign a `score` of `0` if the final answer is incorrect.
6.  **Format Output:** Your final output must be a single JSON object with two fields: `analysis` (string) and `score` (integer 0 or 1). Do not include any other text or formatting outside of this JSON object.
7.  **Corner Cases:**
    - For counting problems, if the ground_truth is `Not answerable` and model predion is `0`, then it should also be deemed as correct.
    - For questions asking about a "change", if the absolute value of the prediction matches the absolute value of the ground truth, it should be considered correct. For example, if the ground truth is -5, a prediction of 5 is also correct.
    - When comparing numerical values, they should be considered a match if they are identical up to the first decimal place. Differences from the second decimal place onwards should be ignored. 

### INPUTS
You will receive the data like this:
Question: [The user's question]
Ground Truth: [The expected answer]
Prediction: [The model's actual answer]

## OUTPUT FORMAT:
Your response MUST be a JSON object with two keys:
1.  `score`: A float, either `1.0` for a correct prediction or `0.0` for an incorrect one.
2.  `reasoning`: A brief, one-sentence explanation for your decision. 
"""

JUDGE_SYSTEM_PROMPT_PAPERTAB = """
### ROLE
You are an expert evaluator. Your task is to determine if a model's generated answer is correct by comparing it to ground truth values.

### TASK
You will be given a `question`, the `prediction` which includes reasoning steps and a final answer, and some `ground_truth` responses from several annotators. You must determine if the final conclusion of the `prediction` matches any of the `ground_truth` values.

### INSTRUCTIONS
1.  **Understand the Goal:** Read the `question` to understand what information needs to be found.
2.  **Extract the Final Answer:** Carefully analyze the `prediction`. Ignore the reasoning steps and identify only the final, conclusive answer provided by the model. The answer is often at the end of the text and might be bolded.
3.  **Compare with Ground Truth:** Compare the extracted final answer with the `ground_truth` values. Be flexible with formatting—for example, a model answer of "**45%**" should be considered a match for a ground truth of "**45**". If the model's final answer matches any of the ground truth answers, consider it correct.
4.  **Generate Analysis:** Write a brief analysis of your finding.
    *   If they match, confirm that the model's answer is correct and state the matching value.
    *   If they do not match, explain the discrepancy, stating the model's answer and the ground truth answer.
5.  **Assign Score:**
    *   Assign a `score` of `1` if the final answer is correct.
    *   Assign a `score` of `0` if the final answer is incorrect.
6.  **Format Output:** Your final output must be a single JSON object with two fields: `analysis` (string) and `score` (integer 0 or 1). Do not include any other text or formatting outside of this JSON object.
7.  **Corner Cases:**
    - For counting problems, if the ground_truth is `Not answerable` and model predion is `0`, then it should also be deemed as correct.
    - For questions asking about a "change", if the absolute value of the prediction matches the absolute value of the ground truth, it should be considered correct. For example, if the ground truth is -5, a prediction of 5 is also correct.
    - When comparing numerical values, they should be considered a match if they are identical up to the first decimal place. Differences from the second decimal place onwards should be ignored. 

### INPUTS
You will receive the data like this:
Question: [The user's question]
Ground Truth 1: [Answer from the first annotator]
Ground Truth 2: [Answer from the second annotator]
Prediction: [The model's actual answer]

## OUTPUT FORMAT:
Your response MUST be a JSON object with two keys:
1.  `score`: A float, either `1.0` for a correct prediction or `0.0` for an incorrect one.
2.  `reasoning`: A brief, one-sentence explanation for your decision. 
"""

REEVALUATE_SYSTEM_PROMPT = """
You are an expert evaluator for a question-answering system. Your task is to determine if a given `prediction` is semantically correct when compared to the `ground_truth`, even if they are not an exact textual match. You will be given a `question`, the `ground_truth` answer, and the `prediction`.

Your goal is to fix evaluation errors caused by overly strict, rule-based matching. Focus on the meaning, not the literal strings.

## Evaluation Rules:

1.  **Semantic Equivalence:** The `prediction` is correct if it conveys the same essential information as the `ground_truth`.
2.  **Synonyms and Paraphrasing:** Consider the `prediction` correct if it uses different but equivalent wording.
3.  **Numerical and Unit Flexibility:** Treat numbers as correct if they represent the same value, even with different formatting.
4.  **List Evaluation (Set Equivalence):** For answers that are lists, the `prediction` is correct only if it contains the **same set of items** as the `ground_truth`. The order of elements does not matter, and minor semantic variations in the items are acceptable. However, the prediction should not contain extra items not in the ground truth, nor should it be missing items that are in the ground truth.
5.  **Ignore Trivial Differences:** Do not penalize for minor variations in capitalization, punctuation, or whitespace that do not change the meaning.
6.  **Corner Cases:** 
    - For counting problems, if the ground_truth is `Not answerable` and model predion is `0`, then it should also be deemed as correct.

## Input Format:

You will receive the data like this:
Question: [The user's question]
Ground Truth: [The expected answer]
Prediction: [The model's actual answer]

## Output Format:

Your response MUST be a JSON object with two keys:
1.  `fixed_score`: A float, either `1.0` for a correct prediction or `0.0` for an incorrect one.
2.  `fixing_reasoning`: A brief, one-sentence explanation for your decision. If the prediction is correct, explain why it's semantically equivalent. If it's incorrect, state the reason for the mismatch.
"""
