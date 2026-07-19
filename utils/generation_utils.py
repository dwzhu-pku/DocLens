"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
import os
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropicVertex

import openai
import google.auth
import google.auth.transport.requests

## initialize clients
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
ANTHROPIC_VERTEX_REGION = os.environ.get("ANTHROPIC_VERTEX_REGION", "global")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://api.siliconflow.cn/v1")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")

gemini_client = None
anthropic_client = None
qwen_client = None


def _get_google_cloud_project() -> str:
    if not GOOGLE_CLOUD_PROJECT:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set to use Vertex AI models.")
    return GOOGLE_CLOUD_PROJECT


def _get_gemini_client():
    global gemini_client
    if gemini_client is None:
        gemini_client = genai.Client(
            vertexai=True,
            project=_get_google_cloud_project(),
            location=GOOGLE_CLOUD_LOCATION,
        )
    return gemini_client


def _get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = AsyncAnthropicVertex(
            region=ANTHROPIC_VERTEX_REGION,
            project_id=_get_google_cloud_project(),
        )
    return anthropic_client


def _get_qwen_client():
    global qwen_client
    if qwen_client is None:
        if not QWEN_API_KEY:
            raise RuntimeError("QWEN_API_KEY must be set to use Qwen models.")
        qwen_client = openai.AsyncOpenAI(
            base_url=QWEN_BASE_URL,
            api_key=QWEN_API_KEY,
            max_retries=0,
            timeout=None,
        )
    return qwen_client


def write_located_pages_to_input_file(result_list, input_filename):
    """
    Write the located pages back to the input file for phase 2 processing.
    """

    def get_unique_key(data):
        return data["question"] + data["doc_id"] + data.get("answer", "")

    result_map = {get_unique_key(result): result for result in result_list}
    with open(input_filename, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for data in data_list:
        unique_id = get_unique_key(data)
        result = result_map.get(unique_id)
        key_to_located_pages = "pgnav_all_located_pages"
        if result and key_to_located_pages in result:
            data[key_to_located_pages] = str(result[key_to_located_pages])
        else:
            data[key_to_located_pages] = "[]"

    with open(input_filename, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    return


def _convert_to_qwen_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of OpenAI style input.

    OpenAI API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,..."},
        ...
    ]
    """
    qwen_parts = []
    for item in contents:
        if item.get("type") == "text":
            qwen_parts.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            qwen_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source.get('media_type')};base64,{source.get('data')}"
                    },
                }
            )
    return qwen_parts


async def call_qwen_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Qwen API with asynchronous retry logic.
    """
    response_text_list = []
    total_consumed_token_num = 0
    current_contents = await resize_contents_images(contents, scale_factor=0.5)

    for attempt in range(max_attempts):
        try:
            qwen_contents = _convert_to_qwen_parts(current_contents)
            qwen_messages_with_system = [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": qwen_contents},
            ]
            response = await _get_qwen_client().chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=qwen_messages_with_system,
                max_tokens=config["max_output_tokens"],
                temperature=config["temperature"],
                n=config["candidate_num"],
            )
            raw_response_text_list = [
                choice.message.content for choice in response.choices
            ]
            response_text_list.extend(
                [r for r in raw_response_text_list if r.strip() != ""]
            )
            total_consumed_token_num += response.usage.total_tokens
            if len(response_text_list) >= config["candidate_num"]:
                response_text_list = response_text_list[: config["candidate_num"]]
                return response_text_list, total_consumed_token_num
            else:
                continue  # Not enough candidates, retry

        except Exception as e:
            error_str = str(e)
            context_msg = f" for {error_context}" if error_context else ""

            if "longer than the maximum model length" in error_str:
                print(
                    f"Error: Input size exceeds limit{context_msg}. Resizing images and retrying..."
                )
                current_contents = await resize_contents_images(
                    current_contents, scale_factor=0.5
                )

            print(
                f"Attempt {attempt + 1} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")

    if len(response_text_list) < config["candidate_num"]:
        response_text_list.extend(
            ["Error"] * (config["candidate_num"] - len(response_text_list))
        )
    return response_text_list, total_consumed_token_num


def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.

    Gemini API's format:
    [
        types.Part.from_text(text="some text"),
        types.Part.from_bytes(data=b"...", mime_type="image/jpeg"),
        ...
    ]
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    response_text_list = []
    total_consumed_token_num = 0
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await _get_gemini_client().aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )
            raw_response_list = [
                part.text
                for candidate in response.candidates
                for part in candidate.content.parts
            ]
            response_text_list.extend([r for r in raw_response_list if r.strip() != ""])
            total_consumed_token_num += response.usage_metadata.total_token_count

            if len(response_text_list) >= target_candidate_count:
                response_text_list = response_text_list[:target_candidate_count]
                break
            else:
                continue  # Not enough candidates, retry
        except Exception as e:
            error_str = str(e)
            context_msg = f" for {error_context}" if error_context else ""

            if "INVALID_ARGUMENT" in error_str:
                print(
                    f"Error: Invalid argument received{context_msg}. Keep only the ocr text and try again"
                )
                # If images are too large, try to only keep the text part
                new_contents = []
                for content in current_contents:
                    if content.get("type") == "text":
                        new_contents.append(content)
                current_contents = new_contents

            print(
                f"Attempt {attempt + 1} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                response_text_list = ["Error"] * target_candidate_count

    if len(response_text_list) < target_candidate_count:
        response_text_list.extend(
            ["Error"] * (target_candidate_count - len(response_text_list))
        )
    return response_text_list, total_consumed_token_num


def _resize_image_sync(base64_data, media_type, scale_factor=0.5):
    """
    Resize a base64 encoded image synchronously.
    """
    try:
        image_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_bytes))
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img_format = media_type.split("/")[-1].upper()

        if img_format == "JPG":
            img_format = "JPEG"
        if img_format == "JPEG":
            img_resized.save(buffer, format=img_format, quality=85, optimize=True)
        else:
            img_resized.save(buffer, format=img_format)

        resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return resized_base64
    except Exception as e:
        print(f"[Warning] Failed to resize image: {e}")
        return base64_data  # return original if failed


async def resize_base64_image(base64_data, media_type, scale_factor=0.5):
    """
    Helper function to resize base64 image asynchronously.
    """
    loop = asyncio.get_event_loop()
    func = partial(_resize_image_sync, base64_data, media_type, scale_factor)
    resized_data = await loop.run_in_executor(None, func)
    return resized_data


async def resize_contents_images(contents, scale_factor=0.5):
    """
    Resize all the base64 images in the contents list by the given scale factor.
    """
    tasks = []
    image_info = []

    for i, content in enumerate(contents):
        if content.get("type") == "image":
            source = content.get("source", {})
            if source.get("type") == "base64":
                base64_data = source.get("data", "")
                media_type = source.get("media_type", "image/jpeg")
                task = resize_base64_image(base64_data, media_type, scale_factor)
                tasks.append(task)
                image_info.append((i, media_type))

    if tasks:
        resized_images = await asyncio.gather(*tasks)

        new_contents = []
        resized_idx = 0
        for i, content in enumerate(contents):
            is_image = any(idx == i for idx, _ in image_info)
            if is_image:
                _, media_type = image_info[resized_idx]
                new_contents.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": resized_images[resized_idx],
                        },
                    }
                )
                resized_idx += 1
            else:
                new_contents.append(content)
        return new_contents

    return contents


def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []
    current_scale = 1.0  # Start with original size

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await _get_anthropic_client().messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""

            # Handle size-related errors by modifying the persistent claude_contents
            if "exceeds" in error_str or "too long" in error_str:
                current_scale *= 0.5
                if current_scale < 0.0625:
                    print(
                        f"[Log] Input is too large even after max resizing{context_msg}. Removing all images."
                    )
                    current_contents = [
                        c for c in current_contents if c.get("type") != "image"
                    ]
                else:
                    print(
                        f"[Log] Input size exceeds limit{context_msg}. Resizing persistent images to {current_scale*100:.1f}%..."
                    )
                    current_contents = await resize_contents_images(
                        current_contents, scale_factor=0.5
                    )
            elif "must be non-empty" in error_str:
                print(f"[Log] Filtering empty text parts from content{context_msg}.")
                current_contents = [
                    c
                    for c in current_contents
                    if not (c.get("type") == "text" and not c.get("text", "").strip())
                ]

            elif "too many images" in error_str:
                print(f"[Log] Filtering to 90 images in content{context_msg}.")
                new_contents = []
                image_count = 0
                for content in current_contents:
                    if content.get("type") == "image":
                        image_count += 1
                        if image_count > 90:
                            continue
                    new_contents.append(content)
                current_contents = new_contents

            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
                _get_anthropic_client().messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],  # Use the now-validated content
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def get_doc_content_list_async(
    data: dict, input_pages: str, input_mode: str, exp_config: dict
):
    """
    Main function to get the document content list based on input mode.
    """
    doc_id = data.get("doc_id", "")
    doc_file_path = (
        exp_config.work_dir / "data" / exp_config.dataset_name / "documents" / doc_id
    )
    content_list = []

    # load target pages
    image_directory = doc_file_path.parent / doc_file_path.stem
    sorted_image_paths = sorted(
        image_directory.glob("*.jpeg"), key=lambda p: int(p.stem)
    )
    all_page_list = [int(image_path.stem) for image_path in sorted_image_paths]
    if input_pages == "all":
        target_pages = all_page_list
    else:
        try:
            target_pages = data.get(input_pages, [])
            target_pages = literal_eval(target_pages)
            target_pages = [page for page in target_pages if page in all_page_list]
        except Exception as e:
            print(
                f"Warning: Could not parse {input_pages} {data.get(input_pages, '')}, setting to all pages. Error: {e}"
            )
            target_pages = []
        target_pages = all_page_list if target_pages == [] else target_pages

    async def _read_image(path):
        async with aiofiles.open(path, mode="rb") as f:
            return await f.read()

    tasks = [
        _read_image(image_directory / f"{page_idx}.jpeg") for page_idx in target_pages
    ]
    page_images_bytes = await asyncio.gather(*tasks)
    assert len(page_images_bytes) == len(
        target_pages
    ), "Mismatch in number of images read"

    for page_idx, page_bytes in zip(target_pages, page_images_bytes):
        content_list.append(
            {
                "type": "text",
                "text": f"---- Screenshot of page {page_idx} ----\n",
            }
        )
        content_list.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(page_bytes).decode("utf-8"),
                },
            }
        )

        if input_mode == "use_ocr":
            content_list.append(
                {"type": "text", "text": f"---- OCR of page {page_idx} ----\n"}
            )
            # content_list.append(types.Part.from_text(text=ocr_text))
            async with aiofiles.open(
                image_directory / f"MinerU_Page{page_idx}/{page_idx}.md", mode="r"
            ) as f:
                markdown_text = await f.read()
            content_list.append({"type": "text", "text": markdown_text})

        elif input_mode == "use_element_localizer":
            content_list.append(
                {"type": "text", "text": f"---- Markdown of page {page_idx} ----\n"}
            )
            async with aiofiles.open(
                image_directory / f"MinerU_Page{page_idx}/{page_idx}.md",
                mode="r",
            ) as f:
                markdown_text = await f.read()
            content_list.append({"type": "text", "text": markdown_text})

            content_list.append(
                {
                    "type": "text",
                    "text": "---- Zoomed-in Figures and Charts of this page ----\n",
                }
            )
            zoomed_image_paths = list(
                (image_directory / f"MinerU_Page{page_idx}/images/").glob("*.jpg")
            )
            for zoomed_image_path in zoomed_image_paths:
                async with aiofiles.open(zoomed_image_path, mode="rb") as f:
                    zoomed_image_bytes = await f.read()
                content_list.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(zoomed_image_bytes).decode(
                                "utf-8"
                            ),
                        },
                    }
                )
        elif input_mode == "vanilla":
            pass
        else:
            raise ValueError(
                f"Invalid input mode: {input_mode}. Use [vanilla, use_ocr, use_element_localizer]"
            )

    return content_list
