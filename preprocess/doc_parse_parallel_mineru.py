"""
A script to preprocess document directories in parallel using multiple GPUs.
It processes each PDF directory using the `mineru` tool and organizes the output.
"""

import os
import subprocess
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_single_pdf_directory(gpu_id, pdf_dir_path):
    """
    Core worker function to process a single PDF directory.
    Each subprocess will load its own pipeline instance to execute this function.
    """
    doc_id = pdf_dir_path.name
    try:
        path_to_output_folder = pdf_dir_path
        # Run command line command: MINERU_TABLE_ENABLE=false mineru -p {pdf_dir_path} -o {path_to_output_folder}
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        my_env["MINERU_TABLE_ENABLE"] = "false"
        command = ["mineru", "-p", str(pdf_dir_path), "-o", str(path_to_output_folder)]
        # To see output for debugging, uncomment the second line and comment the first.
        subprocess.run(
            command,
            env=my_env,
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for item in path_to_output_folder.iterdir():
            if item.is_dir() and item.name.isdigit():
                page_num = item.name
                page_dir = item
                source_dir = page_dir / "auto"
                destination_dir = path_to_output_folder / f"MinerU_Page{page_num}"
                if source_dir.is_dir() and not destination_dir.exists():
                    shutil.move(str(source_dir), str(destination_dir))
                    shutil.rmtree(page_dir)
                elif source_dir.is_dir() and destination_dir.exists():
                    shutil.rmtree(destination_dir)
                    shutil.move(str(source_dir), str(destination_dir))
                    shutil.rmtree(page_dir)
                elif not source_dir.is_dir():
                    shutil.rmtree(page_dir)
        return doc_id, "processed"

    except Exception as e:
        return doc_id, f"error: {e}"


def main(args):
    """
    Main function to set up parallel processing of PDF directories.
    """
    print("Scanning image folders to prepare for Doc Parsing tasks...")
    base_document_path = Path(args.dataset_folder)

    # Tasks are all the directories under base_document_path
    tasks = []
    pdf_list = [d for d in base_document_path.iterdir() if d.is_dir()]
    for pdf_dir in tqdm(pdf_list, desc="Scanning PDF directories for tasks"):
        img_list = list(pdf_dir.glob("*.jpeg"))
        needs_processing = False
        for img in img_list:
            img_page_num = img.stem
            mineru_page_dir = pdf_dir / f"MinerU_Page{img_page_num}"
            if (
                not mineru_page_dir.exists()
                or not (mineru_page_dir / f"{img_page_num}_content_list.json").exists()
            ):
                needs_processing = True
                break
        if needs_processing:
            print(f"Scheduling document {pdf_dir.name} for processing...")
            tasks.append(pdf_dir)

    num_gpus = args.num_gpus
    max_workers = args.max_workers
    print(
        f"Found {len(tasks)} valid PDF tasks. Using {max_workers} processes for parallel processing..."
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for i, doc_path in enumerate(tasks):
            gpu_id = i % num_gpus
            future = executor.submit(process_single_pdf_directory, gpu_id, doc_path)
            future_to_info[future] = doc_path.name
        for future in tqdm(
            as_completed(future_to_info),
            total=len(tasks),
            desc="Processing Directories",
        ):
            try:
                _, _ = future.result()
            except Exception as e:
                doc_id_err = future_to_info[future]
                print(
                    f"Main process caught an exception for task: document={doc_id_err}, error={e}"
                )

    print("\nAll PDF processing tasks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel document processing script.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="The full path to the directory containing document subdirectories.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs available for processing.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker processes to use.",
    )
    parser_args = parser.parse_args()
    main(parser_args)
