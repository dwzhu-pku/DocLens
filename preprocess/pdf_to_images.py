"""
A script to convert all pages of PDF files in a specified directory into images in parallel.
Each PDF file will have its own subdirectory containing the images of its pages.
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf  # fitz
from tqdm import tqdm


def process_single_page(
    pdf_path_str: str, page_index: int, output_path_str: str, dpi: int
):
    """
    An independent worker function responsible for processing a single page.
    Each parallel process will execute this function.
    """
    try:
        doc = pymupdf.open(pdf_path_str)
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(output_path_str)
        doc.close()
        return f"Successfully processed page: {page_index + 1}"
    except Exception as e:
        return f"Error processing page {page_index + 1}: {e}"


def convert_pdf_to_images_parallel(pdf_path: str, dpi: int, image_format: str):
    """
    The main function that orchestrates the parallel PDF page conversion tasks.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        print(f"Error: File not found -> {pdf_path}")
        return

    output_dir = pdf_file.parent / pdf_file.stem
    if output_dir.exists() and any(output_dir.glob(f"*.{image_format}")):
        print(f"Directory {output_dir} already exists and contains images, skipping.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with pymupdf.open(pdf_file) as doc:
            page_count = len(doc)
    except Exception as e:
        print(f"Error: Could not open PDF file -> {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(page_count):
            output_file_path = output_dir / f"{i + 1}.{image_format}"
            future = executor.submit(
                process_single_page, str(pdf_file), i, str(output_file_path), dpi
            )
            futures.append(future)

        for future in tqdm(
            as_completed(futures), total=page_count, desc=f"Converting {pdf_file.name}"
        ):
            future.result()

    print(f"\nAll pages of {pdf_file.name} have been successfully converted to images!")


def main(args):
    """The main entry point of the script."""
    pdf_directory = Path(args.pdf_directory)
    if not pdf_directory.is_dir():
        print(f"Error: Directory not found -> {pdf_directory}")
        return

    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process in {pdf_directory}")
    for pdf_file in pdf_files:
        convert_pdf_to_images_parallel(str(pdf_file), args.dpi, args.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all pages of PDF files into images in parallel."
    )
    parser.add_argument(
        "--pdf_directory",
        type=str,
        required=True,
        help="Path to the directory containing PDF files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots Per Inch (DPI), affecting the resolution of the output images.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpeg",
        help="The output image format (e.g., 'jpeg', 'png').",
    )
    args = parser.parse_args()
    main(args)
