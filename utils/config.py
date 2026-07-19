"""
Configuration for experiments
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExpConfig:
    """Experiment configuration"""

    phase1_input_pages: str = "all"
    phase1_input_mode: str = "use_ocr"
    phase1_candidate_num: int = 8

    phase2_input_pages: str = "pgnav_all_located_pages"
    phase2_input_mode: str = "use_element_localizer"
    phase2_candidate_num: int = 2

    dataset_name: Literal["MMLongBenchDoc", "FinRAGBench-V"] = "MMLongBenchDoc"
    split_name: str = "samples"
    model_name: str = "gemini-2.5-pro"
    phase_name: str = "end2end"
    temperature: float = 0.7
    exp_name: str = ""
    timestamp: str | None = None
    work_dir: Path = Path(__file__).parent.parent

    def __post_init__(self):

        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()  # Only needed once after setting TZ
        timestamp = (
            time.strftime("%m%d_%H%M") if self.timestamp is None else self.timestamp
        )
        self.exp_name = f"{timestamp}_{self.exp_name}"

        # mkdir result_dir if not exists
        self.result_dir = (
            self.work_dir / f"results_{self.model_name}" / self.dataset_name
        )
        self.result_dir.mkdir(exist_ok=True, parents=True)
