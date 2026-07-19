"""
Download dataset from Hugging Face Hub
"""

from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Lillianwei/Mdocagent-dataset"
target_dir = Path(__file__).parent.parent / "data"
target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    repo_type="dataset",
    allow_patterns="FetaTab/*",
)
