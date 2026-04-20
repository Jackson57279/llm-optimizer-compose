"""Model download utilities."""

from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


async def download_from_huggingface(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> Path:
    """
    Download file from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "MiniMaxAI/MiniMax-M2.7")
        filename: Specific file to download
        cache_dir: Custom cache directory
        local_files_only: Only use local cache

    Returns:
        Path to downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )

    if filename:
        # Download specific file
        path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
            ),
        )
        return Path(path)
    else:
        # Download entire repo
        path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
            ),
        )
        return Path(path)


import asyncio
