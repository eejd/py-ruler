"""prepare.py runs its generator with the current interpreter and reports failures."""
import subprocess
import sys
from pathlib import Path

import nltk
import pytest

SCRIPT = Path(__file__).with_name("prepare.py")


def test_prepare_fails_loudly_on_generator_error(tmp_path):
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        pytest.skip("NLTK punkt data not installed")
    # An unloadable tokenizer makes the generator subprocess fail; prepare must not exit 0.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--save_dir", str(tmp_path), "--benchmark", "synthetic",
         "--task", "niah_single_1", "--tokenizer_path", str(tmp_path / "missing"),
         "--tokenizer_type", "hf", "--max_seq_length", "4096", "--model_template_type", "base",
         "--num_samples", "1"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert not (tmp_path / "niah_single_1" / "validation.jsonl").exists()
