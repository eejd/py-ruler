"""evaluate.py scores correct and wrong predictions differently without NeMo."""
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_scores_known_answers(tmp_path):
    rows = [
        {"index": 0, "input": "q", "outputs": ["4242"], "pred": "The number is 4242", "others": {}},
        {"index": 1, "input": "q", "outputs": ["777"], "pred": "I don't know", "others": {}},
    ]
    (tmp_path / "niah_single_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    script = Path(__file__).with_name("evaluate.py")
    subprocess.run([sys.executable, str(script), "--data_dir", str(tmp_path), "--benchmark", "synthetic"],
                   check=True, capture_output=True)
    summary = (tmp_path / "summary-niah_single_1.csv").read_text().splitlines()
    assert summary[2].split(",")[1] == "50.0"
