"""
Laya (System-One) decision model provider.

Laya is not a plain HF sequence classifier: it pairs a ModernBERT encoder with a
from-scratch decision head, and its repo ships the `rl_common.py` module that
defines both. We load that module from the downloaded snapshot rather than
vendoring a copy, so the sequence layout and temperature handling stay in sync
with upstream.

Install extras first:

    uv sync --extra laya
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from decisionops.schemas import DecisionCase, DecisionPrediction

DEFAULT_REPO = "convaiinnovations/laya"
# Files needed for local inference (skips assets, multilingual, eval images).
SNAPSHOT_PATTERNS = [
    "rl_common.py",
    "rl_agent_config.json",
    "encoder/config.json",
    "model.safetensors",
    "tokenizer/*",
]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Laya provider needs torch/transformers.\n"
            "Install with:  uv sync --extra laya"
        ) from exc
    return torch


def download_snapshot(repo_id: str = DEFAULT_REPO, subdir: str | None = None) -> Path:
    """
    Fetch the minimal file set for local Laya inference.

    `subdir` selects a variant stored inside the main repo, e.g. "typed-decisions".
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Need huggingface_hub. Install with: uv sync --extra laya") from exc

    patterns = list(SNAPSHOT_PATTERNS)
    if subdir:
        patterns = ["rl_common.py"] + [f"{subdir}/{p}" for p in SNAPSHOT_PATTERNS[1:]]

    path = Path(snapshot_download(repo_id, allow_patterns=patterns))
    return path / subdir if subdir else path


def _load_rl_common(snapshot: Path):
    """Import the upstream rl_common module from the snapshot directory."""
    # rl_common.py lives at the repo root even when weights are in a subdir.
    candidates = [snapshot / "rl_common.py", snapshot.parent / "rl_common.py"]
    module_path = next((c for c in candidates if c.exists()), None)
    if module_path is None:
        raise FileNotFoundError(
            f"rl_common.py not found near {snapshot}. Re-run download_snapshot()."
        )
    if "rl_common" in sys.modules:
        return sys.modules["rl_common"]
    spec = importlib.util.spec_from_file_location("rl_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load rl_common from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_common"] = module
    spec.loader.exec_module(module)
    return module


def _pick_device(requested: str | None) -> str:
    torch = _require_torch()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def laya_question(
    case: DecisionCase,
    *,
    instructions: str | None = None,
    contract_instructions: str | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Jev-shaped choice question for Laya.

    Upstream `render_options` renders a non-empty criterion as "choice: text",
    so contract descriptions reach the model without any format work here.
    """
    descriptions = descriptions or {}
    return {
        "t": "choice",
        "ins": instructions or contract_instructions or case.question or "Pick the best option.",
        # dict keeps option order; None means "label only", as before descriptions
        "crit": {choice: descriptions.get(choice) for choice in case.choices},
    }


class LayaProvider:
    """
    Local Laya decision model.

    Each DecisionCase becomes a Jev-shaped `choice` question whose options are
    the case's `choices`, and the model returns a probability over those options.
    """

    name = "laya"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str = DEFAULT_REPO,
        subdir: str | None = None,
        device: str | None = None,
        apply_builtin_temperature: bool = True,
        instructions: str | None = None,
        descriptions: dict[str, str] | None = None,
        contract_instructions: str | None = None,
    ) -> None:
        torch = _require_torch()
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        snapshot = Path(model_dir) if model_dir else download_snapshot(repo_id, subdir)
        self.snapshot = snapshot
        self.rl = _load_rl_common(snapshot)

        with (snapshot / "rl_agent_config.json").open() as f:
            self.cfg: dict[str, Any] = json.load(f)

        self.device = torch.device(_pick_device(device))
        self.tok = AutoTokenizer.from_pretrained(str(snapshot / "tokenizer"))
        self.model = self.rl.build_model(self.cfg, encoder_dir=str(snapshot / "encoder"))
        state = load_file(str(snapshot / "model.safetensors"))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # torch.compile is a loss for small batches; upstream disables it too.
        if hasattr(self.model.encoder, "config"):
            self.model.encoder.config.reference_compile = False

        # Laya ships post-hoc temperatures, including per-option-count buckets.
        self.apply_builtin_temperature = apply_builtin_temperature
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.instructions = instructions
        self.descriptions = dict(descriptions or {})
        self.contract_instructions = contract_instructions
        self.model_name = self.cfg.get("model_name", "laya")
        if subdir:
            self.model_name = f"{self.model_name}/{subdir}"

    # ---------------------------------------------------------------- internals

    def _question(self, case: DecisionCase) -> dict[str, Any]:
        return laya_question(
            case,
            instructions=self.instructions,
            contract_instructions=self.contract_instructions,
            descriptions=self.descriptions,
        )

    def _temperature_for(self, qtype: int, n_options: int) -> float:
        if not self.apply_builtin_temperature:
            return 1.0
        bucket = self.rl.temp_bucket(qtype, n_options)
        return float(
            self.temperature_by_options.get(bucket, self.temperature[qtype])
        )

    # ------------------------------------------------------------------- public

    def predict(self, case: DecisionCase) -> DecisionPrediction:
        return self.predict_many([case])[0]

    def predict_many(
        self,
        cases: list[DecisionCase],
        *,
        batch_size: int = 8,
        progress_every: int = 0,
    ) -> list[DecisionPrediction]:
        torch = _require_torch()
        rl = self.rl
        qtype = rl.QTYPES["choice"]
        out: list[DecisionPrediction] = []

        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            items = []
            option_counts = []
            for case in chunk:
                q = self._question(case)
                seq, markers = rl.build_sequence(
                    self.tok, case.state, q, self.cfg["max_len"], self.cfg["head_max_len"]
                )
                n_opts = len(rl.render_options(q))
                if len(markers) != n_opts:
                    raise ValueError(
                        f"case {case.id!r}: {n_opts} options do not fit in "
                        f"head_max_len={self.cfg['head_max_len']} tokens"
                    )
                option_counts.append(n_opts)
                items.append(
                    {
                        "ids": seq,
                        "markers": markers,
                        "qtype": qtype,
                        "target": [0.0] * len(markers),
                        "label": -1,
                        "episode": 0,
                        "ep_step": 0,
                        "ep_len": 1,
                        "src": "decisionops",
                    }
                )

            batch = rl.collate_items([items], self.tok.pad_token_id)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, _act = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["marker_pos"].to(self.device),
                    batch["marker_mask"].to(self.device),
                    batch["qtype"].to(self.device),
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_case_ms = elapsed_ms / max(len(chunk), 1)
            logits = logits.float().cpu()

            for row, case in enumerate(chunk):
                k = option_counts[row]
                t = self._temperature_for(qtype, k)
                z = logits[row, :k] / t
                probs_t = torch.softmax(z, dim=-1)
                probs = {
                    choice: float(probs_t[i]) for i, choice in enumerate(case.choices[:k])
                }
                total = sum(probs.values()) or 1.0
                probs = {key: value / total for key, value in probs.items()}
                prediction = max(probs, key=probs.get)
                out.append(
                    DecisionPrediction(
                        id=case.id,
                        prediction=prediction,
                        probabilities=probs,
                        latency_ms=round(per_case_ms, 3),
                        provider=self.name,
                        model=self.model_name,
                        metadata={
                            "temperature_applied": t,
                            "n_options": k,
                            "device": str(self.device),
                        },
                    )
                )

            if progress_every and (start // batch_size) % progress_every == 0:
                done = min(start + batch_size, len(cases))
                print(f"  laya: {done}/{len(cases)} cases", flush=True)

        return out
