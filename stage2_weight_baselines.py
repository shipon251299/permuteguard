import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.permuteguard_large import get_module_by_name


@dataclass
class SignBitConfig:
    watermark_key: str
    metadata_dir: str = "experiments"
    bit_fraction: float = 0.002
    verification_threshold: float = 0.80
    fingerprint_k: int = 4096


class SignBitWatermark:
    def __init__(self, config: SignBitConfig):
        self.config = config

    @staticmethod
    def _safe_id(model_id: str) -> str:
        return model_id.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _meta_path(self, model_id: str) -> str:
        os.makedirs(self.config.metadata_dir, exist_ok=True)
        return os.path.join(self.config.metadata_dir, f"{self._safe_id(model_id)}_signbit.json")

    def _seed(self, *parts: str) -> int:
        s = "::".join(parts)
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16) % (2**32)

    def _randperm(self, n: int, *parts: str) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed(self.config.watermark_key, *parts))
        return torch.randperm(n, generator=g)

    def _bit_pattern(self, n: int, *parts: str) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed(self.config.watermark_key, *parts, "bits"))
        return torch.randint(0, 2, (n,), generator=g, dtype=torch.int64)

    def _select_indices(self, layer_name: str, n: int) -> torch.Tensor:
        k = max(1, int(round(self.config.bit_fraction * n)))
        k = min(k, n, self.config.fingerprint_k)
        return self._randperm(n, layer_name, "idx")[:k]

    def embed(self, model: nn.Module, model_id: str, selected_layers: Sequence[str]):
        layer_metadata: Dict[str, Any] = {}
        with torch.no_grad():
            for layer_name in selected_layers:
                module = get_module_by_name(model, layer_name)
                flat = module.weight.data.reshape(-1)
                idx = self._select_indices(layer_name, flat.numel())
                bits = self._bit_pattern(idx.numel(), layer_name)
                values = flat[idx.to(flat.device)]
                mags = values.abs().clamp_min(1e-8)
                signed = torch.where(bits.to(flat.device).bool(), mags, -mags)
                flat[idx.to(flat.device)] = signed.to(flat.dtype)
                layer_metadata[layer_name] = {"indices": idx.tolist(), "bits": bits.tolist()}

        meta = {
            "model_id": model_id,
            "key_hash": hashlib.sha256(self.config.watermark_key.encode("utf-8")).hexdigest(),
            "config": asdict(self.config),
            "selected_layers": list(selected_layers),
            "layer_metadata": layer_metadata,
        }
        with open(self._meta_path(model_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return model, meta

    def load_metadata(self, model_id: str) -> Dict[str, Any]:
        with open(self._meta_path(model_id), "r", encoding="utf-8") as f:
            return json.load(f)

    def verify(self, model: nn.Module, model_id: str, watermark_key: Optional[str] = None) -> Dict[str, Any]:
        try:
            meta = self.load_metadata(model_id)
        except FileNotFoundError:
            return {"verified": False, "confidence": 0.0, "error": "metadata_missing", "layers_checked": 0, "layer_scores": {}}

        if watermark_key is not None:
            supplied_hash = hashlib.sha256(watermark_key.encode("utf-8")).hexdigest()
            if supplied_hash != meta["key_hash"]:
                return {"verified": False, "confidence": 0.0, "error": "watermark_key_mismatch", "layers_checked": 0, "layer_scores": {}}

        scores = {}
        total = 0.0
        for layer_name in meta["selected_layers"]:
            module = get_module_by_name(model, layer_name)
            flat = module.weight.data.reshape(-1).detach().float().cpu()
            idx = torch.tensor(meta["layer_metadata"][layer_name]["indices"], dtype=torch.long)
            bits = torch.tensor(meta["layer_metadata"][layer_name]["bits"], dtype=torch.long)
            selected = flat[idx]
            observed = (selected >= 0).long()
            match = (observed == bits).float().mean().item()
            scores[layer_name] = float(match)
            total += float(match)

        conf = total / max(len(meta["selected_layers"]), 1)
        return {
            "verified": bool(conf >= meta["config"]["verification_threshold"]),
            "confidence": float(conf),
            "layers_checked": len(meta["selected_layers"]),
            "layer_scores": scores,
            "threshold_used": meta["config"]["verification_threshold"],
            "selected_layers": meta["selected_layers"],
        }
