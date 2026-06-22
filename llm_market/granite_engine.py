"""
Step 3 — Reasoning Engine: batched Granite inference.

Wraps an IBM Granite model and exposes a single `.chat(batch_messages, schemas)`
call that returns one text completion per conversation. Three backends:

  * "vllm"          : high-throughput batched offline inference on the A100
                      (recommended on Gilbreth). Uses structured/guided JSON
                      decoding when the installed vLLM supports it.
  * "transformers"  : Hugging Face fallback (slower, simple batching).
  * "mock"          : no model — returns valid JSON heuristically. Lets you test
                      the whole pipeline locally (e.g. on a laptop) with no GPU.

Batching: all agents' prompts for a single period are sent in ONE call so the GPU
processes them together, mimicking the advisor's "batch prompts, one per agent".
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"


# ----------------------------------------------------------------------
# Mock backend (no GPU / no model needed)
# ----------------------------------------------------------------------
class _MockBackend:
    """Returns valid action JSON without any model.

    Produces generation around a configurable fraction of capacity with small
    noise, so the full pipeline (translate -> prompt -> parse -> env -> log -> plot)
    can be exercised end-to-end locally before running Granite on the cluster.
    """

    def __init__(self, target_fraction: float = 0.62, noise: float = 0.05, seed: int = 0):
        self.target_fraction = target_fraction
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def chat(self, batch_messages, schemas=None, **kwargs) -> list[str]:
        outputs = []
        for i, _msgs in enumerate(batch_messages):
            n = 1
            if schemas and i < len(schemas) and schemas[i]:
                n = int(schemas[i]["properties"]["generation_mw"].get("maxItems", 1))
            # caps are not known to the mock; emit fractions*placeholder, parser clips.
            # We emit plausible MW by assuming a generic 100 MW scale; the action
            # parser clips to the true capacity, so values are always valid.
            frac = np.clip(
                self.target_fraction + self.rng.normal(0, self.noise, size=n),
                0.05,
                1.0,
            )
            mw = (frac * 100.0).round(1).tolist()
            outputs.append(
                json.dumps({
                    "reasoning": "[mock] heuristic output around target fraction.",
                    "strategy": "[mock] hold output near a fixed fraction of capacity.",
                    "generation_mw": mw,
                })
            )
        return outputs


# ----------------------------------------------------------------------
# vLLM backend
# ----------------------------------------------------------------------
class _VLLMBackend:
    def __init__(self, model: str, max_model_len: int = 8192,
                 tensor_parallel_size: int = 1, dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90,
                 quantization: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 256,
                 seed: int = 0):
        from vllm import LLM  # noqa: F401  (import error surfaces clearly)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        llm_kwargs = dict(
            model=model,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)

        # Detect which structured-output API this vLLM version exposes.
        self._struct_mode = self._detect_structured_api()

    def _detect_structured_api(self) -> Optional[str]:
        # Newer vLLM (>= ~0.12): SamplingParams(structured_outputs=StructuredOutputsParams(json=...))
        try:
            from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
            return "structured_outputs"
        except Exception:
            pass
        # Older vLLM (~0.8-0.11): SamplingParams(guided_decoding=GuidedDecodingParams(json=...))
        try:
            from vllm.sampling_params import GuidedDecodingParams  # noqa: F401
            return "guided_decoding"
        except Exception:
            pass
        return None

    def _sampling_params(self, schema: Optional[dict]):
        from vllm import SamplingParams

        common = dict(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        if schema is None or self._struct_mode is None:
            return SamplingParams(**common)

        try:
            if self._struct_mode == "structured_outputs":
                from vllm.sampling_params import StructuredOutputsParams
                return SamplingParams(
                    structured_outputs=StructuredOutputsParams(json=schema),
                    **common,
                )
            if self._struct_mode == "guided_decoding":
                from vllm.sampling_params import GuidedDecodingParams
                return SamplingParams(
                    guided_decoding=GuidedDecodingParams(json=schema),
                    **common,
                )
        except Exception:
            # Schema rejected by backend (e.g. xgrammar feature gap) -> plain sampling.
            return SamplingParams(**common)
        return SamplingParams(**common)

    def chat(self, batch_messages, schemas=None, **kwargs) -> list[str]:
        schemas = schemas or [None] * len(batch_messages)
        sp_list = [self._sampling_params(s) for s in schemas]
        outs = self.llm.chat(messages=batch_messages, sampling_params=sp_list, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


# ----------------------------------------------------------------------
# Hugging Face transformers backend
# ----------------------------------------------------------------------
class _TransformersBackend:
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 256,
                 dtype: str = "auto", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.max_tokens = max_tokens
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        self.model.eval()

    def chat(self, batch_messages, schemas=None, **kwargs) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batch_messages
        ]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


# ----------------------------------------------------------------------
# Public engine
# ----------------------------------------------------------------------
class GraniteEngine:
    """Backend-agnostic reasoning engine."""

    def __init__(self, backend: str = "mock", model: str = DEFAULT_MODEL,
                 temperature: float = 0.7, max_tokens: int = 256,
                 max_model_len: int = 8192, tensor_parallel_size: int = 1,
                 quantization: Optional[str] = None, seed: int = 0,
                 mock_target_fraction: float = 0.62):
        self.backend_name = backend
        self.model = model

        if backend == "mock":
            self.backend = _MockBackend(target_fraction=mock_target_fraction, seed=seed)
        elif backend == "vllm":
            self.backend = _VLLMBackend(
                model=model, max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                quantization=quantization, temperature=temperature,
                max_tokens=max_tokens, seed=seed,
            )
        elif backend == "transformers":
            self.backend = _TransformersBackend(
                model=model, temperature=temperature, max_tokens=max_tokens, seed=seed,
            )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use mock|vllm|transformers.")

    def chat(self, batch_messages: list, schemas: Optional[list] = None) -> list[str]:
        """Run one batched inference. Returns one completion string per conversation."""
        return self.backend.chat(batch_messages, schemas=schemas)
