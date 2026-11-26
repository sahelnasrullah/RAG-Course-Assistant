"""Answer generation with an open-source LLM (transformers or Ollama)."""
from __future__ import annotations

import json
import urllib.request

from . import config

RAG_SYSTEM = (
    "You answer questions about university course documents. "
    "Use ONLY the context passages provided. "
    "If the answer is not in the context, reply exactly: "
    "\"I could not find this in the provided documents.\" "
    "Answer in one or two complete sentences in your own words; do not copy table rows. "
    "Be precise with numbers (credits, counts, dates, grades). "
    "End your answer with the passage numbers you used, e.g. [1] or [1][3]."
)

# Used to detect refusals in evaluation.
REFUSAL = "could not find this in the provided documents"

NO_RAG_SYSTEM = (
    "You answer questions about university courses. Be concise and precise."
)


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c['source']}, page {c['page']}, section: {c['section']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def build_messages(question: str, chunks: list[dict] | None) -> list[dict]:
    if chunks is None:  # baseline without retrieval
        return [
            {"role": "system", "content": NO_RAG_SYSTEM},
            {"role": "user", "content": question},
        ]
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]


class HFGenerator:
    """Runs a chat model with transformers. Uses the GPU if one is available."""

    def __init__(self, model_name: str = config.LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.LLM_DTYPE == "auto":
            # bf16 halves memory (3B model: ~6 GB instead of ~12 GB). T4 GPUs lack bf16.
            if device == "cuda" and not torch.cuda.is_bf16_supported():
                dtype = torch.float16
            else:
                dtype = torch.bfloat16
        else:
            dtype = getattr(torch, config.LLM_DTYPE)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()

    def __call__(self, messages: list[dict]) -> str:
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False
            )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OllamaGenerator:
    def __init__(self, model_name: str = config.LLM_MODEL):
        self.model_name = model_name

    def __call__(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()


def get_generator():
    if config.LLM_BACKEND == "hf":
        return HFGenerator()
    if config.LLM_BACKEND == "ollama":
        return OllamaGenerator()
    raise ValueError(f"Unknown LLM backend: {config.LLM_BACKEND}")
