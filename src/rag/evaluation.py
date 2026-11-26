"""Scoring helpers for the evaluation (kept free of model code so they are unit-testable)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .generate import REFUSAL

# Phrases that count as "the model declined to answer".
REFUSAL_MARKERS = (
    REFUSAL,
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "not found in the",
    "not mentioned in",
    "not specified",
    "not stated in",
    "not provided in",
    "does not specify",
    "does not mention",
    "doesn't specify",
    "doesn't mention",
    "no information",
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "i'm not sure",
    "i am not sure",
    "not able to provide",
    "unable to provide",
    "cannot provide",
    "can't provide",
)


def normalize(text: str) -> str:
    """Lowercase, unify dashes and whitespace, drop markdown emphasis."""
    text = text.lower().replace("*", "")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)  # en/em dashes, minus
    text = re.sub(r"\s*-\s*", "-", text)  # "1 – 15" -> "1-15"
    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)  # "15th" -> "15"
    text = re.sub(r"\b(\d{1,2}) of ", r"\1 ", text)  # "15 of may" -> "15 may"
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word phrase match on normalized text ("1" does not match "10")."""
    p = normalize(phrase)
    if not p:
        return False
    pattern = re.escape(p)
    if p[0].isalnum():
        pattern = r"(?<![\w])" + pattern
    if p[-1].isalnum():
        pattern += r"(?![\w])"
    return re.search(pattern, normalize(text)) is not None


def keyfact_recall(answer: str, must_include: list[list[str]]) -> float:
    """Fraction of key facts present. Each fact is a list of accepted alternatives."""
    if not must_include:
        return 1.0
    hits = sum(any(contains_phrase(answer, alt) for alt in fact) for fact in must_include)
    return hits / len(must_include)


def is_refusal(answer: str) -> bool:
    ans = normalize(answer)
    return any(m in ans for m in REFUSAL_MARKERS)


def auto_correct(item: dict, answer: str) -> bool:
    """Deterministic correctness: all key facts for answerable questions, refusal otherwise.

    For answerable questions a refusal ("... does not specify whether it includes lab
    sessions") is wrong even if a key-fact string happens to appear in it.
    """
    if not item.get("answerable", True):
        return is_refusal(answer)
    return keyfact_recall(answer, item["must_include"]) == 1.0 and not is_refusal(answer)


def gold_rank(item: dict, chunks: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk from the gold document/section."""
    gold = item.get("gold")
    if not gold:
        return None
    for rank, c in enumerate(chunks, 1):
        if c["id"].split(":")[0] == gold["doc"] and gold["section"].lower() in c["section"].lower():
            return rank
    return None


# ---------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are grading a question-answering system about university documents.

Question: {question}
Reference answer: {reference}
System answer: {answer}
{context_block}
Grade the system answer. Reply with JSON only, filling the fields in this order:
{{"unsupported_claims": [<claims in the system answer that the context does not support or that contradict it>],
  "reason": "<one short sentence about correctness>",
  "correct": 0 or 1,
  "faithful": {faithful_spec}}}

- "correct" = 1 if the system answer gives the same essential facts as the reference answer
  (wording may differ; extra correct detail is fine). If the reference says the information is
  not in the documents, "correct" = 1 only if the system says it cannot find / does not know it.
- "faithful" = {faithful_def}
"""


def _as_grade(parsed) -> dict:
    """The model sometimes wraps the grade object in a list: [{...}]."""
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict) or "correct" not in parsed:
        raise json.JSONDecodeError("unexpected judge output", str(parsed), 0)
    return parsed


class GeminiJudge:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    def _call(self, prompt: str) -> dict:
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 4000,  # newer models spend tokens on thinking first
            },
        }).encode()
        req = urllib.request.Request(
            self.URL.format(model=self.model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read())
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                return _as_grade(json.loads(text))
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
            except (KeyError, json.JSONDecodeError, TimeoutError, urllib.error.URLError):
                if attempt == 4:
                    raise
            time.sleep(5 * 2**attempt)
        raise RuntimeError("unreachable")

    def grade(self, question: str, reference: str, answer: str, context: str | None) -> dict:
        if context is None:
            context_block, f_spec, f_def = "", "null", "null (no context was used); leave unsupported_claims empty."
        else:
            context_block = f"\nRetrieved context given to the system:\n{context}\n"
            f_spec = "0 or 1"
            f_def = (
                "1 if unsupported_claims is empty, i.e. every factual claim in the system answer "
                "(every number, grade, date, name) is stated in the retrieved context; 0 if any "
                "claim is missing from or contradicts the context. A refusal counts as faithful."
            )
        return self._call(JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer,
            context_block=context_block, faithful_spec=f_spec, faithful_def=f_def,
        ))
