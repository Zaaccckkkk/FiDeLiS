from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from openai import OpenAI

from vanilla_baseline.data_loading import VanillaSample


def load_api_key(config_path: Optional[str], explicit_api_key: Optional[str]) -> str:
   if explicit_api_key:
      return explicit_api_key
   if config_path:
      with open(config_path, "r") as handle:
         config = json.load(handle)
      api_key = config.get("OPENAI_API_KEY", "")
      if api_key:
         return api_key
   raise ValueError("OpenAI API key is missing. Set --api_key or fill OPENAI_API_KEY in --config_path.")


def build_prompt(sample: VanillaSample) -> List[dict]:
   if sample.task_type == "claim_verification":
      instruction = (
         "Decide whether the claim is true or false. "
         "Return exactly one token: True or False. Do not explain."
      )
      user = f"Claim: {sample.question}"
   elif sample.dataset == "crlt" and sample.answers and set(sample.answers).issubset({"True", "False"}):
      instruction = (
         "Answer the question using your own knowledge. "
         "Return exactly one token: True or False. Do not explain."
      )
      user = f"Question: {sample.question}"
   else:
      instruction = (
         "Answer the question using your own knowledge only. Do not use retrieved passages or a knowledge graph. "
         "Return the answer as plain text. If there are multiple correct answers, put each answer on a separate line. "
         "Do not include explanations."
      )
      user = f"Question: {sample.question}"

   return [
      {"role": "system", "content": instruction},
      {"role": "user", "content": user},
   ]


def parse_predictions(text: str, task_type: str) -> List[str]:
   stripped = text.strip()
   if not stripped:
      return []

   if task_type == "claim_verification" or stripped.lower() in {"true", "false"}:
      if re.search(r"\btrue\b", stripped, flags=re.IGNORECASE):
         return ["True"]
      if re.search(r"\bfalse\b", stripped, flags=re.IGNORECASE):
         return ["False"]

   try:
      parsed = json.loads(stripped)
      if isinstance(parsed, list):
         return [str(item).strip() for item in parsed if str(item).strip()]
      if isinstance(parsed, str):
         return [parsed.strip()]
   except Exception:
      pass

   lines = []
   for line in stripped.splitlines():
      cleaned = re.sub(r"^\s*[-*]\s*", "", line)
      cleaned = re.sub(r"^\s*\d+[\).\s-]+", "", cleaned)
      cleaned = cleaned.strip()
      if cleaned:
         lines.append(cleaned)

   if len(lines) == 1 and "," in lines[0]:
      parts = [part.strip() for part in lines[0].split(",") if part.strip()]
      if 1 < len(parts) <= 8:
         return parts
   return lines


class OpenAIZeroShotRunner:
   def __init__(
      self,
      api_key: str,
      model: str,
      temperature: float = 0.0,
      max_retries: int = 5,
      retry_sleep: float = 2.0,
   ):
      self.client = OpenAI(api_key=api_key)
      self.model = model
      self.temperature = temperature
      self.max_retries = max_retries
      self.retry_sleep = retry_sleep

   def run(self, sample: VanillaSample) -> dict:
      messages = build_prompt(sample)
      last_error = None
      for attempt in range(self.max_retries):
         try:
            response = self.client.chat.completions.create(
               model=self.model,
               messages=messages,
               temperature=self.temperature,
            )
            text = response.choices[0].message.content or ""
            return {
               "prediction_text": text,
               "predictions": parse_predictions(text, sample.task_type),
               "prompt_messages": messages,
               "usage": response.usage.model_dump() if response.usage is not None else {},
            }
         except Exception as exc:
            last_error = exc
            time.sleep(self.retry_sleep * (attempt + 1))
      raise RuntimeError(f"OpenAI call failed after {self.max_retries} attempts: {last_error}") from last_error

