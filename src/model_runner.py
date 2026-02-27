# src/model_runner.py
import json
import hashlib
import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# local imports (assumes these exist)
from src.rewrite_prompt import RewritePromptBuilder, RewriteValidator


class ResumeRewriter:
    """
    GPU-only resume rewrite runner using a quantized phi-3-mini instruction model.
    - rewrite_resume() sends the full, strict prompt and validates JSON output.
    - rewrite_line() kept for quick single-line rewriting (backwards-compatible).
    """

    def __init__(self,
                 model_id: str = "microsoft/Phi-3-mini-4k-instruct",
                 max_new_tokens: int = 512):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        # small in-memory cache: prompt_hash -> (parsed_json, raw_text)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # quantization config (keep as in your snippet)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True
        )

        # Ensure tokenizer has pad token (some instruction models need it)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model load: device_map set to cuda to ensure GPU usage only
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="eager"
        ).eval()

        # Prompt builder & validator
        self.builder = RewritePromptBuilder()
        self.validator = RewriteValidator(max_projects=2, bullet_word_limit=25)

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def _generate_from_prompt(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Tokenize on GPU, generate, decode. Deterministic settings (temperature 0, no sampling).
        Returns raw generated string (decoded tokens after prompt).
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        input_len = inputs.input_ids.shape[-1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        gen_tokens = out[0][input_len:]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return decoded

    def rewrite_line(self, original_line: str, allowed_keywords: Optional[list] = None) -> str:
        """
        Backwards-compatible single-line rewrite helper (keeps original behavior).
        This still uses GPU and quantized model.
        """
        allowed_keywords = allowed_keywords or []
        prompt = (
            f"Rewrite resume line.\n"
            f"Line: {original_line}\n"
            f"Keywords: {', '.join(allowed_keywords)}\n"
            "Rules: same meaning, one sentence, active voice, concise.\n"
            "Output: rewrite only (no extra commentary)."
        )

        return self._generate_from_prompt(prompt, max_new_tokens=40)

    def rewrite_resume(self, decision_plan: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full-prompt rewrite flow:
        - Builds strict prompt via RewritePromptBuilder
        - Uses cache if available
        - Calls model, validates output with RewriteValidator
        - If validation fails: retry once with corrective prompt containing the single validation error
        - Returns parsed JSON on success or raises RuntimeError on persistent failure
        """
        prompt = self.builder.build_prompt(decision_plan, resume_data)
        ph = self._prompt_hash(prompt)
        # cache hit
        if ph in self._cache:
            return self._cache[ph]["parsed"]

        # Single-run then one retry on validation failure
        raw_out = self._generate_from_prompt(prompt, max_new_tokens=self.max_new_tokens)

        ok, err = self.validator.validate(raw_out, decision_plan, resume_data)
        if ok:
            parsed = json.loads(raw_out)
            # cache and return
            self._cache[ph] = {"parsed": parsed, "raw": raw_out, "time": time.time()}
            return parsed

        # If validation failed, attempt a single corrective retry
        correction_prompt = (
            "The previous output was invalid. Please fix the JSON only. "
            f"Validation error: {err}\n\n"
            "Original prompt (for reference):\n" +
            prompt
        )

        raw_retry = self._generate_from_prompt(correction_prompt, max_new_tokens=self.max_new_tokens)
        ok2, err2 = self.validator.validate(raw_retry, decision_plan, resume_data)
        if ok2:
            parsed = json.loads(raw_retry)
            self._cache[ph] = {"parsed": parsed, "raw": raw_retry, "time": time.time()}
            return parsed

        # persistent failure -> raise with both validation messages and raw outputs
        raise RuntimeError(f"LLM output failed validation.\nFirst error: {err}\nFirst raw output:\n{raw_out}\n\n"
                           f"Retry error: {err2}\nRetry raw output:\n{raw_retry}")
