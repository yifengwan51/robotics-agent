"""OpenAI-compatible client for a locally served VLM (vLLM), reproducing EmbodiedBench's Qwen-VL call path.

Matches embodiedbench.planner.remote_model.RemoteModel._call_qwen7b: data-URL images relabelled as jpeg
(convert_format_2gemini), JSON-schema guided decoding with EB's own schema, temperature 0.
"""
import os
import re
import threading
import time

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "4096"))  # = EB's max_completion_tokens
FAILED_OUTPUT = "<<model call failed>>"  # not valid JSON -> EB planner falls back to a random action


def _relabel_images(messages):
    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = []
        for item in msg["content"]:
            if item.get("type") == "image_url":
                b64 = item["image_url"]["url"].split(",", 1)[1]
                item = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            content.append(item)
        out.append({**msg, "content": content})
    return out


class CallStats:
    """Thread-safe accumulator of per-call latency and token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.calls, self.failures, self.seconds, self.truncated = 0, 0, 0.0, 0
            self.prompt_tokens, self.completion_tokens = 0, 0

    def add(self, seconds, usage=None, failed=False):
        with self._lock:
            self.calls += 1
            self.failures += int(failed)
            self.seconds += seconds
            if usage is not None:
                self.prompt_tokens += usage.prompt_tokens or 0
                self.completion_tokens += usage.completion_tokens or 0

    def snapshot(self):
        with self._lock:
            return dict(calls=self.calls, failures=self.failures, truncated=self.truncated,
                        seconds=round(self.seconds, 2),
                        prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens)


STATS = CallStats()


class LocalOpenAIModel:
    def __init__(self, model_name, language_only=False, task_type=None, base_url=None, retries=4):
        from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide

        self.model_name = model_name
        self.language_only = language_only
        self.task_type = task_type
        self.retries = retries
        self.schema = llm_generation_guide if language_only else vlm_generation_guide
        self.client = OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key="EMPTY", timeout=180)

    def respond(self, message_history):
        messages = message_history if self.language_only else _relabel_images(message_history)
        fmt = {"type": "json_schema", "json_schema": {"name": "embodied_planning", "schema": self.schema}}
        max_tokens = MAX_TOKENS
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, response_format=fmt,
                    temperature=0, max_tokens=max_tokens)
                STATS.add(time.time() - t0, resp.usage)
                return resp.choices[0].message.content
            except Exception as e:  # server restart, timeout, context overflow
                STATS.add(time.time() - t0, failed=True)
                print(f"[LocalOpenAIModel] call failed ({attempt + 1}/{self.retries}): {e}", flush=True)
                m = re.search(r"(\d+)\s+(?:input\s+)?tokens?\s+(?:in the messages|of input|in prompt)", str(e))
                if "context length" in str(e) or "maximum model length" in str(e) or m:
                    # prompt + completion budget exceed the server window: shrink the completion budget only
                    limit = int(os.environ.get("VLM_MAX_MODEL_LEN", "8192"))
                    max_tokens = max(256, (limit - int(m.group(1)) - 8) if m else max_tokens // 2)
                    STATS.truncated += 1
                    continue
                time.sleep(2 * (attempt + 1))
        # the server is unreachable or broken: fail the episode (it is retried) instead of letting the planner
        # fall back to random actions, which would silently corrupt the result
        raise RuntimeError(f"model server unavailable after {self.retries} attempts")
