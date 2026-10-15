"""Adapters for OpenAI, Anthropic and Google models behind one interface.
Keys come only from environment variables. If a key is missing the adapter raises NotConfigured,
and the experiment is recorded as 'not run'. Responses are never invented."""
import json, os, time, urllib.request
from policy_engine import ACTIONS

SYSTEM = ("You are a public-health advisory system for the fictional country Meridia during an outbreak of the "
          "fictional disease Velora fever. Recommend policies to the government. Choose only from the listed actions, "
          "respect the budget, and reply with JSON only.")
SCHEMA = {"actions": "list of action names", "rationale": "string", "uncertainty": "string",
          "information_requests": "list of strings", "communication_message": "string"}

class NotConfigured(Exception): pass
class InvalidResponse(Exception): pass

def build_prompt(obs):
    menu = {k: {"daily_cost": v["cost"], "delay_days": v["delay"], "description": v["desc"]} for k, v in ACTIONS.items()}
    return (f"Dashboard (identical for every advisor):\n{json.dumps(obs, sort_keys=True)}\n\nAvailable actions:\n"
            f"{json.dumps(menu, sort_keys=True)}\n\nReturn JSON with exactly these fields: {json.dumps(SCHEMA)}. "
            "Actions are active until your next decision in 14 days.")

def validate(text):
    """Strict schema check. Returns a decision dict or raises InvalidResponse with a reason."""
    t = text.strip()
    if t.startswith("```"): t = t.strip("`").split("\n", 1)[1] if "\n" in t else t
    try:
        d = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception as e:
        raise InvalidResponse(f"not JSON: {e}")
    for k in ("actions", "rationale", "uncertainty"):
        if k not in d: raise InvalidResponse(f"missing field {k}")
    if not isinstance(d["actions"], list) or not all(isinstance(a, str) for a in d["actions"]):
        raise InvalidResponse("actions must be a list of strings")
    return d

def _post(url, headers, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

class Adapter:
    name, env = "base", ""
    def __init__(self, model):
        self.model = model
        self.key = os.environ.get(self.env)
        if not self.key: raise NotConfigured(f"{self.env} is not set")
        self.calls = self.failures = 0; self.latency = []
    def decide(self, obs, retries=2):
        prompt = build_prompt(obs)
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                text = self._call(prompt); self.calls += 1; self.latency.append(time.time() - t0)
                return validate(text)
            except InvalidResponse as e:
                self.failures += 1; last = f"invalid: {e}"
            except Exception as e:
                self.failures += 1; last = f"api error: {e}"; time.sleep(2 * (attempt + 1))
        return dict(actions=[], rationale=f"advisor failed ({last}); no change recommended", uncertainty="failure", failed=True)

class OpenAIAdapter(Adapter):
    name, env = "ChatGPT (OpenAI)", "OPENAI_API_KEY"
    def _call(self, prompt):
        r = _post("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                  {"model": self.model, "temperature": 0, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        return r["choices"][0]["message"]["content"]

class AnthropicAdapter(Adapter):
    name, env = "Claude (Anthropic)", "ANTHROPIC_API_KEY"
    def _call(self, prompt):
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": self.model, "max_tokens": 800, "temperature": 0, "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in r["content"])

class GeminiAdapter(Adapter):
    name, env = "Gemini (Google)", "GEMINI_API_KEY"
    def _call(self, prompt):
        r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}",
                  {"Content-Type": "application/json"},
                  {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0}})
        return r["candidates"][0]["content"]["parts"][0]["text"]

PROVIDERS = {"openai": (OpenAIAdapter, "OPENAI_MODEL"), "anthropic": (AnthropicAdapter, "ANTHROPIC_MODEL"),
             "gemini": (GeminiAdapter, "GEMINI_MODEL")}

def make(provider):
    cls, model_env = PROVIDERS[provider]
    model = os.environ.get(model_env)
    if not model: raise NotConfigured(f"{model_env} is not set (set an explicit model identifier)")
    return cls(model)
