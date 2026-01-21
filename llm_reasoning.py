import json
from pathlib import Path

PROMPT_PATH = Path("llm_prompt.md")
SAMPLES_DIR = Path("samples")
OUT_DIR = Path("outputs")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def build_message(prompt_template: str, fit_score: int, resume_json: dict, job_json: dict) -> str:
    # We embed the input objects explicitly so the LLM can reason deterministically.
    payload = {
        "fit_score": fit_score,
        "resume_json": resume_json,
        "job_json": job_json,
    }
    return (
        prompt_template
        + "\n\n"
        + "### INPUT\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        + "### OUTPUT (JSON only)\n"
    )

def call_llm(message: str) -> str:
    """
    Replace this with your LLM call.
    Must return the raw text response from the model (expected to be JSON).
    """
    raise NotImplementedError("Connect your LLM provider here (OpenAI/Azure/etc).")

def safe_parse_json(text: str) -> dict:
    # Some models may wrap JSON with whitespace. We keep it strict.
    text = text.strip()
    return json.loads(text)

def run_one(job_path: Path, resume_path: Path, fit_score: int) -> dict:
    prompt_template = load_prompt()

    resume_json = json.loads(resume_path.read_text(encoding="utf-8"))
    job_json = json.loads(job_path.read_text(encoding="utf-8"))

    message = build_message(prompt_template, fit_score, resume_json, job_json)

    raw = call_llm(message)
    result = safe_parse_json(raw)

    # Enforce fit_score unchanged
    result["fit_score"] = fit_score
    return result

def main():
    """
    Example usage:
    - samples/resume_1.json
    - samples/job_1.json, job_2.json, job_3.json
    """
    resume_path = SAMPLES_DIR / "resume_1.json"

    jobs = [
        (SAMPLES_DIR / "job_1.json", 82),
        (SAMPLES_DIR / "job_2.json", 64),
        (SAMPLES_DIR / "job_3.json", 45),
    ]

    for job_path, score in jobs:
        result = run_one(job_path, resume_path, score)
        out_path = OUT_DIR / f"reasoning_{job_path.stem}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", out_path)

if __name__ == "__main__":
    main()