import re
from pathlib import Path
import pandas as pd

# Paths
DATA_PATH = Path("dataset/raw/postings.csv")
OUT_PATH  = Path("dataset/processed/jobs_clean.csv")


# Text cleaning (general)
def clean_text(s: str) -> str:
    """Basic text cleanup for title/location/work_type/url/experience_level."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")                # non-breaking space
    s = re.sub(r"<[^>]+>", " ", s)              # remove HTML tags if any
    s = re.sub(r"\s+", " ", s).strip()          # normalize whitespace
    return s



# Description cleaning (safe for matching)
_HTML = re.compile(r"<[^>]+>")
_URLS = re.compile(r"(https?://\S+|www\.\S+)")
_MULTI_SPACE = re.compile(r"\s+")

# Allow letters/numbers/space + common tech tokens
# Keeps: + # . - / % ( ) : ,  (useful for C++, C#, Node.js, 4-10, CI/CD, etc.)
_ALLOWED_DESC = re.compile(r"[^A-Za-z0-9\s\+\#\.\-\/\%\(\)\:\,]")

def clean_description(text: str) -> str:
    """
    Cleaning for description (still A2-clean-only, but safer):
    - remove HTML
    - remove URLs
    - keep English letters/numbers + important tech punctuation
    - normalize whitespace
    """
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("\u00a0", " ")
    s = _HTML.sub(" ", s)
    s = _URLS.sub(" ", s)
    s = _ALLOWED_DESC.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s



# Main
def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    print("Raw shape:", df.shape)

    # 2) Validate required columns
    required = ["job_id", "title", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    # 3) Build standardized output (cleaning scope)
    out = pd.DataFrame({
        "job_id": df["job_id"],
        "title": df["title"],
        "company_id": df["company_id"] if "company_id" in df.columns else "",
        "location": df["location"] if "location" in df.columns else "",
        "description": df["description"],

        # keep as raw column but rename to work_type
        "work_type": df["formatted_work_type"] if "formatted_work_type" in df.columns else "",

        # keep as raw columns (no inference)
        "remote_allowed": df["remote_allowed"] if "remote_allowed" in df.columns else "",
        "experience_level": df["formatted_experience_level"] if "formatted_experience_level" in df.columns else "",
        "job_posting_url": df["job_posting_url"] if "job_posting_url" in df.columns else "",
    })

    # 4) Clean text fields
    out["title"] = out["title"].map(clean_text)
    out["location"] = out["location"].map(clean_text)
    out["work_type"] = out["work_type"].map(clean_text)
    out["experience_level"] = out["experience_level"].map(clean_text)
    out["job_posting_url"] = out["job_posting_url"].map(clean_text)

    # IMPORTANT: safer description cleaning (for matching)
    out["description"] = out["description"].map(clean_description)

    # 5) Handle missing values (cleaning only)
    out["remote_allowed"] = out["remote_allowed"].fillna("")

    # 6) Drop empty/too-short descriptions (length in characters)
    out["desc_len"] = out["description"].str.len()
    before = len(out)
    out = out[out["description"].ne("")]
    out = out[out["desc_len"] >= 80]
    print("Dropped empty/too-short descriptions:", before - len(out))

    # 7) Remove duplicates
    out["desc_sig"] = out["description"].str[:250]
    before = len(out)
    out = out.drop_duplicates(subset=["title", "location", "desc_sig"])
    print("Dropped duplicates:", before - len(out))

    # 8) Finalize + Save
    out = out.drop(columns=["desc_len", "desc_sig"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("Clean shape:", out.shape)
    print("Saved to:", OUT_PATH)

    # 9) Small sample (optional)
    print("\nSample rows:")
    print(out.head(3)[["job_id", "title", "location", "work_type", "remote_allowed", "experience_level"]]
          .to_dict(orient="records"))

if __name__ == "__main__":
    main()
