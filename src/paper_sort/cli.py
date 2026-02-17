#!/usr/bin/env python3
"""paper-sort: LLM-powered academic paper organizer."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "api": "ollama",
    "model": "mistral-large-3:675b-cloud",
    "ollama_host": "https://ollama.com",
    "category_guide": {
        "min": 10,
        "max": 15,
    },
    "duplicated_folder": "_duplicated",
    "uncategorized_folder": "_uncategorized",
    "index_file": "index.json",
}


def load_config(root: Path) -> dict:
    config_path = root / "paper_sort_config.json"
    cfg = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def load_index(root: Path, cfg: dict) -> dict:
    """Load index.json. Structure:
    {
        "categories": { "Category Name": ["file1.pdf", ...], ... },
        "files": { "file1.pdf": { "category": "...", "added": "..." } },
        "version": 1
    }
    """
    idx_path = root / cfg["index_file"]
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"categories": {}, "files": {}, "version": 1}


def save_index(root: Path, cfg: dict, index: dict):
    idx_path = root / cfg["index_file"]
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

SPECIAL_DIRS = {"_duplicated", "_uncategorized", ".git", "__pycache__"}


def scan_root_pdfs(root: Path) -> list[str]:
    """Return PDF filenames sitting directly in root (unclassified)."""
    return sorted(
        f.name for f in root.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    )


def scan_all_pdfs(root: Path) -> list[str]:
    """Return all PDF filenames recursively (for --init)."""
    results = []
    for f in root.rglob("*.pdf"):
        rel = f.relative_to(root)
        # skip special dirs
        if any(part in SPECIAL_DIRS for part in rel.parts):
            continue
        results.append(f.name)
    return sorted(results)


def get_existing_categories(root: Path) -> dict[str, list[str]]:
    """Scan existing subfolders and their PDFs."""
    cats = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name not in SPECIAL_DIRS and not d.name.startswith("."):
            pdfs = sorted(f.name for f in d.iterdir() if f.suffix.lower() == ".pdf")
            if pdfs:
                cats[d.name] = pdfs
    return cats


def extract_title(filename: str) -> str:
    """Extract human-readable title from filename like '2025 - Author - Title.pdf'."""
    name = filename.removesuffix(".pdf").removesuffix(".PDF")
    # Try to extract just the title part after second ' - '
    parts = name.split(" - ", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    elif len(parts) == 2:
        return parts[1].strip()
    return name


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], cfg: dict) -> str:
    """Call Ollama and return response text."""
    api = cfg.get("api", "ollama")
    if api != "ollama":
        raise ValueError(f"Unsupported API '{api}'. Set api='ollama' in paper_sort_config.json.")
    return _call_ollama(messages, cfg)


def _call_ollama(messages: list[dict], cfg: dict) -> str:
    from ollama import Client

    client = Client(host=cfg.get("ollama_host", "https://ollama.com"))
    response = client.chat(
        model=cfg.get("model", "mistral-large-3:675b-cloud"),
        messages=messages,
        options={"num_ctx": 32768},
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Classification prompts
# ---------------------------------------------------------------------------

INIT_SYSTEM = """You are an expert academic librarian specializing in neuroscience, psychiatry, and related fields.

Your task: Given a list of academic paper titles, create a well-organized category system and assign each paper to exactly one category.

Rules:
- Create between {min_cats} and {max_cats} categories
- Category names should be concise but descriptive (3-6 words), using title case
- Category names must be valid folder names (no special characters like /, \\, :, etc.)
- Every paper must be assigned to exactly one category
- If a paper fits multiple categories, choose the most specific/relevant one
- Avoid overly granular grouping (do NOT create one-paper categories unless truly unavoidable)
- Prefer cohesive themes and balanced groups across papers
- Create an "_Uncategorized" category only if absolutely necessary

Respond in EXACTLY this JSON format, nothing else:
{{
    "categories": {{
        "Category Name": ["filename1.pdf", "filename2.pdf", ...],
        ...
    }}
}}"""


INCREMENTAL_SYSTEM = """You are an expert academic librarian specializing in neuroscience, psychiatry, and related fields.

Your task: Classify new academic papers into existing categories. You may also create new categories if needed.

Existing categories and their representative papers:
{categories_context}

Rules:
- Assign each new paper to the BEST matching existing category
- If a paper clearly doesn't fit ANY existing category, assign it to a NEW category
- New category names should be concise (2-5 words), title case, valid folder names
- Be conservative about creating new categories — only if truly needed
- For each paper, provide a confidence level: HIGH, MED, or LOW

Respond in EXACTLY this JSON format, nothing else:
{{
    "assignments": [
        {{"file": "filename.pdf", "category": "Category Name", "confidence": "HIGH"}},
        ...
    ],
    "new_categories": ["New Category Name", ...]
}}"""


def build_categories_context(categories: dict[str, list[str]], max_per_cat: int = 8) -> str:
    """Build a context string showing existing categories and sample papers."""
    lines = []
    for cat, files in sorted(categories.items()):
        samples = files[:max_per_cat]
        titles = [extract_title(f) for f in samples]
        more = f" ... and {len(files) - max_per_cat} more" if len(files) > max_per_cat else ""
        lines.append(f"\n## {cat} ({len(files)} papers)")
        for t in titles:
            lines.append(f"  - {t}")
        if more:
            lines.append(f"  {more}")
    return "\n".join(lines)


def _effective_category_bounds(total_papers: int, cfg: dict) -> tuple[int, int]:
    """Compute dynamic category bounds to prevent over-granular clustering."""
    cfg_min = int(cfg["category_guide"]["min"])
    cfg_max = min(int(cfg["category_guide"]["max"]), 15)

    if total_papers <= 6:
        dyn_max = max(2, total_papers)
    else:
        dyn_max = max(3, total_papers // 2)

    max_cats = min(cfg_max, dyn_max, total_papers)
    min_cats = min(cfg_min, max_cats)
    return min_cats, max_cats


def _validate_init_categories(
    categories: dict,
    all_pdfs: list[str],
    min_cats: int,
    max_cats: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if not isinstance(categories, dict) or not categories:
        return False, ["`categories` must be a non-empty JSON object."]

    cat_count = len(categories)
    if cat_count < min_cats or cat_count > max_cats:
        issues.append(f"Category count is {cat_count}; it must be between {min_cats} and {max_cats}.")

    all_set = set(all_pdfs)
    seen_counts: dict[str, int] = {}
    singleton_count = 0

    for cat, files in categories.items():
        if not isinstance(files, list):
            issues.append(f"Category '{cat}' value must be a list of filenames.")
            continue
        if len(files) == 1:
            singleton_count += 1
        for f in files:
            seen_counts[f] = seen_counts.get(f, 0) + 1

    unknown = sorted(f for f in seen_counts.keys() if f not in all_set)
    if unknown:
        issues.append(f"Includes unknown files: {', '.join(unknown[:6])}")

    missing = sorted(all_set - set(seen_counts.keys()))
    if missing:
        issues.append(f"Missing assignments for {len(missing)} file(s).")

    duplicates = sorted(f for f, n in seen_counts.items() if n > 1)
    if duplicates:
        issues.append(f"{len(duplicates)} file(s) assigned to multiple categories.")

    # Guardrail against highly fragmented outputs.
    singleton_limit = max(2, len(all_pdfs) // 4)
    if singleton_count > singleton_limit:
        issues.append(
            f"Too many singleton categories ({singleton_count}); limit is {singleton_limit}. "
            "Merge related singletons into broader themes."
        )

    return (len(issues) == 0), issues


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_init(root: Path, cfg: dict, dry: bool = False):
    """Full initialization: classify all papers from scratch."""
    print("=" * 60)
    print("  paper-sort: Initial full classification")
    print("=" * 60)

    existing_cats = get_existing_categories(root)
    if existing_cats:
        print("\nExisting category folders detected.")
        print("Initial full classification only runs on a fresh root without category subfolders.")
        print("Use default mode (`paper-sort`) for incremental routing.")
        return

    all_pdfs = scan_all_pdfs(root)
    if not all_pdfs:
        print("\nNo PDF files found.")
        return

    print(f"\nFound {len(all_pdfs)} PDF files.")
    min_cats, max_cats = _effective_category_bounds(len(all_pdfs), cfg)
    print(f"Target category range: {min_cats} to {max_cats}")

    # Build prompt
    file_list = "\n".join(all_pdfs)
    system = INIT_SYSTEM.format(
        min_cats=min_cats,
        max_cats=max_cats,
    )

    print("Calling LLM for classification...")
    last_response = ""
    categories = None
    max_attempts = 3
    correction = ""

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"Re-trying classification (attempt {attempt}/{max_attempts})...")

        user_prompt = f"Classify these papers:\n\n{file_list}"
        if correction:
            user_prompt += (
                "\n\nYour previous output violated constraints. "
                "Regenerate the FULL JSON from scratch and satisfy all rules.\n"
                f"{correction}"
            )

        response = call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ], cfg)
        last_response = response

        result = _parse_json_response(response)
        if not result or "categories" not in result:
            correction = "- Response was not valid JSON in the required format."
            continue

        candidate = result["categories"]
        ok, issues = _validate_init_categories(candidate, all_pdfs, min_cats, max_cats)
        if ok:
            categories = candidate
            break
        correction = "\n".join(f"- {issue}" for issue in issues)

    if categories is None:
        print("ERROR: Failed to generate a valid classification after retries.")
        if correction:
            print("Validation issues:")
            print(correction)
        print("Raw response:")
        print(last_response[:2000])
        return

    # Report
    print(f"\nProposed {len(categories)} categories:")
    print("-" * 50)
    total_assigned = 0
    for cat, files in sorted(categories.items()):
        print(f"  📁 {cat} ({len(files)} papers)")
        total_assigned += len(files)

    # Check for unassigned
    assigned_set = set()
    for files in categories.values():
        assigned_set.update(files)
    unassigned = set(all_pdfs) - assigned_set
    if unassigned:
        print(f"\n  ⚠️  {len(unassigned)} papers not assigned by LLM")

    print(f"\nTotal: {total_assigned} assigned / {len(all_pdfs)} total")

    if dry:
        print("\n[DRY RUN] No files moved.")
        return

    # Execute moves
    print("\nMoving files...")
    index = {"categories": {}, "files": {}, "version": 1}
    moved = 0
    errors = 0

    for cat, files in categories.items():
        cat_dir = root / cat
        cat_dir.mkdir(exist_ok=True)
        index["categories"][cat] = []

        for fname in files:
            src = _find_file(root, fname)
            if src is None:
                print(f"  ⚠️  Not found: {fname}")
                errors += 1
                continue

            dst = cat_dir / fname
            if src == dst:
                # Already in correct place
                index["categories"][cat].append(fname)
                index["files"][fname] = {
                    "category": cat,
                    "added": datetime.now().isoformat(),
                }
                continue

            if dst.exists():
                # Handle conflict
                dup_dir = root / cfg["duplicated_folder"]
                dup_dir.mkdir(exist_ok=True)
                shutil.move(str(src), str(dup_dir / fname))
                print(f"  ⚠️  Duplicate → _duplicated/: {fname}")
            else:
                shutil.move(str(src), str(dst))
                moved += 1

            index["categories"][cat].append(fname)
            index["files"][fname] = {
                "category": cat,
                "added": datetime.now().isoformat(),
            }

    # Handle unassigned
    if unassigned:
        uncat_dir = root / cfg["uncategorized_folder"]
        uncat_dir.mkdir(exist_ok=True)
        index["categories"]["_Uncategorized"] = []
        for fname in unassigned:
            src = _find_file(root, fname)
            if src:
                dst = uncat_dir / fname
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                index["categories"]["_Uncategorized"].append(fname)
                index["files"][fname] = {
                    "category": "_Uncategorized",
                    "added": datetime.now().isoformat(),
                }

    save_index(root, cfg, index)

    print(f"\n✅ Done. {moved} files moved, {errors} errors.")
    print(f"   Index saved to {cfg['index_file']}")


def run_incremental(root: Path, cfg: dict, dry: bool = False):
    """Incremental mode: classify new files in root into existing categories."""
    print("=" * 60)
    print("  paper-sort: Incremental classification")
    print("=" * 60)

    index = load_index(root, cfg)
    existing_cats = get_existing_categories(root)

    if not existing_cats:
        print("\nNo existing category folders found.")
        print("Create initial classification by running `paper-sort` in a fresh root directory.")
        return

    # Scan new files in root
    new_files = scan_root_pdfs(root)
    if not new_files:
        print("\nNo new PDF files in root directory.")
        return

    # Check for duplicates by filename against existing library inventory.
    known_files = set(index.get("files", {}).keys())
    for files in existing_cats.values():
        known_files.update(files)

    duplicates = []
    truly_new = []
    for f in new_files:
        if f in known_files:
            duplicates.append(f)
        else:
            truly_new.append(f)

    # Handle duplicates
    if duplicates:
        dup_dir = root / cfg["duplicated_folder"]
        if not dry:
            dup_dir.mkdir(exist_ok=True)
        print(f"\n⚠️  {len(duplicates)} duplicate(s) found:")
        for f in duplicates:
            existing_cat = index.get("files", {}).get(f, {}).get("category", "existing library")
            print(f"  → {f}")
            print(f"    Already in: {existing_cat}/")
            if not dry:
                shutil.move(str(root / f), str(dup_dir / f))

    if not truly_new:
        print("\nNo new files to classify.")
        return

    print(f"\n{len(truly_new)} new file(s) to classify.")

    # Refresh existing categories (include index data for richer context)
    # Merge file-system state with index
    for cat, files in index.get("categories", {}).items():
        if cat not in existing_cats:
            existing_cats[cat] = files
        else:
            existing_cats[cat] = sorted(set(existing_cats[cat]) | set(files))

    context = build_categories_context(existing_cats)
    system = INCREMENTAL_SYSTEM.format(categories_context=context)
    file_list = "\n".join(truly_new)

    print("Calling LLM for classification...")
    response = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Classify these new papers:\n\n{file_list}"},
    ], cfg)

    result = _parse_json_response(response)
    if not result or "assignments" not in result:
        print("ERROR: Failed to parse LLM response.")
        print("Raw response:")
        print(response[:2000])
        return

    assignments = result["assignments"]
    new_categories = result.get("new_categories", [])

    # Report
    print(f"\nClassification results:")
    print("-" * 50)

    by_cat: dict[str, list] = {}
    for a in assignments:
        cat = a["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(a)

    for cat in sorted(by_cat.keys()):
        is_new = cat in new_categories
        marker = " [NEW]" if is_new else ""
        print(f"\n  📁 {cat}{marker}")
        for a in by_cat[cat]:
            conf = a["confidence"]
            icon = {"HIGH": "🟢", "MED": "🟡", "LOW": "🔴"}.get(conf, "⚪")
            title = extract_title(a["file"])
            print(f"    {icon} {title}")

    if dry:
        print("\n[DRY RUN] No files moved.")
        return

    # Execute moves
    print("\nMoving files...")
    moved = 0
    errors = 0

    for a in assignments:
        fname = a["file"]
        cat = a["category"]
        src = root / fname

        if not src.exists():
            print(f"  ⚠️  Not found: {fname}")
            errors += 1
            continue

        cat_dir = root / cat
        cat_dir.mkdir(exist_ok=True)
        dst = cat_dir / fname

        if dst.exists():
            dup_dir = root / cfg["duplicated_folder"]
            dup_dir.mkdir(exist_ok=True)
            shutil.move(str(src), str(dup_dir / fname))
            print(f"  ⚠️  Already exists in {cat}/ → _duplicated/: {fname}")
        else:
            shutil.move(str(src), str(dst))
            moved += 1

        # Update index
        if cat not in index["categories"]:
            index["categories"][cat] = []
        index["categories"][cat].append(fname)
        index["files"][fname] = {
            "category": cat,
            "added": datetime.now().isoformat(),
            "confidence": a.get("confidence", "N/A"),
        }

    save_index(root, cfg, index)

    print(f"\n✅ Done. {moved} files moved, {errors} errors.")
    if new_categories:
        print(f"   New categories created: {', '.join(new_categories)}")
    print(f"   Index saved to {cfg['index_file']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON block in markdown
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # Try to find raw JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None


def _find_file(root: Path, fname: str) -> Path | None:
    """Find a file anywhere under root."""
    # Check root first
    p = root / fname
    if p.exists():
        return p
    # Search subdirs
    for f in root.rglob(fname):
        if any(part in SPECIAL_DIRS for part in f.relative_to(root).parts):
            continue
        return f
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="paper-sort: LLM-powered academic paper organizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-sort                    # Auto mode in current directory
  paper-sort ~/Papers           # Auto mode in a specific directory
  paper-sort --dry ~/Papers     # Preview without moving files
        """,
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Dry run: show classification results without moving files"
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="Root directory containing papers (default: current directory)"
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    cfg = load_config(root)

    if get_existing_categories(root):
        run_incremental(root, cfg, dry=args.dry)
    else:
        run_init(root, cfg, dry=args.dry)


if __name__ == "__main__":
    main()
