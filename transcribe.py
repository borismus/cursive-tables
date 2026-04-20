#!/usr/bin/env python3
"""Transcribe archival record images with Gemini and score against ground truth.

Two modes:
  --format json  (default): ask Gemini for a JSON array via response_schema; score
                            per-field on concatenated values (no row alignment).
  --format tsv            : ask Gemini for TSV; parse to rows; Hungarian-align rows
                            to GT; score per-field on aligned rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.optimize import linear_sum_assignment

PROMPT_JSON = """This is an archival record from a Belarusian archive, in Russian \
with pre-reform orthography (letters ѣ, і, ъ, ѳ). The page contains a table with the \
following columns (key — description):

{schema_block}

Transcribe every data row of the table verbatim. Preserve original spelling exactly — \
do not modernize.

Output format: a JSON array. Each element is an object representing one row of the \
table, with keys exactly matching the schema above. For blank cells, use an empty \
string "".

Paired male/female columns: if some columns describe male attributes and others \
describe female attributes of the same family/household row, keep the paired male \
and female entries on the SAME object — do not split a paired entry into two separate \
objects.

Group-identifier columns: if a column holds an identifier that groups multiple \
consecutive rows (e.g. a family number / № семьи / № семейства, or any column whose \
value is only written once but applies to a block of rows below it), REPEAT that \
identifier on EVERY object in the block — never leave it as "" on a continuation row.

Single-line cells: every cell value must be a single line with NO newline characters. \
If you see two vertically adjacent values in a column, they belong to two DIFFERENT \
rows — emit two separate objects, not one object with a newline-joined string.

Ditto marks: if a cell contains ditto marks (,, or " meaning "same as above"), \
expand them to the actual repeated value.

Skip the header row. Return only the JSON array, no markdown fences, no commentary."""

PROMPT_TSV = """This is an archival record from a Belarusian archive, in Russian \
with pre-reform orthography (letters ѣ, і, ъ, ѳ). The page contains a table with \
exactly {ncols} columns in this order:

{schema_block}

Transcribe every data row of the table verbatim. Preserve original spelling exactly — \
do not modernize.

STRICT OUTPUT FORMAT — FOLLOW EXACTLY:
- Every line of output must contain exactly {ncols} fields separated by tab \
characters (exactly {nsep} TAB characters per line).
- If a cell is blank, still emit it as an empty field — keep the tab separators.
- Single-line cells only: NO newline characters inside any cell. Two vertically \
adjacent values in one column belong to TWO DIFFERENT rows — emit two separate lines.
- Paired male/female columns: if some columns describe male and others female \
attributes of the same family row, keep them on the SAME line.
- Group-identifier columns: repeat group identifiers (family number, etc.) on every \
continuation line.
- Ditto marks: expand ,, or " to the actual repeated value.
- Skip the header row. Output only the TSV data rows — no commentary, no fences."""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def parse_schema(text: str) -> list[dict]:
    cols = []
    for line in text.splitlines():
        m = re.match(r"^\s*Col\s*(\d+)\s*\[([a-zA-Z_][a-zA-Z0-9_]*)\]\s*:\s*(.*)$", line)
        if m:
            cols.append({"num": int(m.group(1)), "key": m.group(2),
                         "description": m.group(3).strip()})
    if not cols:
        sys.exit("schema has no 'Col N [key]: description' lines")
    keys = [c["key"] for c in cols]
    if len(set(keys)) != len(keys):
        sys.exit(f"schema has duplicate keys: {keys}")
    return cols


def schema_block(cols: list[dict]) -> str:
    return "\n".join(f"- {c['key']} — {c['description']}" for c in cols)


def response_schema(cols: list[dict]) -> types.Schema:
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={c["key"]: types.Schema(type=types.Type.STRING) for c in cols},
            required=[c["key"] for c in cols],
            property_ordering=[c["key"] for c in cols],
        ),
    )


def parse_tsv(text: str, keys: list[str]) -> list[dict]:
    rows = []
    n = len(keys)
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) < n:
            cells = cells + [""] * (n - len(cells))
        elif len(cells) > n:
            cells = cells[:n]
        rows.append(dict(zip(keys, (c.strip() for c in cells))))
    return rows


def resolve_images(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    matches = [Path(m) for m in glob.glob(pattern)]
    return sorted(m for m in matches if m.is_file() and m.suffix.lower() in IMAGE_EXTS)


def resolve_schema(image: Path, fallback: str | None) -> tuple[list[dict], str]:
    sibling = image.parent / f"{image.stem}.schema.txt"
    if sibling.exists():
        text, path = sibling.read_text(encoding="utf-8"), str(sibling)
    elif fallback:
        text, path = Path(fallback).read_text(encoding="utf-8"), fallback
    else:
        sys.exit(f"no schema for {image}: expected {sibling} or pass --schema <path>")
    return parse_schema(text), path


def resolve_gt(image: Path, gt_dir: str | None) -> list[dict] | None:
    candidate = (Path(gt_dir) if gt_dir else image.parent) / f"{image.stem}.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _stream_and_collect(stream) -> str:
    start = time.monotonic()
    stop = threading.Event()

    def tick() -> None:
        while not stop.wait(1.0):
            print(f"\rwaiting for first token... {time.monotonic() - start:.0f}s",
                  end="", flush=True)

    threading.Thread(target=tick, daemon=True).start()
    chunks: list[str] = []
    first = True
    for chunk in stream:
        if chunk.text:
            if first:
                stop.set()
                print("\r" + " " * 50 + "\r", end="", flush=True)
                first = False
            chunks.append(chunk.text)
            print(chunk.text, end="", flush=True)
    stop.set()
    print(f"\n[{time.monotonic() - start:.1f}s total]", flush=True)
    return "".join(chunks).strip()


def transcribe_json(client: genai.Client, model: str, image: Path,
                    cols: list[dict]) -> tuple[str, list[dict]]:
    data = image.read_bytes()
    mime = MIME[image.suffix.lower()]
    prompt = PROMPT_JSON.format(schema_block=schema_block(cols))
    print("--- prediction (streaming JSON) ---", flush=True)
    stream = client.models.generate_content_stream(
        model=model,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema(cols),
        ),
    )
    raw = _stream_and_collect(stream)
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"invalid JSON from model: {e}\nraw:\n{raw[:1000]}")
    if not isinstance(rows, list):
        sys.exit(f"expected JSON array, got {type(rows).__name__}")
    return raw, rows


def transcribe_tsv(client: genai.Client, model: str, image: Path,
                   cols: list[dict]) -> tuple[str, list[dict]]:
    data = image.read_bytes()
    mime = MIME[image.suffix.lower()]
    ncols = len(cols)
    prompt = PROMPT_TSV.format(
        schema_block=schema_block(cols), ncols=ncols, nsep=ncols - 1,
    )
    print("--- prediction (streaming TSV) ---", flush=True)
    stream = client.models.generate_content_stream(
        model=model,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
    )
    raw = _stream_and_collect(stream)
    keys = [c["key"] for c in cols]
    rows = parse_tsv(raw, keys)
    bad = [(i + 1, line.count("\t") + 1)
           for i, line in enumerate(raw.splitlines())
           if line.strip() and line.count("\t") + 1 != ncols]
    if bad:
        preview = ", ".join(f"L{n}={c}" for n, c in bad[:5])
        print(f"[warn] {len(bad)} line(s) with wrong field count (expected {ncols}): {preview}",
              flush=True)
    return raw, rows


def normalize_for_score(s: str) -> str:
    """Lowercase (NFC) and strip Unicode punctuation; collapse whitespace."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return " ".join(s.split())


def _row_signature(r: dict, keys: list[str]) -> str:
    return normalize_for_score(" ".join(str(r.get(k, "") or "") for k in keys))


def align_hungarian(ref: list[dict], hyp: list[dict],
                    keys: list[str]) -> tuple[list[dict], dict]:
    """Reorder hyp to best match ref via Hungarian on row-level CER.
    Returns (aligned_hyp_same_length_as_ref, stats_including_unmatched)."""
    empty = {k: "" for k in keys}
    n_ref, n_hyp = len(ref), len(hyp)
    if n_ref == 0 or n_hyp == 0:
        return hyp[:], {"matched_pairs": [], "unmatched_hyp": list(range(n_hyp)),
                        "unmatched_ref": list(range(n_ref))}

    ref_sigs = [_row_signature(r, keys) for r in ref]
    hyp_sigs = [_row_signature(h, keys) for h in hyp]

    # Build square cost matrix with dummy rows/cols at cost 1.0 (full mismatch).
    n = max(n_ref, n_hyp)
    cost = np.ones((n, n), dtype=float)
    for i in range(n_ref):
        for j in range(n_hyp):
            r, h = ref_sigs[i], hyp_sigs[j]
            if not r and not h:
                cost[i][j] = 0.0
            else:
                cost[i][j] = float(jiwer.cer(r or " ", h or " "))
                if cost[i][j] > 1.0:
                    cost[i][j] = 1.0

    row_ind, col_ind = linear_sum_assignment(cost)

    aligned = [dict(empty) for _ in range(n_ref)]
    matched_pairs: list[tuple[int, int, float]] = []
    matched_hyp_idx: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        if i < n_ref and j < n_hyp and cost[i][j] < 1.0:
            aligned[i] = hyp[j]
            matched_hyp_idx.add(j)
            matched_pairs.append((int(i), int(j), float(cost[i][j])))

    unmatched_hyp = [j for j in range(n_hyp) if j not in matched_hyp_idx]
    unmatched_ref = [i for i in range(n_ref) if aligned[i] == empty]

    return aligned, {
        "matched_pairs": matched_pairs,
        "unmatched_hyp": unmatched_hyp,
        "unmatched_ref": unmatched_ref,
    }


def _score_str(ref: str, hyp: str) -> tuple[float, float]:
    if not ref and not hyp:
        return 0.0, 0.0
    return jiwer.cer(ref or " ", hyp or " "), jiwer.wer(ref or " ", hyp or " ")


def word_diffs(ref: str, hyp: str, max_shown: int = 15) -> list[str]:
    if not ref.strip() and not hyp.strip():
        return []
    out = jiwer.process_words(ref or " ", hyp or " ")
    ref_words = out.references[0]
    hyp_words = out.hypotheses[0]
    diffs: list[str] = []
    for c in out.alignments[0]:
        t = c.type if isinstance(c.type, str) else c.type.name.lower()
        if t in ("equal", "e"):
            continue
        r = " ".join(ref_words[c.ref_start_idx:c.ref_end_idx])
        h = " ".join(hyp_words[c.hyp_start_idx:c.hyp_end_idx])
        if t.startswith("sub"):
            diffs.append(f"{r!r} → {h!r}")
        elif t.startswith("ins"):
            diffs.append(f"+{h!r}")
        elif t.startswith("del"):
            diffs.append(f"-{r!r}")
    if len(diffs) > max_shown:
        diffs = diffs[:max_shown] + [f"… ({len(diffs) - max_shown} more)"]
    return diffs


def score_structured(ref: list[dict], hyp: list[dict], keys: list[str]) -> dict:
    """Per-field CER/WER on concatenated values. Case/punctuation-normalized."""
    per_field = {}
    for k in keys:
        ref_raw = " ".join(str(r.get(k, "") or "").strip() for r in ref).strip()
        hyp_raw = " ".join(str(h.get(k, "") or "").strip() for h in hyp).strip()
        ref_n = normalize_for_score(ref_raw)
        hyp_n = normalize_for_score(hyp_raw)
        cer, wer = _score_str(ref_n, hyp_n)
        per_field[k] = {"cer": cer, "wer": wer, "diffs": word_diffs(ref_n, hyp_n)}

    flat_ref = " ".join(str(r.get(k, "") or "").strip() for r in ref for k in keys).strip()
    flat_hyp = " ".join(str(h.get(k, "") or "").strip() for h in hyp for k in keys).strip()
    cer, wer = _score_str(normalize_for_score(flat_ref), normalize_for_score(flat_hyp))

    return {
        "per_field": per_field,
        "overall": {"cer": cer, "wer": wer},
        "n_rows_ref": len(ref),
        "n_rows_hyp": len(hyp),
    }


def fmt_scores(s: dict) -> str:
    width = max(len(k) for k in s["per_field"])
    lines = [f"rows: ref={s['n_rows_ref']}  hyp={s['n_rows_hyp']}", "per-field:"]
    for k, v in s["per_field"].items():
        lines.append(f"  {k:<{width}}  CER={v['cer']:.4f}  WER={v['wer']:.4f}")
        for d in v.get("diffs", []):
            lines.append(f"    {d}")
    o = s["overall"]
    lines.append(f"overall: CER={o['cer']:.4f}  WER={o['wer']:.4f}")
    return "\n".join(lines)


def fmt_alignment(stats: dict, ref: list[dict], hyp: list[dict], keys: list[str]) -> str:
    lines = [f"alignment: matched={len(stats['matched_pairs'])}  "
             f"unmatched_hyp={len(stats['unmatched_hyp'])}  "
             f"unmatched_ref={len(stats['unmatched_ref'])}"]
    if stats["unmatched_hyp"]:
        lines.append("  unmatched hyp rows (hallucinated/fragment):")
        for j in stats["unmatched_hyp"][:5]:
            lines.append(f"    hyp[{j}]: {_row_signature(hyp[j], keys)[:100]}")
    if stats["unmatched_ref"]:
        lines.append("  unmatched ref rows (missed):")
        for i in stats["unmatched_ref"][:5]:
            lines.append(f"    ref[{i}]: {_row_signature(ref[i], keys)[:100]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="examples/")
    ap.add_argument("--schema", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--format", choices=["json", "tsv"], default="json",
                    help="json = response_schema + no alignment; tsv = TSV + Hungarian alignment")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set (check .env)")

    images = resolve_images(args.images)
    if not images:
        sys.exit(f"no images found at {args.images}")

    client = genai.Client(api_key=api_key)
    records: list[dict] = []
    all_refs: list[list[dict]] = []
    all_hyps: list[list[dict]] = []
    all_keys: list[str] | None = None

    for img in images:
        cols, schema_path = resolve_schema(img, args.schema)
        keys = [c["key"] for c in cols]
        print(f"\n=== {img.name} ===", flush=True)
        print(f"schema: {schema_path} ({len(cols)} cols)", flush=True)
        print(f"format: {args.format}", flush=True)

        if args.format == "json":
            raw, pred_rows = transcribe_json(client, args.model, img, cols)
        else:
            raw, pred_rows = transcribe_tsv(client, args.model, img, cols)
        print(f"parsed {len(pred_rows)} rows", flush=True)

        gt_rows = resolve_gt(img, args.gt_dir)
        if gt_rows is None:
            print("--- no ground truth (expected sibling <stem>.json) ---")
            records.append({"image": str(img), "schema": schema_path,
                            "format": args.format, "prediction": pred_rows,
                            "ground_truth": None, "scores": None, "scored": False})
            continue

        aligned_pred = pred_rows
        align_stats = None
        if args.format == "tsv":
            aligned_pred, align_stats = align_hungarian(gt_rows, pred_rows, keys)
            print(fmt_alignment(align_stats, gt_rows, pred_rows, keys))

        s = score_structured(gt_rows, aligned_pred, keys)
        print(fmt_scores(s))

        all_refs.append(gt_rows)
        all_hyps.append(aligned_pred)
        if all_keys is None:
            all_keys = keys
        records.append({"image": str(img), "schema": schema_path, "format": args.format,
                        "prediction": pred_rows, "aligned_prediction": aligned_pred,
                        "alignment": align_stats, "ground_truth": gt_rows,
                        "scores": s, "scored": True})

    if all_refs and all_keys is not None:
        concat_ref = [r for rows in all_refs for r in rows]
        concat_hyp = [h for rows in all_hyps for h in rows]
        agg = score_structured(concat_ref, concat_hyp, all_keys)
        print(f"\n=== aggregate ({len(all_refs)}/{len(images)} scored) ===")
        print(fmt_scores(agg))

    if args.out:
        Path(args.out).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
