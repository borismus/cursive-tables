# cursive-tables

Schema-primed OCR and scoring for handwritten tabular archival records, backed by Gemini. Built for Belarusian-archive revision lists (посемейные списки) in pre-reform Russian orthography, but works on any tabular handwriting where you can describe the columns.

## Setup

Requires Python 3.10+.

```
uv sync
cp .env.example .env        # then set GEMINI_API_KEY
```

## File conventions

For an image `examples/foo.jpg`, the script looks for siblings:

- `examples/foo.schema.txt` — column definitions (required; or pass `--schema`)
- `examples/foo.json` — ground-truth rows for scoring (optional; or pass `--gt-dir`)

Schema format — one line per column:

```
Col 1 [family_seq]: № семейства по порядку (family number in order)
Col 2 [male_name]: Прозваніе, имя и отчество лицъ мужскаго пола (male names)
...
```

The bracketed key becomes the JSON key in the prediction and GT. Keys must be unique.

Ground truth is a JSON array of objects keyed by those same column keys.

## Usage

```
# JSON mode (default): response_schema, per-field CER/WER on concatenated values
uv run transcribe.py --images examples/

# TSV mode: model emits TSV, rows Hungarian-aligned to GT before scoring
uv run transcribe.py --images examples/ --format tsv

# Single file, custom model, write full records
uv run transcribe.py --images examples/NHABGrodno_132_1_2_sample.jpg \
  --model gemini-3.1-pro-preview --out results.json
```

Flags:

- `--images` — file, directory, or glob (default `examples/`)
- `--schema` — fallback schema path if no sibling `.schema.txt`
- `--gt-dir` — look for `<stem>.json` ground truth here instead of next to the image
- `--model` — Gemini model ID (default `gemini-3.1-pro-preview`)
- `--format` — `json` (no row alignment) or `tsv` (Hungarian row alignment)
- `--out` — write per-image predictions, alignments, and scores as JSON

## Modes

**JSON** uses Gemini's structured-output `response_schema`. Scoring concatenates all values for each column across rows — robust to row-count drift, blind to row ordering.

**TSV** asks for tab-separated rows, parses them, and aligns to GT via Hungarian matching on row-level CER (dummy cost 1.0 for unmatched). Gives you per-row alignment stats (matched / hallucinated / missed) in addition to per-field scores.

## Scoring

CER/WER via `jiwer` on NFC-lowercased, punctuation-stripped, whitespace-collapsed strings. Reported per-field, overall-per-image, and aggregated across all scored images.
