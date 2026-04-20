# CLAUDE.md

## Prompts are load-bearing

`PROMPT_JSON` and `PROMPT_TSV` in `transcribe.py` encode hard-won rules — don't casually rewrite them. In particular, preserve:

- pre-reform orthography instruction (no modernization of ѣ, і, ъ, ѳ)
- paired male/female columns stay on one row
- group-identifier columns (e.g. family №) repeat on every continuation row
- single-line cells only — vertically adjacent values are separate rows
- ditto marks (`,,`, `"`) expand to the repeated value
- skip the header row

If you change a rule, change it in both prompts.

## Scoring is normalization-sensitive

`normalize_for_score` does NFC → lowercase → strip Unicode punctuation → collapse whitespace, and runs on both ref and hyp before CER/WER. Don't score raw strings; don't add normalization steps that aren't symmetric.

`jiwer.cer("", "")` throws — pad empties with `" "`. The existing code already does this; keep it.

## Schema format is strict

`parse_schema` only accepts `Col N [key]: description`. Keys must match `[a-zA-Z_][a-zA-Z0-9_]*` and be unique. Don't loosen the regex without updating ground-truth files in `examples/`.
