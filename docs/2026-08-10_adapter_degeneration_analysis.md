# Adapter degeneration analysis — `checkpoint-23500` (2026-08-10)

Source: `evaluation/report_finetune_tg_100826.html`, 203 test segments, base
`google/translategemma-12b-it` vs LoRA adapter
`/workspace/translategemma-farsi-science_100826/sft/checkpoint-23500`.
No MetricX or COMET scores were computed in that run, so every number below is
measured directly from the generated text. Training evidence comes from
`logs/20260809_154522_translategemma-farsi-science.log`.

## What the training run actually was

| | |
|---|---|
| Train rows | 2,735,122 → 2,734,476 after dropping 646 with no target tokens |
| Truncated at `max_length=2048` | **105,567 (3.86%)** |
| Packed blocks | 557,753 |
| Trainable params | 65,470,464 / 12,252,795,504 (0.53%) |
| Distributed batch | per_device 6 × world 8 × accum 1 = 48 |
| Planned steps | 34,860 (3 epochs) |
| **Reached** | **step 23,640, epoch 2.03 — log ends mid-run, no `Train end`** |
| Train loss | 0.83 → 0.26 by step 100 → 0.084 at step 23,600, flat and stable |
| Grad norm | 2.5 at start → 0.11–0.13 throughout, no spikes |
| eval_loss | monotone 0.1880 (step 500) → **0.1092 (step 23,000)** → 0.1109 (step 23,500) |
| Template/token boundary warnings | **0** across all 2.73M rows |

Two things follow immediately. The evaluated `checkpoint-23500` is an
**intermediate checkpoint from an unfinished run**, not `sft_final` and not the
best checkpoint — step 23,000 scored better, and `load_best_model_at_end` never
executed because training never ended. And the run was **healthy**: smooth loss,
stable gradient norms, 45 consecutive eval improvements, one 0.0016 uptick at
the very end. Nothing here looks like divergence or collapse.

## Outcome

**Resolved by the decoder fix. No retraining.** Re-evaluating the *unchanged*
`checkpoint-23500` with Fixes 1–3 applied (`evaluation_100826_23500_v2`):

| | base | adapter (before) | adapter (after) |
|---|---:|---:|---:|
| Clean rows | 91.6% | 12.8% | **94.6%** |
| Whitespace flood | 0 | 142 (69.95%) | **0** |
| Loop | 7 (3.45%) | 35 (17.24%) | **6 (2.96%)** |
| Boilerplate leak | 0 | 4 | **0** |
| Length blowup | 4 | 28 | **0** |
| Mean output chars | 466 | 1010 | **423** |
| Max trailing whitespace | 1 | 511 | **0** |
| Similarity to reference | 0.530 | — | **0.860** |
| len(output)/len(reference) | 1.263 | 7.70 | **1.005** |

The adapter now beats the base model on every axis, and closer to the reference
on **193 of 203 rows (95.1%)**. Whole-text similarity to the reference went from
0.530 (base) to 0.860 — the fine-tune was always this good; 85% of its output
was being buried under unstopped decoding.

All 6 residual loops are **shared with the base model**, and in every one the
*source segment itself* contains the repeated span (repeat factor 3–11). There
are zero adapter-only loops. Those are PDF-extraction artefacts in the test set
being translated faithfully, not a model defect.

The base model's numbers are unchanged (466 vs 465 mean chars, 91.6% vs 92.1%
clean), confirming that keeping it on `add_generation_prompt=True` preserved the
baseline rather than moving it.

Everything below is the investigation that produced that fix.

## Headline (pre-fix measurements)

**The fine-tune improved translation and destroyed termination.**

| Measure | base | adapter |
|---|---:|---:|
| Rows with no decoding defect | 196 / 203 (**96.6%**) | 30 / 203 (**14.8%**) |
| Mean output length (chars) | 465 | 1010 |
| Mean output length, trailing whitespace trimmed | 464 | 704 |
| Rows ending in >20 chars of whitespace | 0 (0%) | **142 (70.0%)** |
| Mean trailing whitespace per row | 0 chars | 306 chars |
| Rows where a 6-gram repeats ≥3× | 7 (3.4%) | 35 (17.2%) |
| Rows with leaked MT boilerplate | 0 | 4 (2.0%) |
| len(output)/len(reference), mean | 1.26 | **7.70** |
| First-120-character similarity to the reference | 0.700 | **0.797** |

That last row matters: **the adapter's opening is measurably closer to the
reference than the base model's.** Terminology, transliteration of author names,
Eastern-Arabic numerals, and register all moved toward the reference. The
adaptation worked. What broke is the stop condition, and it broke so badly that
it swamps the gain on any corpus-level metric.

## The three symptoms are one bug

### 1. Whitespace flood (70% of rows)

62,124 of the 62,125 trailing characters across the adapter's outputs are `\n`.
One is a space. Not "some texts repeated" — a correct, complete translation
followed by up to 511 newlines, running until `max_new_tokens: 1024` is
exhausted.

Report row 39 (`Stress Detection on WESAD Dataset`, 33 chars):

```
base    : تشخیص استرس در مجموعه داده WESAD.                  (33 chars)
adapter : تشخیص استرس در مجموعه داده WESAD + 506 newlines    (539 chars)
```

### 2. Turn restart, read as "repetition" (17% of rows)

The loops are not stuttered phrases. They are the **entire translation emitted
again from the top**, separated by `\n\n`. Report row 36:

```
آیا همه بیماران LQT3 به یک ICD نیاز دارند: درست یا نادرست؟ ... 612009
\n\n
یک ICD برای همه بیماران LQT3 ضروری است: درست یا نادرست؟ ... 612009   ← paraphrase
\n
آیا همه بیماران LQT3 به یک ICD نیاز دارند: ...                        ← then identical, ×21
```

The second attempt is a *different* translation of the same source, then it
locks into an identical repeat. That is a model starting a fresh conversational
turn, not a model stuck in an n-gram loop.

Six of the 35 loop rows (indices 29, 32, 67, 85, 141, 166) also loop in the
base model, and their **source text already contains the duplicated span** —
those are PDF-extraction artefacts in the test set, not adapter regressions.
The adapter-specific loop count is ~29 rows (14%).

### 3. Leaked training boilerplate (4 rows)

Rows 10, 20, 98, 155 emit, repeatedly:

- `ترجمه شده توسط هوش مصنوعی` — "Translated by AI"
- `لطفاً توجه داشته باشید که این ترجمه ممکن است به طور کامل دقیق نباشد و نیاز به ویرایش بیشتر دارد` — "Please note this translation may not be fully accurate and needs further editing"

The base model never produces these. They are footers from whatever pipeline
produced the Farsi side of the corpus, and they are memorised. This is a data
contamination finding independent of everything else.

### What the user called "extra text"

Report row 19: source is `. S K Mckenzie, ...`. The adapter reproduces the
leading `. ` from the source; the base drops it. Report row 25: the source ends
`...regression analysis was used.P151`; the reference silently drops `P151`, the
base renders it as `(P151)`, the adapter emits it on its own line. Both are
faithful-to-a-fault behaviour on dirty source, not hallucination — the adapter
learned to copy segmentation noise the reference had cleaned up. Real, but a
second-order issue next to termination.

## Audit results (2026-08-10, production host)

All three audits ran. `logs/training_termination_audit.json`,
`logs/sft_corpus_audit.json`, `logs/degeneration_audit.json`.

| Hypothesis | Verdict |
|---|---|
| **A** Decoder deaf to `<end_of_turn>` | **CONFIRMED — this is the bug** |
| **B** Train/eval prompt mismatch | **CONFIRMED**, every row, different shape than predicted |
| **C** Stop token lost to truncation | Confirmed at 3.69%, secondary |
| **D** Trailing whitespace in targets | **REFUTED — zero rows** |
| **E** Corpus boilerplate contamination | Confirmed, 0.39% gross, ~0.17% unambiguous |
| **F** Over-training | Ruled out (training log) |

The decisive line:

```json
"eos_token_ids_used_by_evaluation": [1],           // <eos>
"eos_token_ids_in_generation_config_json": [1, 106],
"missing_from_evaluation": [106],                   // <end_of_turn>
"end_of_turn_is_a_stop_token": false
```

And the other half of the proof, from the same report — the 50,000 most common
rendered training tails:

```
[".", "<end_of_turn>", "\n"]   20439
["▁.", "<end_of_turn>", "\n"]   4603
[").", "<end_of_turn>", "\n"]   2221
```

Every training target ends in token 106. The decoder stops on token 1 only.
The adapter emits its stop token 203 times out of 203 and is not stopped once.

`degeneration_audit.json` reproduces the failure rates independently of the HTML
report: adapter clean **12.8%** (whitespace_flood 69.95%, loop 17.24%,
length_blowup 13.79%, near_budget 13.30%, boilerplate 1.97%) against base clean
**92.1%**.

## Root cause: ranked

### A. The decoder is not listening for the token the adapter learned to emit — **CONFIRMED**

`evaluate_translations.py:167` builds the generation config from the **model
config**:

```python
"generation_config": make_deterministic_generation_config(model_config, processor)
```

and `train.py:162`:

```python
generation_config = GenerationConfig.from_model_config(model_config)
...
if generation_config.eos_token_id is None:
    generation_config.eos_token_id = tokenizer.eos_token_id
```

`GenerationConfig.from_model_config()` reads `config.json`. It **never reads
`generation_config.json`**, which is the file Gemma-family chat checkpoints use
to list the turn ender `<end_of_turn>` (id 106) alongside `<eos>` (id 1). And
the `if ... is None` guard cannot repair it: `eos_token_id` is not `None`, it is
merely *incomplete*, so the fallback never fires.

Meanwhile training targets are rendered with
`apply_chat_template(..., add_generation_prompt=False)` (`train.py:228`), which
terminates every assistant turn with `<end_of_turn>\n`. Those tokens are inside
the label span (`train.py:268`), so **the adapter is explicitly trained to emit
`<end_of_turn>`**.

Put together: the adapter emits its stop token, `generate()` does not recognise
it as a stop, and decoding continues past the turn boundary. What follows a
`<end_of_turn>` in the training distribution is `\n` — hence the newline flood —
and eventually a new turn, hence the full re-translation. `batch_decode(...,
skip_special_tokens=True)` then strips the `<end_of_turn>` markers out of the
report, which is why this reads as "trailing newlines" instead of "unstopped
turns".

This single mechanism explains all three symptoms **and** explains why the base
model is unaffected: base TranslateGemma ends its output with a token that *is*
in the truncated stop set, so it was never exposed to the gap.

The training log makes this the leading hypothesis by elimination. 2,628,909
examples (96.14%) carried an untruncated `<end_of_turn>` inside their label span,
and the model saw them **twice over** before step 23,500. A LoRA adapter given
5.2 million untruncated stop-token demonstrations, converging smoothly to
train loss 0.084 with no gradient instability, does not fail to *emit* the stop
token. It is overwhelmingly more likely that it emits it and is not heard.

Confirmed exactly as described: `<eos>` (1) is the whole stop set in use,
`generation_config.json` publishes `[1, 106]`, `<end_of_turn>` (106) is missing,
and every rendered training target ends in 106.

### B. Train/inference prompt mismatch — **CONFIRMED**, and not where I expected

`train.py:229-239` derives the prompt by rendering the *training* template with
a marker, with an explicit comment that TranslateGemma's generation prompt "is
not guaranteed to be a literal token prefix of its completed assistant turn".
Evaluation uses `add_generation_prompt=True` (`evaluate_translations.py:194`).
If those two strings differ by even one token, the adapter is queried
off-distribution at inference, which is a classic degeneration trigger.

The log confirms the *training-side* boundary is sound: zero
"Template/token boundary mismatch" warnings across 2.73M rows, meaning the
marker-rendered prompt is always an exact token prefix of the full rendering.
That is a different comparison from the one that matters here.

The audit compared the pair that matters, and **200 of 200 rows mismatch**, with
`same_token_prefix: false`. The difference is entirely in the assistant turn
header:

```
training : ... <end_of_turn> \n <start_of_turn> model "\n\n" "▁▁▁▁▁▁▁▁"
evaluation: ... <end_of_turn> \n <start_of_turn> model "\n"
```

TranslateGemma's chat template opens the assistant turn with a blank line and
**eight literal spaces** — unsuppressed Jinja block indentation leaking into the
rendering. `add_generation_prompt=True` does not emit them. So the adapter was
trained on 2.7M examples whose target begins after `model\n\n␣␣␣␣␣␣␣␣`, and is
queried at inference after `model\n`. Two tokens of drift, on every row.

Note this whitespace sits in the *prompt* span, masked to `-100`, so it is not
a label defect — it is a conditioning defect. Its observable effect turned out
to be small: no adapter output begins with whitespace (0 of 203, same as base),
so the model recovers on the first token. But it means every measurement in the
2026-08-10 run was taken off-distribution, and it is free to fix.

### C. Stop token lost to truncation — confirmed at 3.86%, secondary

`train.py:259` truncates with `full_ids[:max_length]` at `max_length: 2048`.
Any example longer than that loses its tail, and the tail is where
`<end_of_turn>` lives.

The log quantifies it: **105,567 rows (3.86%)** were truncated. I previously
called this "probably 0%" from benchmark logs on 1000-row samples; that was
wrong, and it is a real defect worth fixing.

But it does not explain the failure, and the same number is why. Those rows do
not teach "continue forever" — under BFD packing with position-id resets, a
truncated document simply ends and the next document's prompt tokens are masked
to `-100`. They *withhold* one stop-token example each. Against 2.63M rows that
supply one, at 2+ epochs, a 3.86% dilution cannot produce a 70% failure rate.
Fix it, but do not expect it to be the cure. The audit puts the exact figures on
record over a 50,000-row sample: `truncated` 3.70%,
`stop_token_lost_to_truncation` **3.694%**, `labels_without_stop_token` 3.67%,
`no_target_tokens` 0.024%. So essentially every truncated example loses its
terminator — the truncation is never harmlessly landing inside trailing
template tokens.

### D. Trailing whitespace in the training targets — **REFUTED**

My highest-prior data hypothesis, and it is wrong. Across all 2,735,122 rows of
`data/splits/train.jsonl`: **zero** targets with trailing whitespace, zero with
leading whitespace, and an empty trailing-whitespace histogram. The targets are
already clean. Nothing in the corpus teaches the newline flood — it is entirely
a decode-side artefact of (A), the model filling the budget after a stop token
that was ignored.

### E. Data contamination — **CONFIRMED**, with a caveat on the count

10,800 rows (0.39%) match a boilerplate pattern. Breakdown:

| Pattern | Rows | Read as |
|---|---:|---|
| `هوش مصنوعی` | 6,248 | **mostly false positives** — "artificial intelligence" is ordinary vocabulary in a science corpus |
| `کلیه حقوق محفوظ است` | 4,358 | "all rights reserved" — real footer |
| `ترجمه شده توسط` | 157 | "translated by" — real, and the exact string the adapter memorised |
| `Google Translate` | 16 | real |
| `ترجمه:` at line end | 11 | real |
| `All rights reserved` | 6 | real |
| `machine-translat` | 2 | real |
| `Downloaded from` | 2 | real |

So ~4,550 rows (0.17%) are unambiguous contamination, and the
`هوش مصنوعی` bucket needs the co-occurrence filter (`ترجمه … هوش مصنوعی` on the
same line) before deletion, or the corpus loses legitimate AI-domain content.
0.17% was enough for the model to memorise and reproduce the footer.

### E2. Targets that already loop — **new finding, 2.67%**

Not in my original list, and it matters: **73,110 rows (2.67%)** have a target
whose most common 6-gram repeats 3+ times. Combined with 52,874 duplicate-source
rows (1.93%, up to 51 copies of a single source) and 18,143 untranslated targets
(0.66%), the corpus contains a real, if small, "repeat yourself" signal. This is
the plausible reason the loop symptom takes the shape of *clean re-translations*
rather than random continuation, and it is worth cleaning before the next run —
but at 2.67% it cannot be the primary driver of a 17% loop rate, let alone the
70% flood.

### F. Over-training — **ruled out by the log**

I listed this before seeing the curves. They do not support it. Train loss is
flat at 0.083–0.085 for thousands of steps with grad norm pinned at ~0.12, and
eval_loss improved at all 45 evaluations from step 500 to step 23,000, with a
single 0.0016 uptick at 23,500. There is no entropy collapse signature here.
The run also never reached 3 epochs — it stopped at epoch 2.03.

What survives from this item is the *measurement* point, and it is important:
**`eval_loss` is structurally incapable of seeing this failure.** It is
teacher-forced next-token loss over the reference, so it never asks the model
to terminate on its own, and it never decodes. A model can post its best
eval_loss ever while being unable to stop. That is exactly what happened, and
it is why the defect reached evaluation unnoticed.

### G. The evaluated checkpoint was neither final nor best — housekeeping

The run ended (or was killed) at step 23,640 of 34,860 with no `Train end` line,
so `load_best_model_at_end` never ran and no `sft_final` was written. Of the two
retained checkpoints (`save_total_limit: 2`), **step 23,000 had the better
eval_loss** (0.10923 vs 0.110881); 23,500 was evaluated. Not a cause of anything
observed — the gap is 0.0016 — but worth knowing that the artefact under test is
a mid-flight snapshot at epoch 2.02 of a planned 3.

## Fixes

Fixes 1–3 are **implemented**. The shared contract lives in `prompting.py`
(`resolve_stop_token_ids`, `render_training_prompt`, `render_inference_prompts`,
`tokenize_prompts_for_generation`), and all four generation entry points now go
through it: `evaluate_translations.py`, `inference.py`,
`translation_benchmark/generation.py`, and `train.py`'s own tokenization.
`tests/test_generation_chat_template.py` fails if any entry point starts
rendering prompts or resolving stop tokens on its own again.

One design point worth flagging: `render_inference_prompts` takes a
`use_training_rendering` flag, and the baseline is generated with
`add_generation_prompt=True` while the adapter uses the SFT rendering. Applying
the training rendering to both would have queried the untouched base model
off-distribution and quietly moved the baseline. Each system is prompted the way
it was trained.

Fixes 4–7 are not implemented; they need a retrain and are not on the critical
path.

### Fix 1 — make the decoder honour the published stop set (do this first)

`train.py`, in `make_deterministic_generation_config`, prefer the repository's
`generation_config.json` and fall back to the model config:

```python
try:
    generation_config = GenerationConfig.from_pretrained(base_model_id)
except OSError:
    generation_config = GenerationConfig.from_model_config(model_config)
```

Then union in the chat turn ender explicitly, so the fix holds even if a
checkpoint ships an incomplete `generation_config.json`:

```python
end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
stop_ids = as_id_set(generation_config.eos_token_id)
if isinstance(end_of_turn, int) and end_of_turn >= 0:
    stop_ids.add(end_of_turn)
generation_config.eos_token_id = sorted(stop_ids)
```

**This requires no retraining.** Re-run evaluation against the same
`checkpoint-23500`. Expect the whitespace floods and turn restarts to disappear
and the adapter's quality gain — already visible in the first 120 characters of
every row — to become the whole output.

### Fix 2 — render the evaluation prompt the way training rendered it

`evaluate_translations.py:191` uses `add_generation_prompt=True`, which drops the
`\n\n␣␣␣␣␣␣␣␣` the training template puts after `<start_of_turn>model`. Rather
than duplicating the marker trick, lift `train.py`'s prompt derivation into a
shared helper and call it from both sides, so the two can never drift again:

```python
# language_pairs.py or a new prompting.py, imported by train.py and
# evaluate_translations.py alike.
def render_training_prompt(processor, user_message):
    """Render the exact prefix the assistant turn is trained to continue.

    add_generation_prompt=True is NOT this string: TranslateGemma's template
    opens the assistant turn with a blank line and eight spaces of Jinja block
    indentation, which the generation prompt omits.
    """
    marker = "<|translategemma-target-boundary|>"
    text = processor.apply_chat_template(
        [user_message, {"role": "assistant", "content": marker}],
        tokenize=False, add_generation_prompt=False,
    )
    return text[: text.rindex(marker)]
```

Then tokenize that string directly for generation instead of calling
`apply_chat_template(..., add_generation_prompt=True)`. Batched generation still
needs left padding, which the current code already sets.

### Fix 3 — belt and braces at generation time

In `evaluate_translations.py`, pass the stop set on the call as well, so a stale
cached generation config cannot reintroduce the bug:

```python
generation_kwargs["eos_token_id"] = stop_ids          # same set as above
generation_kwargs["stop_strings"] = ["<end_of_turn>"] # needs tokenizer=...
```

`max_new_tokens: 1024` against a corpus whose 99th-percentile reference is well
under that is also a large blast radius. Scale it from the source length
(for example `min(1024, 64 + 2 * source_tokens)`) so a termination bug costs
seconds instead of minutes.

### Fix 4 — never truncate away the stop token

Independent of the above, `train.py:259` should not be able to produce a labelled
example with no terminator. 105,567 rows currently are. Either force the
terminator back on after truncation:

```python
input_ids = full_ids[:max_length]
if len(full_ids) > max_length:
    # A truncated example must still end the turn, or it contributes a
    # completion the model is never taught to finish.
    input_ids[-len(turn_end_ids):] = turn_end_ids
```

or drop over-length rows outright and record the count. Forcing the terminator
keeps the data (at the cost of a mid-sentence ending); dropping keeps the
targets clean. Given 3.86%, dropping is the simpler call and costs little.

### Fix 5 — clean the corpus, before the *next* training run

Not urgent — none of this caused the observed failure, and Fixes 1–3 need no
retraining. Do it when the next run is scheduled. Sized from
`logs/sft_corpus_audit.json`:

| Action | Rows | Note |
|---|---:|---|
| Drop unambiguous MT/copyright footers | ~4,550 (0.17%) | `ترجمه شده توسط`, `کلیه حقوق محفوظ است`, `Google Translate`, `Downloaded from` |
| Review `هوش مصنوعی` matches with a co-occurrence filter | 6,248 | do **not** bulk-delete; "artificial intelligence" is legitimate here |
| Drop or repair targets that already loop | 73,110 (2.67%) | largest single cleanup |
| Drop untranslated targets (Latin share > 0.6) | 18,143 (0.66%) | |
| Deduplicate exact source repeats | 52,874 (1.93%) | max 51 copies of one source |
| Drop `target == source` | 13 | |
| Drop short/long ratio outliers | 1,815 (0.07%) | |

Skip the `rstrip()`/`lstrip()` step from the earlier draft: the audit found zero
rows needing it.

### Fix 6 — measure termination, not just loss

`eval_loss` is blind here. Add `scripts/audit_degeneration.py` to the run:

```bash
uv run python scripts/audit_degeneration.py --eval-dir evaluation --fail-over 0.05
```

It exits non-zero when more than 5% of a system's rows show a decoding defect.
On the 2026-08-10 output it reports adapter `clean 12.8%` against base
`clean 92.1%` and fails — the check that should have run before anyone opened
the HTML report.

Longer term, `metric_for_best_model: "eval_loss"` should be paired with a
free-running generation check on a fixed 200-segment probe at each save, so
checkpoint selection can see termination collapse. The current run improved
eval_loss 45 times in a row while producing a model that fails on 85% of test
segments; that is the whole argument for adding the probe.

### Fix 7 — if degeneration survives Fixes 1–3

Only then reach for training changes. Note that the log gives no support for the
usual suspects: no divergence, no gradient instability, and the run stopped at
epoch 2.03, so "3 epochs was too many" is not an available explanation for this
checkpoint. `repetition_penalty` / `no_repeat_ngram_size` are diagnosis aids,
not fixes — they mask an unstopped model rather than teaching it to stop, and
they damage legitimate repetition in technical text.

## Scripts added

| Script | Needs | Answers |
|---|---|---|
| `scripts/audit_training_termination.py` | tokenizer only | Is `<end_of_turn>` in the decoder's stop set? Do train and eval prompts match? How many rows lose their stop token to truncation? |
| `scripts/audit_sft_corpus.py` | nothing (streams JSONL) | Trailing/leading whitespace, boilerplate, internal loops, copy-source, untranslated, duplicates, length outliers |
| `scripts/audit_degeneration.py` | evaluation CSVs | Per-system decoding failure rates; usable as a CI gate |

All three were run on the production host on 2026-08-10; results are in
`logs/*_audit.json` and summarised above. To reproduce:

```bash
uv run python scripts/audit_training_termination.py --config config.yaml
uv run python scripts/audit_degeneration.py --eval-dir evaluation
uv run python scripts/audit_sft_corpus.py --dataset data/splits/train.jsonl
```

## Next steps

Fixes 1–3 are done and the failure is gone (see Outcome). What remains is
optional and none of it is blocking:

- **Enable MetricX and COMET.** Both were off for these runs, so there is still
  no proper quality number. They were pointless while 85% of the output was
  degenerate; now they would measure something real.
- **Finish the training run.** It stopped at step 23,640 of 34,860, epoch 2.03
  of 3, with eval_loss still falling. Resuming is an improvement, not a repair.
- **Clean the corpus (Fix 5) before the next run.** Largest item is the 73,110
  self-repeating targets. Note the 4 boilerplate leaks disappeared with proper
  stopping — they were post-turn filler, not mid-output memorisation — so this
  is lower priority than it looked.
- **Fix truncation (Fix 4)** in the same pass, so no future run trains 105k
  examples without a terminator.
- **Wire `audit_degeneration.py --fail-over` into the evaluation run** so this
  class of failure can never again require a human to notice it.
