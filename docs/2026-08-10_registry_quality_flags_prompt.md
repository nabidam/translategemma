# Handoff: corpus quality flags in mtdataregistry

The corpus audit behind `2026-08-10_adapter_degeneration_analysis.md` found
defects in `data/splits/train.jsonl`, which is a snapshot exported from the
**MT Dataset Registry** at `/home/dev/projects/mtdataregistry`. Fixing them in
this repository would be wrong: every future export would carry them back.
They belong upstream.

The prompt below is written to be pasted into a fresh agent session **with the
registry as the working directory**. It carries the measurements, the
constraint that the data is already in use, and the reasoning for the
constraint — but deliberately does not dictate a schema, because that repo's
`AGENTS.md` asks for the simplest thing that works and its author knows the
codebase better than this analysis does.

Nothing here blocks the current adapter. This is preparation for the next
training run.

---

## The prompt

> You are working in the MT Dataset Registry (`/home/dev/projects/mtdataregistry`).
> Read `AGENTS.md` and `CONVENTIONS.md` first and follow them — in particular
> "avoid overengineering", "no TDD", and the existing FastAPI / SQLAlchemy /
> Alembic / Pydantic / polars / DuckDB stack. Do not add new infrastructure.
>
> **Context.** A snapshot exported from this registry was used to fine-tune a
> 12B translation model. A downstream audit of the exported 2,735,122-row
> training split found quality defects that survived ingestion. The model
> memorised some of them and reproduced them at inference. The registry is the
> right place to detect these, because every snapshot inherits whatever the
> registry stores.
>
> **Measured defects, in the exported `train.jsonl`** (columns `source_text`,
> `target_text`, and the audit script is
> `/home/dev/projects/forgpt/codes/2026/mt/translategemma/scripts/audit_sft_corpus.py`
> — read it, it defines each check precisely):
>
> | Defect | Rows | Share | Note |
> |---|---:|---:|---|
> | Target already contains an internal loop (a 6-gram repeating 3+ times) | 73,110 | 2.67% | largest item |
> | Duplicate source across the corpus (normalised) | 52,874 | 1.93% | up to 51 copies of one source |
> | Duplicate source+target pair | 43,611 | 1.59% | subset of the above |
> | Target is mostly Latin script (never actually translated) | 18,143 | 0.66% | |
> | Translator / copyright boilerplate in the target | ~4,550 | 0.17% | see the caveat below |
> | `target_text == source_text` | 13 | ~0 | |
> | Target far shorter or longer than the source | 1,815 | 0.07% | |
>
> Two boilerplate strings were reproduced verbatim by the trained model:
> `ترجمه شده توسط هوش مصنوعی` ("translated by AI") and
> `لطفاً توجه داشته باشید که این ترجمه ممکن است به طور کامل دقیق نباشد و نیاز به ویرایش بیشتر دارد`.
> Also present: `کلیه حقوق محفوظ است` (4,358 rows), `Google Translate`,
> `Downloaded from`.
>
> **Caveat you must respect.** A naive search for `هوش مصنوعی` matches 6,248
> rows, but that phrase simply means "artificial intelligence" and this is a
> science corpus, so most of those are legitimate content. Only ~4,550 rows are
> genuine contamination. Any boilerplate rule needs co-occurrence (for example
> `ترجمه` near `هوش مصنوعی`, or a match anchored to the start or end of the
> text), not a bare substring match. Report how many rows each rule matches
> before anything acts on it.
>
> **What is already handled, so do not redo it.**
> `backend/services/ingestion/normalize.py` already applies `str.strip_chars()`
> to both text columns and de-duplicates on `(source_text, target_text)` within
> a single import. That is why the audit found zero rows with leading or
> trailing whitespace. The gap is that de-duplication is per-import, so the same
> source appearing in two batches survives, and no content-quality checks exist
> at all.
>
> **Hard constraint: the data is in use.** Existing batches have been exported
> into snapshots that are training real models, and those snapshots must stay
> byte-reproducible. So:
>
> - Do not mutate or rewrite existing normalized Parquet objects.
> - Do not delete or alter existing batches, samples, or snapshots.
> - Existing snapshots must continue to resolve to exactly the same rows.
> - A newly built dataset or snapshot may exclude flagged rows, but only when
>   that exclusion is recorded in the snapshot's own metadata, so a reader can
>   tell which filters produced it.
>
> **What to build.**
>
> 1. A per-sample quality assessment computed from `source_text` and
>    `target_text`, covering the defects in the table. Store the result
>    alongside the sample rather than by editing it. Flags, a bitmask, a
>    sidecar table keyed by `sample_id` — your call; pick whatever fits the
>    existing model layer and say why in the PR description.
> 2. Run it on new imports, and provide a way to backfill existing batches
>    without rewriting their data.
> 3. Let the dataset builder and/or snapshot export filter on it, defaulting to
>    **no filtering** so existing behaviour is unchanged unless asked for.
> 4. Cross-batch duplicate source detection. Per-import de-duplication already
>    exists; corpus-level does not. Duplicates should be *flagged*, not removed
>    — which copy to keep is a dataset-building decision, not an ingestion one.
> 5. Surface the counts in the UI/API wherever batch statistics already appear,
>    so a researcher can see a batch is 12% self-repeating before training on it.
>
> **Deliberately out of scope.** Do not attempt to repair or rewrite text. Do
> not build a scoring model. Do not add a queue, a cache, or a new service.
>
> **Definition of done.** Backfill the existing corpus, then report the flag
> distribution per batch and confirm the total matches the table above (within a
> small margin — the audit ran on the post-split export, not on raw batches).
> Confirm that an existing snapshot still exports identical rows.

---

## Why this is not urgent

The four boilerplate leaks in the 2026-08-10 evaluation **disappeared** once the
decoder stopped correctly: they were post-turn filler, not mid-output
memorisation. The 2.67% self-repeating targets are the strongest remaining
argument for this work, and the adapter's residual loop rate (2.96%) is already
below the base model's (3.45%).

Do this before the next training run, not before shipping the current adapter.
