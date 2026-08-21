# Deterministic Glossary Memory for the Serving API

Status: design, approved for planning. Supersedes `api/TRANSLATION_MEMORY_DESIGN.md`,
whose skeleton this adopts and whose default enforcement strategy it changes.

## Problem

A system administrator must be able to declare that certain words or phrases
translate to specified target terms, and have the served translation honour that
declaration. Greedy decoding makes the model reproducible; it does not make it
terminologically compliant. Compliance has to be enforced outside the model.

This is a **termbase**, not conversation history and not a translation memory of
whole segments. Segment-level TM was considered and dropped: the requirement is
term-level.

## What the existing code constrains

`api/` is a stateless gateway. `translator.py` renders the TranslateGemma prompt
with the checkpoint's own tokenizer and posts **token ids** to vLLM
`/v1/completions`. The rendering is byte-exact with what SFT trained
(`prompting.py:render_training_prompt`), and
`docs/2026-08-10_adapter_degeneration_analysis.md` records what happens when a
prompt drifts off that distribution: fluent output that never stops.

Three consequences bind this design:

1. **The prompt rendering is not available as an injection point.** Putting
   glossary instructions into the user turn changes the prefix the adapter was
   conditioned on. Prompt injection is therefore a *hint* under
   `terminology_mode=suggest` only, never an enforcement mechanism.
2. **Anything inserted into the source text is off-distribution input** to a
   model fine-tuned on clean segments. Placeholder sentinels are such an
   insertion. The risk is not only a mangled sentinel — it is decoding straight
   into a repetition loop, which is the failure mode this repository already has
   a document about. Sentinel format must be chosen by measurement.
3. **Constrained decoding is out.** Guided decoding can force "this string
   appears somewhere in the output"; it cannot express "this string appears
   where that source span was". Alignment is the actual requirement.

## Prior art

What the commercial engines and the research literature actually do, and what
this design takes from each.

**No major vendor guarantees the term.** Google Cloud Translation's glossary
influences rather than guarantees the output, which is why the response carries
`translations` and `glossaryTranslations` as separate fields. Amazon Translate
states it plainly: custom terminology "doesn't guarantee that it will use the
target term for every translation" — it weighs the term against translation
context. This is the strongest available argument for best-effort-plus-report as
the default and against `strict` as the default.

**The industry moved from replacement to morphological adaptation.** Amazon
Translate originally did exactly post-hoc replacement — find the source term,
locate the corresponding string in the proposed translation, substitute. In 2023
it was enhanced to inflect: if the stored term is singular and the sentence needs
a plural, it converts, "without compromising fluency". Yandex does the same,
explicitly adapting for case and gender, and supports glossaries only on
specific language pairs — morphological adaptation is per-pair engineering, not
a free feature. This is the `preferred` mode above, and it is why the
affix-preserving rewrite is core rather than polish. Persian is morphologically
rich; nothing here comes for free.

**Inline markup protection is a shipped, documented product feature.** Azure
Translator's *dynamic dictionary* takes exactly the approach this design calls
`exact` mode, as request markup:

```
The word <mstrans:dictionary translation="wordomatic">wordomatic</mstrans:dictionary> is a dictionary entry.
```

Microsoft attaches two conditions that transfer directly here. The feature is
"safe only for compound nouns like proper names and product names" — so `exact`
mode must be restricted to that class of term, not offered as a general
enforcement lever. And the request must carry an explicit `From` language;
autodetect is forbidden, because the dictionary cannot be resolved before the
language pair is known.

The important caveat: Azure's model is trained to honour that markup. This
project's merged SFT adapter is not trained to honour anything of the kind, so
the pattern is validated but the sentinel probe below still decides the format.

Azure's four-way split is worth naming because it maps onto the modes above —
*phrase dictionary* (case-sensitive find-and-replace, which Microsoft warns
reduces sentence quality and advises against for verbs and adjectives),
*sentence dictionary* (whole-segment, i.e. translation memory), *dynamic
dictionary* (inline markup), and *neural phrase dictionary* (model adjusts
surrounding context while holding terminology accuracy — the train-time option
again).

**DeepL bakes glossary behaviour into training.** DeepL includes preferred
translations in the neural model during training rather than applying them at
inference. That is the Dinu et al. (2019) family: annotate target terms inline
in the source during training so the model learns to consume run-time
terminology. It beat constrained decoding on both quality and speed. It is not
available at inference time — but it *is* available to this project, which owns
its SFT pipeline. See the future phase below.

**Constrained decoding is disfavoured in the literature too.** Beyond the
alignment problem noted above, constrained decoding "adds significant
computational overhead to the inference step" and was outperformed by the
train-time approach. Two independent reasons to leave it out.

**Translate-then-refine with negative constraints is the practical win.** The
WMT23 terminology work detects violations by alignment, then re-decodes with the
violating word *negatively* constrained; a second variant refines with an LLM.
Both improved terminology recall. The negative-constraint half of this is
directly implementable here (see phase 2.5) because the gateway already posts
token ids.

**Ceilings are real.** Reported terminology integration rises from ~37% baseline
to ~73% after LLM post-editing. Even strong inference-time methods do not reach
100%. Any claim of determinism must come from the sentinel path, where the model
never sees the term, not from persuading the model.

## Two constraints that shape everything below

**The feature is optional and must be removable at runtime.** A deployment sets
`TG_GLOSSARY_ENABLED=false` and the gateway behaves exactly as it does today —
not approximately, exactly. No database connection is attempted, no admin
router is mounted, no response field is added or set to null, and the bytes sent
to vLLM are unchanged. This is a hard requirement, not a convenience flag: the
kill switch is what makes the feature safe to ship at all, because a
terminology bug in production is one environment variable and a restart away
from being gone.

This forces a structural rule: glossary application is **one seam**, not a
scatter of `if enabled` branches through `translator.py`. When disabled the
engine holds no glossary object and the translate path does not branch on it,
so "disabled" is provably identical to "absent" rather than a code path that
merely aims to be. Tests assert byte-identical output with the flag off.

**Callers usually cannot supply a domain.** In practice the request carries
text and little else — the integrations calling `/translate` are not
terminology-aware and will not be taught to be. So the domain-less path is the
*normal* path, not a degraded one, and everything must work well when `domain`
is never sent. Domain-specific glossaries are still wanted, but they are the
exception a knowledgeable caller or a pinned deployment opts into.

## Architecture

New package `api/glossary/`:

| Module | Responsibility | Purity |
|---|---|---|
| `models.py` | SQLAlchemy domain + entry + version tables | DB |
| `store.py` | async CRUD, version bump, audit rows | DB |
| `index.py` | compiled immutable snapshot (Aho-Corasick) | pure |
| `apply.py` | `plan()` / `protect()` / `restore()` | pure |
| `normalize.py` | Unicode + Persian folding | pure |
| `router.py` | admin FastAPI router, API-key guarded | store + index |

`index.py`, `apply.py`, and `normalize.py` are pure functions over plain data.
They carry the entire correctness burden of this feature and are fully testable
with no GPU, no vLLM, and no database — which is what makes this feature
developable on a machine that cannot run the stack.

### Request flow

```
request
  → resolve (source_lang, target_lang, domain, terminology_mode)
  → capture snapshot  (one immutable object + version, held for the whole request)
  → match on the FULL text            ← before splitting, not after
  → split into segments, carrying matched spans through
  → per segment:
        exact-mode spans  → substitute sentinel
        preferred-mode spans → leave text untouched
  → existing _encode / vLLM path, unchanged
  → per segment:
        validate sentinels → restore exact target terms
        locate preferred terms in normalized target → rewrite to canonical
  → rejoin segments
  → response + glossary report + version
```

**Matching happens before sentence splitting.** A term straddling a sentence
boundary would otherwise be lost, and sentinels inserted before splitting would
corrupt pysbd's boundary detection. Match on the whole text, then split, then
protect within segments using the spans already computed.

### Snapshot lifecycle

```
database → validated immutable snapshot → in-memory matcher
```

On any admin write: commit the transaction, build and validate a new matcher,
bump the version, atomically rebind the active snapshot. Requests already in
flight finish against the snapshot they captured. Readers take no lock because
the snapshot is frozen and replaced by rebinding, never mutated.

The database is never queried during a translation request.

## Entry model

Domains are rows, not free strings on the entry. Entries in no domain are
global, expressed as `domain_id NULL` rather than as a row — "global always
applies" is then a property of the query and cannot be disabled or deleted by
an admin action.

```
glossary_domain
  id
  name              UNIQUE; what a request sends
  src_lang, tgt_lang
  description
  enabled
  version           bumped on any write to this domain's entries
  created_by, created_at, updated_at
```

```
entry
id
domain_id           FK NULL = global; a domain entry shadows a global one
src_lang, tgt_lang
source_term
target_term
target_mode         'exact' | 'preferred'
aliases[]           JSON; known model renderings, used by preferred mode
forbidden[]         JSON; renderings that must never survive
case_sensitive
whole_word          default true
priority
enabled
notes
created_by, created_at, updated_at

UNIQUE(domain_id, src_lang, tgt_lang, source_term, case_sensitive)
```

### The two target modes

`target_mode` is the answer to "may the model inflect this term?", and it is a
separate axis from everything else. Collapsing it into a boolean alongside
`case_sensitive` or `enabled` is the mistake that makes a termbase unusable
later.

**`exact` — immutable string, sentinel-protected.**
The source span is replaced by a sentinel before generation; after generation
the sentinel is replaced by `target_term` verbatim. The term cannot be inflected
because the model never sees it. Correct for product names, organizations,
codes, URLs, units, legal designations — anything that must appear
character-for-character and that Persian morphology does not attach to.

Restrict it to that class deliberately. Microsoft ships the equivalent feature
and documents it as "safe only for compound nouns like proper names and product
names"; the admin API warns when an `exact` entry looks like a verb or
adjective.

Validation after generation: every expected sentinel present exactly once, no
unknown sentinel, none missing. On failure the segment is re-translated once
without protection, and the term is reported as a miss.

**`preferred` — base form, model may inflect, post-hoc canonicalization.**
The source text is left untouched, so the sentence stays grammatical. After
generation the target is normalized and searched for `target_term` or any
`aliases[]` entry; a hit is rewritten to canonical, keeping any Persian affix
that surrounds the stem (`می‌`, `ها`, `ی`, `به‌`, ezafe). No hit → reported miss,
translation returned unmodified. Correct for ordinary domain vocabulary, where
an uninflectable exact string would produce ungrammatical Farsi.

`forbidden[]` is checked in both modes: a banned rendering present in the output
is a reported violation even when the mandated term is absent.

## Normalization

Source side (en): NFKC, optional casefold, whitespace and hyphen folding.

Target side (fa) — **required for `preferred` mode to function at all**:

- NFKC
- `ي` (U+064A) → `ی` (U+06CC), `ك` (U+0643) → `ک` (U+06A9)
- ZWNJ (U+200C) folded for matching, preserved in output
- Arabic-Indic and Extended Arabic-Indic digits folded to ASCII for matching

Matching runs on the normalized form; rewriting is applied back to the original
string by offset. Without this, alias matching fails on terms that are visibly
present in the output, and the whole `preferred` path silently no-ops.

## Matching rules

- Aho-Corasick over the normalized source, one O(n) pass, at every glossary size.
  No size-dependent strategy switch.
- Word-boundary filtering by default (`whole_word`); never unrestricted substring
  replacement. `art` must not match inside `partial`.
- Overlap resolution: longest match wins, then `priority`, then leftmost.
  Entries for `attention` and `multi-query attention` resolve to the longer.
- Each occurrence is an independent span with its own sentinel.
- Snapshot is keyed by `(src_lang, tgt_lang)`; global and domain layers are
  merged at build time, domain shadowing global on identical `source_term`.

## API surface

### Translation

`Prompt` and `BatchPrompt` gain two optional fields. Both may be omitted
forever; that is the expected case.

```
domain: str | None
terminology_mode: 'off' | 'suggest' | 'enforce' | 'strict' | None
```

Default is `enforce`. `suggest` is the prompt-hint path and is available for the
base system only — it is off-distribution for the adapter and is documented as
such.

### Domain resolution

Resolution order, first hit wins:

```
request.domain  →  TG_GLOSSARY_DEFAULT_DOMAIN  →  global layer only
```

Omitting `domain` is **not** an error and not a degraded path. It resolves to
the global layer, which is the layer that receives most curation effort. A
deployment serving one field can pin its termbase with
`TG_GLOSSARY_DEFAULT_DOMAIN` and its callers never learn the concept exists.

An **unknown** domain name is different from an omitted one, and returns 404
listing the valid names. That distinction is the whole reason domains are rows:
silently translating a medical document with the general termbase because a
caller typed `medcial` is the failure this design exists to prevent, and it
returns HTTP 200 in every other arrangement.

`TG_GLOSSARY_UNKNOWN_DOMAIN=reject|fallback` exists for a deployment that would
rather degrade than fail. Default is `reject`.

The global layer always applies. A domain layers on top and shadows global on
an identical `source_term`; it never replaces the global set.

`strict` returns a controlled error with diagnostics when a mandated term cannot
be enforced. It is offered but is **not** the default: on Persian morphology a
strict default would fail ordinary sentences. Note that no commercial engine
offers a strict mode at all — Google and Amazon both decline to guarantee the
term. Offering one is a real differentiator and a real source of 4xx responses;
callers must opt in deliberately.

Response gains:

```json
{
  "translation": "...",
  "raw_translation": "...",
  "glossary_version": 12,
  "glossary": {
    "status": "applied",
    "applied": [
      {"source_term": "...", "target_term": "...", "mode": "exact", "count": 1}
    ],
    "misses": [
      {"source_term": "...", "reason": "sentinel_lost"}
    ],
    "violations": [
      {"source_term": "...", "forbidden": "..."}
    ]
  }
}
```

`status` distinguishes: no term matched, matched and enforced, matched but
enforcement failed, glossary disabled.

`raw_translation` is the model output before any glossary rewriting. Google
Cloud Translation returns exactly this pair (`translations` alongside
`glossaryTranslations`), and for the same reason: without the unmodified output
an administrator cannot tell whether a bad translation came from the model or
from the termbase, and cannot curate aliases. It is the single cheapest
debugging affordance in the feature.

### Admin

Router at `/admin/glossary`, every route behind a dependency checking a static
`TG_ADMIN_API_KEY` against a request header. There is no reverse-proxy auth in
front of this deployment and no second credential stage; this key is the whole
authorization boundary, so it is read from the environment and must not have a
default value.

```
GET    /admin/glossary/domains
POST   /admin/glossary/domains
PATCH  /admin/glossary/domains/{name}
DELETE /admin/glossary/domains/{name}
GET    /admin/glossary/entries
POST   /admin/glossary/entries
PATCH  /admin/glossary/entries/{id}
DELETE /admin/glossary/entries/{id}
POST   /admin/glossary/entries:bulk     CSV / JSON import
POST   /admin/glossary/reload
GET    /admin/glossary/versions
POST   /admin/glossary/dry-run          match a sample text, no model call
```

`dry-run` is not optional polish. It is how an administrator sees which entries
fire, how overlaps resolved, and what would be protected — before the change
reaches production traffic.

Writes validate language codes, reject empty terms, detect conflicting entries,
record audit fields, bump the version, and publish the new snapshot atomically.

Bulk import accepts TSV and CSV — TSV because it is what Google and DeepL both
use for glossary interchange, so an administrator's existing files load without
reshaping. An import is validated in full and published as a **new version**,
never applied as incremental mutation of live entries.

CORS is not authorization and is not part of this boundary.

### Admin-side validation

Rules taken from Amazon Translate's published best practices and Google's
stopword behaviour, encoded as write-time validation rather than left to
administrator discipline:

- **Reject stopword entries.** An entry whose `source_term` exactly matches a
  stopword is refused. Google ships per-language lists including Persian, and
  ignores such entries silently; refusing loudly at write time is better, since
  a silently ignored entry looks like a broken feature. Exact match only — `the`
  is a stopword, `the cloud` is not.
- **Reject conflicting duplicates** — same source term, same scope, different
  target. Already enforced by the unique constraint; surface it as a clear 409.
- **Reject terms used to control spacing, punctuation, or capitalization.** Not
  what a termbase is for, and it interacts badly with normalization.
- **Warn on long phrases.** Google caps around five words; long phrases rarely
  fit target grammar. Warn, do not block.
- **Warn on verbs and adjectives in `exact` mode.** Microsoft's guidance, and
  the same reasoning as the morphology problem above.
- **Warn above a soft entry-count threshold.** Curated and small beats large:
  every vendor says so, and each additional entry is another chance to damage a
  sentence.

### Explicit source language

A request that uses the glossary must resolve a source language explicitly.
Today `source_lang` falls back to `TG_SOURCE_LANG`, which is fine for
translation but not for glossary selection: a silently wrong default selects the
wrong dictionary and produces confidently wrong terminology. Both Azure and
DeepL forbid autodetect with glossaries for this reason. When the glossary is
active and the caller omitted `source_lang`, the resolved default is echoed in
the response so the choice is visible.

## Configuration

```
TG_GLOSSARY_ENABLED          bool, default false        ← the kill switch
TG_GLOSSARY_DB_URL           default sqlite+aiosqlite:///./data/glossary.db
TG_ADMIN_API_KEY             required when enabled; no default, no fallback
TG_GLOSSARY_DEFAULT_DOMAIN   str | None; used when a request omits domain
TG_GLOSSARY_UNKNOWN_DOMAIN   'reject' | 'fallback', default reject
TG_TERMINOLOGY_MODE          default enforce
```

`TG_GLOSSARY_ENABLED` defaults to **false**. The feature ships dark and is
turned on deliberately, which also means merging it cannot change the behaviour
of the running deployment.

Validation at startup, so a misconfiguration fails loudly at boot rather than
quietly at the first admin request: enabling the glossary without
`TG_ADMIN_API_KEY` is a startup error, and `TG_GLOSSARY_DEFAULT_DOMAIN` naming
a domain that does not exist is a startup error.

SQLite is correct for a single gateway with one administrator and modest write
volume. It lives on a mounted volume, never inside the container filesystem.
`store.py` is written against a repository interface so a move to PostgreSQL —
required the moment there are replicas — does not touch `apply.py` or `index.py`.

## Measured on the served checkpoint

Everything below was run against the real deployment — vLLM **0.13.0**, model
`translategemma`, merged checkpoint — not inferred. `scripts/probe_logit_bias.py`
covers the API half.

### Sentinel survival — phase 2 is de-risked

Three candidates, three sentence frames each (including one frame with the same
sentinel twice), greedy decode:

| Candidate | Survived | Degeneration |
|---|---|---|
| `__TG_TERM_001__` | 3/3, both occurrences intact | none, `finish_reason: stop` |
| `⟦0⟧` | 3/3, both occurrences intact | none |
| target term inline in source | 3/3 | none |

The failure this design was most afraid of — an off-distribution sentinel
sending a merged SFT adapter into the repetition loop of
`docs/2026-08-10_adapter_degeneration_analysis.md` — **did not occur once**.
Surrounding Farsi stayed grammatical and word order adapted around the sentinel.

This lowers phase 2's risk from "may not be viable" to "needs a sample size".
The ~200-segment run is still worth doing before shipping `exact` mode, but it
is now a confirmation rather than a gate.

Two caveats the small sample cannot cover. Every sentinel sat in a noun slot
where no Persian affix attached; the morphology cases that motivate `preferred`
mode are exactly the ones untested here. And three frames is three frames.

Worth noting the third candidate: substituting the **Farsi target directly into
the English source** also survived, and produced natural sentences with no
restore step at all. That is close to what Azure's dynamic dictionary does. It
trades away the unambiguous validation an abstract sentinel gives you, and
risks the model inflecting the injected Farsi, so `__TG_TERM_001__` remains the
default — but the inline variant is a cheap fallback if sentinel validation
proves noisy at scale.

### Negative constraints — phase 2.5 confirmed, and narrower than hoped

| Question | Answer on 0.13.0 |
|---|---|
| `logit_bias` on `/v1/completions`? | Yes — in the schema, and it demonstrably changes greedy output |
| `bad_words` on `/v1/completions`? | **No.** Absent from `CompletionRequest`; present on `ChatCompletionRequest` only |
| Unknown fields? | **Silently ignored, HTTP 200** |
| Bias scope in a multi-prompt request? | Request-level. One bias set per HTTP request |

The second and third rows combine into a trap worth stating plainly: sending
`bad_words` to `/v1/completions` on this deployment returns 200 and does
nothing. Verified — the same payload changes output on a newer dev build and is
inert on the pinned 0.13.0. Never infer support from a successful response.

Consequences: negative constraints must be per-token `logit_bias`, so banning a
multi-token Farsi term means banning individual tokens — and banning a term's
first token also penalises it inside unrelated words in that segment. And
because bias is request-scoped, phase 2.5 costs one extra single-prompt request
per violating segment.

`logit_bias` is an **additive penalty, not a ban**. Measured margins: `-1` and
`-2` left a high-confidence token in place, `-3` flipped it; on another sample
`-1` sufficed. The required magnitude is whatever exceeds the argmax margin, so
use `-100` and do not treat any smaller value as reliable.

**What banning actually buys.** Banning the model's rendering of "multi-query
attention" produced fluent, grammatical alternatives every time — no
degeneration, and the sentence reflowed naturally around the change
(`متکی` → `وابسته`):

| Banned | Result |
|---|---|
| — | `مدل به توجه چندپرسشی متکی است.` |
| ` توجه` | `مدل به دقت چندگانه وابسته است.` |
| ` چند` | `مدل به توجه چندین پرسش متکی است.` |
| all four term tokens | `مدل به دقت چندین پرسش وابسته است.` |

Note what this does *not* do: it moves the model **off** a rendering, it does
not move it **onto** the mandated one. Negative constraints can only remove.
Installing the required term remains the job of post-hoc replacement. So phase
2.5 stays scoped exactly as written — a repair for `forbidden[]` violations,
never a mechanism for enforcing `target_term`.

The sentence reflowing is a second-order cost: a re-decode changes text beyond
the banned span, so the response's `raw_translation` should be the *original*
decode, not the re-decoded one, or diffing becomes meaningless.

### Method note

The prompts above used a hand-built Gemma-shaped rendering, since the merged
checkpoint and its tokenizer are not reachable from the development machine.
Output was clean, terminating Farsi, so the runs were in-distribution — but the
definitive sample-size run must render through `prompting.render_training_prompt`
via the gateway, not an approximation of it.

## Disabled behaviour

`TG_GLOSSARY_ENABLED=false` is the shipping default and the state every
deployment can return to. What it must mean, precisely:

- No engine is constructed, no SQLite file is opened or created, no migration
  runs. A deployment that never enables the feature never grows a database.
- The admin router is not mounted. `/admin/glossary/*` returns 404, not 401 —
  there is nothing there.
- `Prompt.domain` and `Prompt.terminology_mode` are accepted and ignored, so a
  caller that learned to send them does not start failing when an operator
  turns the feature off mid-incident.
- Response objects **omit** the glossary fields entirely rather than emitting
  nulls. Callers written against the disabled shape keep working when it is
  enabled, because the fields are additive.
- The token ids posted to vLLM are byte-identical to what today's code sends.
- `/model-info` reports `glossary_enabled: false` so the state is observable
  without reading the container's environment.

Re-enabling restores the previous database untouched; disabling is not a
destructive operation and never drops data.

The corresponding runtime control for a caller who wants translation without
terminology on an enabled deployment is `terminology_mode: "off"`, which skips
matching entirely rather than matching and discarding.

## Delivery phases

**Phase 0 — the seam and the switch.** `TG_GLOSSARY_ENABLED` with nothing behind
it: the single application seam in `translator.py`, the config flag, the
`/model-info` field, and the test asserting byte-identical vLLM payloads with
the flag off. Small, and it is what makes every later phase safe to merge —
each one lands dark behind a switch that is already proven inert.

**Phase 1** — domain and entry schema, store, index, normalization, `preferred`
mode, admin CRUD for domains and entries, dry-run, response metadata,
`terminology_mode` in `{off, enforce}`. Ships without touching the source text
sent to the model, so it cannot regress translation quality even when enabled.

**Phase 2** — sentinel probe, then `exact` mode with validation and single-retry
fallback. `strict` mode.

**Phase 2.5 — negative-constraint re-decode.** When a `preferred` term misses
and the output contains a `forbidden[]` rendering (or an alias the admin has
marked wrong), re-decode the segment with that rendering's token ids negatively
biased, rather than reporting the miss immediately. This is the WMT23
translate-then-refine loop, and it fits the existing transport: the gateway
already sends token ids to `/v1/completions`, and the same tokenizer that
renders the prompt can tokenize the banned string.

Measured on the deployment (see above), so no longer a prerequisite:
`logit_bias` is supported and effective, `bad_words` is not available on
`/v1/completions` in 0.13.0, and bias is request-scoped — so this phase costs
one extra single-prompt request per violating segment. Bans must be per-token
and set to `-100`.

Scope it to what the measurement supports: re-decode only when the output
contains a `forbidden[]` rendering. A ban removes a bad rendering; it cannot
install the mandated one, so this never substitutes for post-hoc replacement.

This is what makes `forbidden[]` load-bearing rather than decorative.

**Phase 3** — `suggest` mode for the base system, if wanted.

**Future — train-time terminology awareness.** The endgame, and the option most
users of a translation API do not have: this project owns `train.py` and
`prepare_data.py`. Augmenting the SFT corpus with inline term annotations
(Dinu-style) would teach the adapter to consume run-time terminology natively,
inflecting it correctly, which is what DeepL ships. It would raise the
`preferred` hit rate at its source instead of repairing output afterwards.
Out of scope for this feature; it is a training-side project and should be
scoped as one, but the entry schema above is already the right input for it.

## Explicitly out of scope

**Span translation and reconstruction** (translate prefix and suffix separately,
concatenate around a fixed term). EN→FA is word-order divergent; reconstructing
across a fixed Farsi span produces broken sentences. It would be a code path
that reliably yields worse output than returning the unprotected translation
with a reported miss.

**Segment-level translation memory.** Different feature, not requested.

**Positively constrained decoding.** Cannot express span alignment, adds
significant inference overhead, and was outperformed by train-time methods in
the literature. Negative constraints are a different and much cheaper thing —
see phase 2.5.

## Testing

Pure-function tests: phrase matching, word boundaries, Unicode and Persian
normalization, ZWNJ, case sensitivity, overlapping and repeated terms,
punctuation adjacency, terms straddling a sentence boundary, domain shadowing,
affix-preserving rewrite, forbidden detection.

Fake-upstream tests: missing / duplicated / unknown / corrupted sentinels,
unprotected retry, batch requests, snapshot stability across a mid-request
reload, strict rejection.

Domain-resolution tests: omitted domain resolves to global, unknown domain 404s
under `reject` and falls back under `fallback`, `TG_GLOSSARY_DEFAULT_DOMAIN`
applies only when the request omits one, a disabled domain is skipped, domain
entries shadow global entries on the same source term.

Kill-switch tests, which are the load-bearing ones:

- With `TG_GLOSSARY_ENABLED=false`, the payload posted to a fake vLLM is
  byte-identical to the payload the pre-glossary code posts for the same input.
- With the flag off, no SQLite file is created anywhere on disk.
- With the flag off, `/admin/glossary/*` returns 404 and response bodies carry
  no glossary keys at all.
- A request sending `domain` and `terminology_mode` succeeds and ignores them
  when the flag is off.
- Enabling, writing entries, disabling, and re-enabling returns the same data.

Integration: persistence across container restart, concurrent admin writes,
startup failure when enabled without `TG_ADMIN_API_KEY`.

The property that must hold:

```
For every accepted strict translation, every matched source term
maps to its exact target term.
```
