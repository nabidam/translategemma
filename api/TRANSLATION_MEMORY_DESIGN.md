# Translation Memory and Deterministic Glossary Design

## Context

`api/` is a stateless FastAPI gateway around vLLM:

- `main.py` exposes `/translate` and `/translate/batch`.
- `translator.py` optionally splits text, renders TranslateGemma prompts, tokenizes them, and sends token IDs to vLLM.
- vLLM performs generation. The API process holds the tokenizer and HTTP client, not model weights.
- `config.py` contains environment configuration, but there is no persistence layer or admin authorization.
- Decoding is greedy by default. This improves reproducibility, but greedy decoding does not guarantee glossary compliance.

The desired feature is system-admin-managed terminology memory: selected words or phrases must translate to specified target terms deterministically.

## Industry Research

Major translation providers separate customization into distinct features:

- **Glossary or terminology:** controlled mappings from source terms to target terms.
- **Translation memory:** previously translated sentences or segments, reused through exact or fuzzy matching.
- **Custom model or adaptive translation:** broader domain, style, and contextual adaptation learned from examples.

These are not interchangeable. The requirement to translate selected words or phrases consistently is primarily a glossary requirement. Reusing approved sentence translations is a separate translation-memory feature.

### Google Cloud Translation

Google exposes glossaries as named, persisted resources. An administrator creates a glossary separately and a translation request explicitly references its resource ID.

Google's design includes:

- Unidirectional glossaries for one source-target language pair.
- Equivalent-term sets spanning multiple languages.
- TSV, CSV, and TMX import formats.
- Words and short phrases, usually fewer than five words.
- Case-sensitive matching by default and a request option to ignore case.
- Glossary-specific IAM permissions.
- Asynchronous glossary creation.
- Contextual glossary translation for Google's Translation LLM.
- A stopword list whose exact entries are ignored even if included in a glossary.

Google can return both regular translation and glossary-applied translation, which makes glossary effects observable. Google also recommends retaining original glossary files because its glossary resources do not provide built-in version control or rollback.

Lessons for this API:

- Treat glossaries as named resources, not one global mutable table.
- Scope each dictionary by language pair.
- Select glossary explicitly in a translation request.
- Preserve source files or implement first-class version history.
- Return glossary version and application metadata.
- Consider stopword or ambiguity validation rather than accepting every admin entry unquestioningly.

Sources:

- [Google Cloud: Creating and using glossaries](https://cloud.google.com/translate/docs/advanced/glossary)
- [Google Cloud: Glossary stopwords](https://cloud.google.com/translate/docs/advanced/stopwords)

### Microsoft Azure Translator

Microsoft provides phrase dictionaries, sentence dictionaries, dynamic dictionaries, and neural phrase dictionaries.

#### Phrase Dictionary

Microsoft describes a phrase dictionary as a case-sensitive exact find-and-replace mechanism. The supplied term is translated as specified while the rest of the sentence is translated normally.

Microsoft recommends phrase dictionaries primarily for:

- Product names.
- Proper names.
- Compound nouns.
- Product features and fixed technical expressions.

Microsoft warns that phrase replacement can reduce context and overall sentence quality. It recommends avoiding broad use for verbs and adjectives because they are highly contextual.

#### Sentence Dictionary

A sentence dictionary matches an entire source sentence and returns its stored target translation. Partial matches are not accepted. This behavior is closer to exact translation memory than terminology matching.

#### Dynamic Dictionary

Azure accepts inline markup that binds a source phrase to its requested translation:

```xml
<mstrans:dictionary translation="wordomatic">
  wordomatic
</mstrans:dictionary>
```

This approach is conceptually similar to protected placeholders or tagged spans. It requires an explicit source language and is recommended for compound nouns, proper names, and product names.

#### Neural Phrase Dictionary

Microsoft's neural phrase dictionary lets its model adjust surrounding context while preserving high terminology accuracy. This addresses the fluency loss caused by literal phrase replacement, but it is model-integrated rather than an application-level exact replacement contract.

Lessons for this API:

- Support both exact phrase enforcement and contextual preference as different policies.
- Use fixed terminology mainly for stable nouns and named entities.
- Treat whole-sentence exact matches separately from phrase dictionaries.
- Protected spans are an established request-time customization pattern.
- Do not claim deterministic enforcement for a contextual neural mode without validating output.

Sources:

- [Microsoft: Custom Translator dictionaries](https://learn.microsoft.com/en-us/azure/ai-services/translator/custom-translator/concepts/dictionaries)
- [Microsoft: Dynamic dictionary](https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/how-to/use-dynamic-dictionary)

### Amazon Translate

Amazon calls the feature custom terminology. Administrators upload terminology files and callers select a terminology resource for translation.

Amazon explicitly states that custom terminology does not guarantee use of the target term for every translation. Its model considers source meaning, target meaning, and sentence context before deciding whether a supplied target term fits.

Amazon recommends:

- Keep terminology small and focused.
- Include only terms whose translations need control.
- Prefer brand names, product names, proper names, and unique terms.
- Avoid duplicate source phrases with conflicting targets.
- Avoid using terminology to control spacing, punctuation, or capitalization.
- Avoid ambiguous source and target terms.
- Ensure target terms are fluent and semantically equivalent.
- Use caution with long phrases because they may not fit target-language grammar naturally.

Lessons for this API:

- A model-aware glossary is a preference mechanism unless output is validated.
- Strict mode needs application-side enforcement and validation.
- Reject conflicting entries during administration.
- Curated terminology quality matters more than raw glossary size.

Sources:

- [Amazon Translate: Custom terminology](https://docs.aws.amazon.com/translate/latest/dg/how-custom-terminology.html)
- [Amazon Translate: Custom terminology best practices](https://docs.aws.amazon.com/translate/latest/dg/ct-best-practices.html)

### DeepL

DeepL exposes named multilingual glossary resources. A glossary contains one or more dictionaries, each mapping source phrases to target phrases for a language pair.

DeepL's design includes:

- Glossary creation, inspection, editing, and deletion.
- Multilingual glossaries containing several language-pair dictionaries.
- TSV and CSV entry formats.
- Explicit `glossary_id` selection in translation requests.
- Up to five selected glossaries in a request.
- Explicit source language when using glossaries.
- Validation that each selected glossary covers the requested language pair.
- Separation between glossary resources and translation-memory resources.

DeepL translation memory stores and reuses previously approved segment translations. Its API supports TMX import and export, exact or fuzzy application through a similarity threshold, and a recommended threshold of at least 75 percent. Translation-memory import and export run as background jobs.

Lessons for this API:

- Keep glossary and translation memory as separate domain objects and APIs.
- Require explicit source language when selecting glossary resources.
- Permit multiple glossaries only after defining deterministic conflict precedence.
- Apply translation memory at segment level with a visible similarity threshold.
- Use TMX as the interoperability format for translation memory.

Sources:

- [DeepL: Multilingual glossaries](https://developers.deepl.com/docs/api-reference/multilingual-glossaries)
- [DeepL: Translate text](https://developers.deepl.com/docs/api-reference/translate)

### Yandex Cloud

Yandex Cloud documentation could not be verified during this research pass because its public documentation endpoint redirected to an anti-bot verification page. No Yandex-specific implementation claim is included without an accessible official source.

Yandex behavior should be verified later through its authenticated Cloud documentation, API reference, console, or support channel.

### Cross-Provider Pattern

Common enterprise architecture is:

```text
admin-managed named glossary
  -> one or more language-pair dictionaries
  -> explicit resource selection per request
  -> model-aware terminology application
  -> separate translation-memory resource
```

Common properties include:

- Named resources with stable IDs.
- Explicit source and target languages.
- Separate resource lifecycle from model deployment.
- Request-time resource selection.
- Administrative permissions.
- TSV, CSV, or TMX interoperability.
- Size and term-length limits.
- Defined case and phrase-matching behavior.
- Separate exact and contextual customization modes.

Common limitations include:

- Model-aware terminology may not guarantee exact output.
- Literal replacement can reduce grammatical fluency.
- Long or ambiguous terms are difficult to apply correctly.
- Case, punctuation, whitespace, and morphology affect matching.
- Glossaries are not substitutes for domain fine-tuning.
- Translation memory and terminology solve different problems.

## Recommendation

Implement a versioned, admin-managed glossary with deterministic enforcement outside the model:

```text
request
  -> resolve language pair
  -> load active glossary snapshot
  -> identify source spans
  -> protect glossary terms
  -> translate with existing model path
  -> validate protected placeholders
  -> restore exact target terms
  -> response with glossary metadata
```

Do not treat this as conversation history. It is a terminology database or termbase with explicit matching, precedence, and enforcement rules.

The recommended initial implementation is:

1. SQLite persistence for one deployment and one administrator.
2. An in-memory immutable glossary snapshot for request-time matching.
3. Longest-match-first phrase matching.
4. Placeholder protection before model generation.
5. Strict placeholder validation after generation.
6. Exact target-term restoration after validation.
7. Prompt instructions as a secondary quality hint, not as the enforcement mechanism.

Keep persistence behind a repository interface so it can move to PostgreSQL if deployments become replicated or multi-tenant.

Following enterprise API patterns, expose glossaries as named, versioned resources selected by `glossary_id`. Do not make one invisible global dictionary the only supported model. A deployment may define a default glossary for backward-compatible callers, but the resolved resource and version should remain observable.

## Glossary Entry

At minimum, each entry should contain:

```json
{
  "source_language": "en",
  "target_language": "fa",
  "source_term": "multi-query attention",
  "target_term": "توجه چندپرسشی",
  "case_sensitive": false,
  "match_mode": "phrase",
  "enabled": true
}
```

A production entry should also include:

```text
id
priority
domain or project scope
created_at
updated_at
created_by
```

Useful optional fields:

```text
whole_word
preserve_case
notes
forbidden_translations
```

Glossary scope should be explicit. The minimum scope is source language plus target language. Domain or project scope should be added if different users or documents require different terminology.

## Enforcement Strategies

### Prompt Instructions Only

Add glossary entries to the model prompt:

```text
Translate using these mandatory terms:
- multi-query attention -> توجه چندپرسشی
- genome sequence -> توالی ژنوم
```

Advantages:

- Easy to implement.
- Preserves full sentence context.
- Does not rewrite model output.

Problems:

- Not deterministic.
- The model can ignore, alter, inflect, or partially translate terms.
- Entries consume context tokens.
- Base and adapter behavior may differ.
- Large glossaries can reduce translation quality.

Use this only as a soft hint.

### Post-Processing Only

Translate normally, then replace terms in the output.

Advantages:

- Can guarantee a target string when it is found.
- Simple runtime implementation.

Problems:

- The source term may have been translated into an unexpected form.
- Arbitrary target replacement can modify unrelated text.
- Replacement can damage grammar, inflection, punctuation, or spacing.
- Matching generated target text can be ambiguous.

Use this as a safety net, not as the primary mechanism.

### Placeholder Protection

Replace matched source spans before model generation:

```text
Input:
The model uses multi-query attention.

Protected input:
The model uses __TG_TERM_001__.

Glossary:
__TG_TERM_001__ -> توجه چندپرسشی
```

After generation, validate and restore the placeholder:

```text
Model output:
The model از __TG_TERM_001__ استفاده می‌کند.

Final output:
The model از توجه چندپرسشی استفاده می‌کند.
```

Advantages:

- Preserves surrounding sentence context.
- Target term is inserted deterministically.
- Works well for product names, legal terms, organization names, technical expressions, and fixed phrases.

Problems:

- The model can alter, omit, or duplicate a placeholder.
- Placeholder formatting must be validated.
- Multiple occurrences require unique placeholders.
- Exact target terms may conflict with grammatical inflection.

This should be the default enforcement strategy, combined with validation and fallback behavior.

### Span Translation and Reconstruction

Split input into fixed glossary and non-glossary spans:

```text
translated_prefix + exact_target_term + translated_suffix
```

Advantages:

- Glossary terms are guaranteed.
- No placeholder corruption is possible.

Problems:

- Context is lost at span boundaries.
- Fragments can translate poorly.
- Word order differences can damage the sentence.
- Many terms can create many model requests.

Use this as a fallback for high-value terms when placeholder validation fails.

### Constrained Decoding

Token-level constraints can force target terms during generation. This is theoretically strongest, but less attractive for the current deployment:

- The gateway uses vLLM `/v1/completions` with token IDs.
- Phrase constraints require guided-decoding support and tokenizer-level constraint construction.
- Multiple terms, repeated terms, language pairs, and output ordering make constraints complex.
- vLLM feature compatibility must be pinned and tested.

Consider this only if strict terminology compliance becomes a core requirement and placeholder enforcement is insufficient.

## Matching Rules

Define matching behavior before implementation:

- Normalize Unicode before matching.
- Match phrases before single words.
- Use longest-match-wins.
- Use word-boundary matching by default.
- Do not perform unrestricted substring replacement.
- Allow explicit case-sensitive or case-insensitive matching.
- Define whitespace and hyphenation normalization.
- Define punctuation behavior.
- Resolve overlaps by match length first, then priority.
- Treat each occurrence as a separate protected span.

For example, entries for `attention` and `multi-query attention` must select the longer phrase when both match.

Avoid matching `art` inside `partial` unless an entry explicitly requests substring matching.

## Request Processing

Recommended request flow:

1. Resolve source and target language using current request and server defaults.
2. Capture one immutable glossary snapshot and its version.
3. Match glossary terms against the source text.
4. Replace every matched occurrence with a unique placeholder.
5. Preserve existing TranslateGemma prompt rendering and vLLM request behavior.
6. Validate model output:
   - Every expected placeholder occurs exactly once.
   - No unknown placeholder occurs.
   - No expected placeholder is missing.
7. Restore exact target terms.
8. Return the translation and glossary version.

Sentence splitting needs special care. Glossary matching should happen before splitting, or matched spans must be carried through the splitter. Otherwise a phrase can be lost when processing sentence segments independently.

An enterprise-style request could be:

```json
{
  "text": "The model uses multi-query attention.",
  "source_lang": "en",
  "target_lang": "fa",
  "glossary_id": "technical-fa",
  "translation_memory_id": "manuals-fa",
  "terminology_mode": "strict",
  "translation_memory_threshold": 85
}
```

Glossary use should require an explicit or resolved source language. Language autodetection and glossary matching are a risky combination because the selected dictionary depends on knowing the language pair before matching.

## Enforcement Modes

Expose terminology behavior explicitly rather than hiding policy inside the server:

```json
{
  "text": "...",
  "source_lang": "en",
  "target_lang": "fa",
  "terminology_mode": "strict"
}
```

Suggested modes:

- `off`: normal translation.
- `suggest`: include glossary terms as model instructions.
- `enforce`: placeholder protection and validation.
- `strict`: retry or reject when a mandatory term cannot be enforced.

For the stated requirement, `enforce` or `strict` should be the default. Existing callers should either adopt this default deliberately or opt in through an API version or request option.

## Failure Behavior

The API must not silently return noncompliant output when a term is mandatory.

Recommended fallback sequence:

1. Generate with protected placeholders.
2. If validation fails, retry once using a more conservative protected representation.
3. If validation still fails, reconstruct around fixed spans where practical.
4. If strict compliance remains impossible, return a controlled error with diagnostic metadata.

The response should distinguish:

- No glossary term matched.
- Glossary term matched and enforced.
- Glossary term matched but enforcement failed.
- Glossary was disabled.

## Persistence

### SQLite

SQLite is the best initial choice when there is one API instance, one system administrator, modest write volume, and a persistent deployment volume.

Advantages:

- No additional service.
- Transactional updates.
- Easy backup.
- Fits the standalone Docker deployment.

Do not keep the database only inside the container filesystem. Mount a persistent data directory.

### PostgreSQL

Use PostgreSQL when there are multiple API replicas, concurrent writes, external administration, multiple users, or a broader translation-management system.

### JSON or YAML

Suitable only for static deployment configuration. It lacks safe transactional updates, audit history, concurrent-write handling, and reliable container persistence.

### Import and Export

Enterprise providers commonly support TSV and CSV for terminology and TMX for translation memory. Supporting these formats would avoid locking administrators into this API's internal schema.

Recommended formats:

- TSV: simple unidirectional glossary import and export.
- CSV: multilingual or metadata-bearing glossary import and export.
- TMX 1.4: translation-memory import and export.

Imports should create a new resource version rather than mutating active entries incrementally. Validate the complete import before publishing it.

## Versioned Snapshots

Every glossary update should create a new version. A translation request should use one coherent snapshot from start to finish.

Recommended runtime design:

```text
database -> validated immutable snapshot -> in-memory matcher
```

On update:

1. Commit the database transaction.
2. Build and validate a new matcher.
3. Atomically replace the active snapshot.
4. Let new requests use the new version.
5. Let active requests finish with the snapshot they captured.

Do not query the database for every term during every translation request.

For large glossaries, use a trie or Aho-Corasick matcher. For small glossaries, sorted longest-first matching may be sufficient. Select based on measured glossary size and latency.

## Admin API

The current API has no authentication. Writable glossary endpoints must not be added without an authorization boundary.

Possible endpoints:

```text
GET    /admin/glossary
POST   /admin/glossary
PATCH  /admin/glossary/{id}
DELETE /admin/glossary/{id}
POST   /admin/glossary/reload
GET    /admin/glossary/versions
```

Authentication options:

- Reverse-proxy or identity-aware authentication.
- Static admin API key from a secret manager.
- JWT or OIDC when an existing identity system is available.

CORS is not authorization. It controls browser behavior, not API permissions.

Admin writes should:

- Validate language codes.
- Reject empty terms.
- Detect conflicting entries.
- Increment glossary version.
- Record audit information.
- Publish a new snapshot atomically.

## Response Metadata

Terminology metadata should eventually be returned:

```json
{
  "translation": "...",
  "system": "adapter",
  "source_lang": "en",
  "target_lang": "fa",
  "glossary_id": "technical-fa",
  "glossary_version": 12,
  "translation_memory_id": "manuals-fa",
  "translation_memory_version": 4,
  "applied_terms": [
    {
      "source_term": "multi-query attention",
      "target_term": "توجه چندپرسشی",
      "count": 1
    }
  ]
}
```

This supports debugging, auditability, and quality analysis without exposing database internals.

## Glossary Versus Translation Memory

The public API and internal model should keep these concepts separate.

### Glossary

A glossary controls words and phrases:

```text
multi-query attention -> توجه چندپرسشی
Google Home -> Google Home
```

Use it for product names, organization names, technical terms, legal terms, named entities, and fixed compound nouns.

### Translation Memory

Translation memory stores approved segment pairs:

```text
Source: Welcome to the control panel.
Target: به پنل کنترل خوش آمدید.
```

Recommended initial behavior:

- Exact normalized segment match: return stored target directly.
- High fuzzy match: use the stored target as a candidate or model context according to policy.
- Lower fuzzy match: do not replace output; optionally provide it as context.
- Record memory ID, version, match score, and application policy.

Do not implement fuzzy translation memory by searching glossary terms. It requires segment-level similarity retrieval and a threshold.

### Style and Context Instructions

Style preferences are also separate:

```text
Use formal Persian.
Prefer terminology used in medical documentation.
```

These instructions influence generation but are not deterministic mappings. They should not be represented as strict glossary entries.

## Exact Terms Versus Inflection

Exact terminology can conflict with grammatical correctness. Decide whether a target entry is:

- An exact immutable string.
- A base form that may be inflected.
- A preferred translation that the model may adapt.
- A context-dependent alternative.

For deterministic behavior, store exact immutable target strings. For linguistic flexibility, support a separate non-strict `preferred` mode. Do not mix both semantics under one boolean dictionary entry.

## Testing Requirements

Tests should cover:

- Exact phrase matching.
- Word boundaries.
- Unicode normalization.
- Case sensitivity.
- Overlapping terms.
- Repeated terms.
- Punctuation adjacency.
- Terms affected by sentence splitting.
- Batch requests.
- Multiple language pairs.
- Missing placeholders.
- Duplicated placeholders.
- Unknown placeholders.
- Model output that corrupts placeholders.
- Glossary updates during active translation.
- Strict-mode rejection.
- Snapshot consistency.
- Concurrent admin updates.
- Persistence after container restart.

The key property is:

```text
For every accepted strict translation, every matched source term maps to its exact target term.
```

## Open Decisions

Before implementation, decide:

1. Is glossary memory global to the deployment, or separated by tenant or project?
2. Must target terms be exact immutable strings, or may the model inflect them?
3. When a placeholder is corrupted or omitted, should the API retry, reconstruct, or return an HTTP error?
4. Will there be one API replica or multiple replicas?
5. Is authentication already provided by a reverse proxy or identity service?
6. Are entries scoped only by language pair, or also by domain or project?
7. Should existing `/translate` callers automatically use the glossary, or must callers opt into `terminology_mode`?

## Conclusion

The strongest practical design for this API is:

```text
named, versioned admin glossary
  -> exact translation-memory lookup
  -> longest-match source lookup
  -> placeholder protection
  -> existing TranslateGemma generation
  -> placeholder validation
  -> deterministic target restoration
```

Prompt hints and greedy decoding can improve consistency, but neither provides a hard guarantee. Deterministic behavior requires an enforcement layer around model generation.

Enterprise research reinforces four decisions:

1. Glossaries should be named resources selected explicitly by ID.
2. Translation memory must be a separate segment-reuse feature.
3. Contextual terminology and strict exact terminology need different modes.
4. Strict guarantees require output validation even when the model receives glossary instructions.
