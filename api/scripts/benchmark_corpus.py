"""Fixed English source material for the speed benchmark.

Kept in the image rather than fetched, so a benchmark on an offline deployment
host measures the same inputs as one anywhere else. Two properties matter:

* The order alternates short / medium / long sentences, so *any prefix* of
  ``SENTENCES`` is length-mixed. The sweep takes the first N sentences for a
  batch of N, and a batch of uniformly short sentences would report a
  throughput the real workload never sees (generation runs until the *longest*
  row in the batch stops, so padding waste is part of the measurement).
* The domain is scientific/technical prose, which is what the SFT adapter was
  trained on. Benchmarking on out-of-domain text changes output length, and
  output length is the denominator of every number in the report.
"""

# Alternating lengths: short, medium, long, repeating.
SENTENCES = [
    "The results were reproducible.",
    "Each sample was incubated for thirty minutes before the second measurement was taken.",
    "Although the initial hypothesis predicted a monotonic decline in enzymatic activity across "
    "the full temperature range, the observed response was biphasic, with a pronounced recovery "
    "above forty degrees that the authors attribute to a conformational change in the binding "
    "pocket rather than to any change in substrate availability.",
    "Statistical power was limited.",
    "The model relies on multi-query attention to process the genome sequence efficiently.",
    "Participants who completed the full protocol showed a statistically significant improvement "
    "on the primary endpoint, but the effect size was modest and the confidence interval wide "
    "enough that a clinically meaningful benefit cannot be established from this trial alone, a "
    "limitation the discussion section acknowledges only briefly.",
    "No adverse events were reported.",
    "Gene expression was quantified by RNA sequencing across four biological replicates.",
    "The proposed architecture replaces the dense feed-forward block with a sparsely gated "
    "mixture of experts, which increases parameter count by an order of magnitude while keeping "
    "the per-token computational cost roughly constant, and the authors show that the resulting "
    "model matches a much larger dense baseline on every downstream benchmark they evaluate.",
    "Data are available on request.",
    "Calibration curves were linear across the concentration range under study.",
    "Because the catalyst degrades rapidly under the reaction conditions used in earlier work, "
    "the authors introduce a protective ligand that suppresses aggregation without measurably "
    "reducing turnover frequency, and they demonstrate that the modified system retains eighty "
    "percent of its initial activity after twenty consecutive cycles.",
    "The effect disappeared after correction.",
    "Two independent reviewers screened all titles and abstracts against the inclusion criteria.",
    "Long-term follow-up data from the extension phase suggest that the treatment effect observed "
    "at twelve months is largely sustained through the third year, although attrition was "
    "substantial and the analysis population differs enough from the original cohort that the "
    "comparison should be read as descriptive rather than confirmatory.",
    "Sequencing depth exceeded thirty-fold.",
    "The instrument was recalibrated before every batch of measurements.",
    "A sensitivity analysis excluding the two centers with the highest protocol deviation rates "
    "produced point estimates consistent with the primary analysis, which the authors interpret "
    "as evidence that the overall finding is not driven by data quality problems at any single "
    "site, though the analysis was not prespecified.",
    "The sample was stored at minus eighty degrees.",
    "Antibody titers were measured at baseline and again at day twenty-eight.",
    "Simulations were run on a cluster of sixty-four nodes, each equipped with four accelerators, "
    "and the reported wall-clock times exclude the initial data-loading phase, which the authors "
    "note dominates total runtime for the smallest problem sizes and becomes negligible above "
    "roughly one million particles.",
    "Funding had no role in study design.",
    "Cell viability was assessed using a standard colorimetric assay.",
    "The authors conclude that the mechanism they describe is unlikely to be species-specific, "
    "citing structural conservation of the relevant domain across vertebrates, but they stop "
    "short of claiming direct clinical relevance and explicitly call for replication in a "
    "primate model before any translational work proceeds.",
]

# Continuous prose for the synthetic "page": a real page is a paragraph flow,
# not a list of unrelated sentences, and sentence splitting behaves differently
# on the two.
PARAGRAPHS = [
    "Recent advances in machine translation have narrowed the gap between general-purpose "
    "systems and domain-specialized ones, but scientific prose remains a difficult case. "
    "Terminology is dense, sentences are long, and a single mistranslated unit or gene name "
    "can invert the meaning of a result. This paper examines whether parameter-efficient "
    "fine-tuning on a modest in-domain corpus is sufficient to close that gap.",
    "We assembled a parallel corpus of abstracts and method sections drawn from open-access "
    "journals across four disciplines. Each pair was filtered for alignment quality and "
    "manually reviewed in a random sample of five hundred segments. The resulting dataset is "
    "smaller than those used in comparable studies, which is deliberate: the question is what "
    "can be achieved under realistic data constraints rather than at scale.",
    "Training used low-rank adaptation applied to the attention projections only, leaving the "
    "feed-forward layers untouched. This keeps the adapter small enough to ship alongside the "
    "base checkpoint and makes it possible to serve both the adapted and the unadapted system "
    "from one copy of the weights. Hyperparameters were selected on a held-out development "
    "split and then frozen for all reported runs.",
    "Evaluation combines automatic metrics with a targeted human review of terminology "
    "handling. We report both because the two disagree in an informative way: the automatic "
    "scores improve modestly, while reviewers judged the adapted output substantially more "
    "usable, largely because it stopped translating established technical terms into "
    "descriptive paraphrases.",
    "The remaining errors cluster in three categories. Numbers and units are occasionally "
    "reordered in ways that change meaning. Citations embedded mid-sentence are sometimes "
    "dropped. And very long sentences, above roughly sixty tokens, show a measurable increase "
    "in omission rate, which suggests that segment length rather than domain difficulty is the "
    "binding constraint for the current system.",
]
