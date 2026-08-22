"""Status derivation in glossary.serializers.to_report.

Only the fully-applied path was exercised via the seam test before this file
existed; `partial` and `enforcement_failed` are user-visible statuses a caller
branches on, and were untested.
"""

from glossary.apply import Applied, Miss, Report, Violation
from glossary.serializers import to_report


def test_no_report_produces_the_no_match_status():
    assert to_report(None).status == "no_match"


def test_an_empty_report_produces_the_no_match_status():
    assert to_report(Report.empty()).status == "no_match"


def test_only_applied_terms_produce_the_applied_status():
    report = Report(
        applied=(Applied(source_term="genome", target_term="ژنوم", mode="preferred", count=1),),
        misses=(),
        violations=(),
    )
    result = to_report(report)
    assert result.status == "applied"
    assert [item.source_term for item in result.applied] == ["genome"]


def test_applied_terms_alongside_a_miss_produce_the_partial_status():
    report = Report(
        applied=(Applied(source_term="genome", target_term="ژنوم", mode="preferred", count=1),),
        misses=(Miss(source_term="allele", reason="not_found"),),
        violations=(),
    )
    result = to_report(report)
    assert result.status == "partial"
    assert [item.source_term for item in result.applied] == ["genome"]
    assert [item.source_term for item in result.misses] == ["allele"]


def test_only_misses_with_no_applied_terms_produce_the_enforcement_failed_status():
    report = Report(
        applied=(),
        misses=(Miss(source_term="allele", reason="not_found"),),
        violations=(),
    )
    result = to_report(report)
    assert result.status == "enforcement_failed"


def test_violations_are_carried_through_regardless_of_status():
    report = Report(
        applied=(Applied(source_term="genome", target_term="ژنوم", mode="preferred", count=1),),
        misses=(),
        violations=(Violation(source_term="genome", forbidden="genom"),),
    )
    result = to_report(report)
    assert [item.forbidden for item in result.violations] == ["genom"]
