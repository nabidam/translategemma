"""Convert the pure-dataclass Report into the API's Pydantic shape.

Kept out of apply.py on purpose: the matching and rewriting code has no
dependency on FastAPI or Pydantic, which is what lets it be tested without the
web stack.
"""

from schemas import AppliedTerm, ForbiddenTerm, GlossaryReport, MissedTerm

from .apply import Report


def to_report(report: Report | None) -> GlossaryReport:
    if report is None or report.is_empty:
        return GlossaryReport(status="no_match")
    if report.misses and not report.applied:
        status = "enforcement_failed"
    elif report.misses:
        status = "partial"
    else:
        status = "applied"
    return GlossaryReport(
        status=status,
        applied=[
            AppliedTerm(
                source_term=item.source_term,
                target_term=item.target_term,
                mode=item.mode,
                count=item.count,
            )
            for item in report.applied
        ],
        misses=[
            MissedTerm(source_term=item.source_term, reason=item.reason)
            for item in report.misses
        ],
        violations=[
            ForbiddenTerm(source_term=item.source_term, forbidden=item.forbidden)
            for item in report.violations
        ],
    )
