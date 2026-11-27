from slopewatch.core import make_demo_comparison
from slopewatch.report import build_report_pdf


def test_demo_report_generates_pdf_with_chinese_text():
    comparison = make_demo_comparison()
    content = build_report_pdf(comparison.metrics, comparison.gullies, {"source": "synthetic_demo"})
    assert content.startswith(b"%PDF-")
    assert len(content) > 10_000

