"""PDF report generation for health analysis."""

from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF

from .data_utils import DataSummary, KNOWN_COLUMNS
from .models import RiskAssessment


class HealthReport(FPDF):
    """Custom PDF report for health analysis results."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, "AI Health Insight Agent", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        # Separator line
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(44, 62, 80)
        self.ln(3)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(189, 195, 199)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def subsection_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value(self, key: str, value: str, indent: int = 10) -> None:
        self.set_x(indent)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(55, 6, f"{key}:")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def risk_badge(self, risk_level: str, score: float) -> None:
        color_map = {
            "Low": (46, 204, 113),
            "Moderate": (243, 156, 18),
            "High": (231, 76, 60),
            "Critical": (192, 57, 43),
        }
        r, g, b = color_map.get(risk_level, (149, 165, 166))
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        badge_text = f"  {risk_level} Risk ({score}/100)  "
        w = self.get_string_width(badge_text) + 8
        x = (210 - w) / 2
        self.set_x(x)
        self.cell(w, 10, badge_text, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)


def generate_report(
    summary: DataSummary,
    assessment: RiskAssessment,
    sleep_analysis: dict | None = None,
) -> bytes:
    """Generate a comprehensive PDF health report.

    Args:
        summary: Data summary statistics.
        assessment: Risk assessment results.
        sleep_analysis: Optional sleep quality analysis results.

    Returns:
        PDF file as bytes.
    """
    pdf = HealthReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Overview ---
    pdf.section_title("Data Overview")
    pdf.key_value("Date Range", f"{summary.date_range[0]} to {summary.date_range[1]}")
    pdf.key_value("Total Records", str(summary.n_rows))
    pdf.key_value("Days Covered", str(summary.n_days))
    pdf.key_value("Metrics Tracked", str(len(summary.available_metrics)))
    pdf.ln(3)

    # --- Overall Risk ---
    pdf.section_title("Overall Health Risk Assessment")
    pdf.risk_badge(assessment.risk_level, assessment.overall_risk_score)
    pdf.ln(2)

    # --- Metric Details ---
    pdf.section_title("Detailed Metric Analysis")

    for col, mr in assessment.metric_risks.items():
        meta = KNOWN_COLUMNS.get(col, {})
        unit = meta.get("unit", "")

        pdf.subsection_title(mr.label)
        pdf.key_value("Current Value", f"{mr.current_value} {unit}")
        pdf.key_value("Healthy Range", f"{mr.healthy_range[0]} - {mr.healthy_range[1]} {unit}")
        pdf.key_value("Status", mr.status)
        pdf.key_value("Trend", mr.trend)
        pdf.key_value("Risk Score", f"{mr.risk_score}/100")
        pdf.ln(2)

    # --- Summary Statistics ---
    pdf.add_page()
    pdf.section_title("Summary Statistics")

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    col_widths = [45, 22, 22, 22, 22, 22, 35]
    headers = ["Metric", "Mean", "Std", "Min", "Max", "Median", "Healthy Range"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    for metric in summary.available_metrics:
        if metric not in summary.stats.index:
            continue
        row = summary.stats.loc[metric]
        meta = KNOWN_COLUMNS.get(metric, {})
        label = meta.get("label", metric)

        values = [
            label[:22],
            f"{row['mean']:.1f}",
            f"{row['std']:.1f}",
            f"{row['min']:.1f}",
            f"{row['max']:.1f}",
            f"{row['50%']:.1f}",
            f"{row['healthy_low']}-{row['healthy_high']}",
        ]
        for w, v in zip(col_widths, values):
            pdf.cell(w, 6, v, border=1, align="C")
        pdf.ln()

    pdf.ln(5)

    # --- Sleep Analysis ---
    if sleep_analysis and sleep_analysis.get("label") != "No data":
        pdf.section_title("Sleep Quality Analysis")
        details = sleep_analysis.get("details", {})
        pdf.key_value("Overall Score", f"{sleep_analysis['score']}/100 ({sleep_analysis['label']})")
        pdf.key_value("Average Duration", f"{details.get('average_hours', 'N/A')} hours")
        pdf.key_value("Consistency Score", f"{details.get('consistency_score', 'N/A')}/100")
        pdf.key_value("Nights in Range (7-9h)", f"{details.get('pct_in_range', 'N/A')}%")
        pdf.key_value("Nights Under 6h", f"{details.get('pct_under_6h', 'N/A')}%")
        pdf.ln(3)

    # --- Recommendations ---
    pdf.add_page()
    pdf.section_title("Personalized Recommendations")

    for i, rec in enumerate(assessment.recommendations, 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(8, 6, f"{i}.")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5.5, rec)
        pdf.ln(3)

    # --- Disclaimer ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.multi_cell(0, 4,
        "Disclaimer: This report is generated by an AI system for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for medical decisions."
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
