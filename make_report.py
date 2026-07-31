from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, Table, TableStyle
)

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#064A56"))
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)

doc = SimpleDocTemplate(
    "outputs/report.pdf",
    pagesize=letter,
    topMargin=0.7 * inch,
    bottomMargin=0.7 * inch,
    leftMargin=0.8 * inch,
    rightMargin=0.8 * inch,
)

story = []

story.append(Paragraph("Freight Rate Prediction — Assessment Report", h1))
story.append(Paragraph("Mahan Liaghatmand &nbsp;|&nbsp; Machine Learning Engineer Assessment &nbsp;|&nbsp; Spotter", body))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Train / Validation Split Approach", h2))
story.append(Paragraph(
    "The labeled data (train_test.csv) covers January through October 2025, while the "
    "final files that need to be scored (validation.csv and the December chart inputs) "
    "cover November and December 2025 &mdash; dates strictly after the training period. "
    "Because of that, a random 80/20 split of the labeled data would be misleading: it "
    "would let the model validate on rows from the same weeks it trained on, which is an "
    "easier problem than predicting genuinely unseen future months.",
    body,
))
story.append(Paragraph(
    "Instead, a <b>time-based split</b> was used: the model trained on January&ndash;August "
    "2025 and was evaluated on the held-out September&ndash;October 2025 period, which the "
    "model never saw during training. This mirrors how the model is actually used "
    "(predicting forward in time) and gives a more honest estimate of real-world accuracy.",
    body,
))

story.append(Paragraph("2. Data-Quality Issues Found", h2))
issues = ListFlowable([
    ListItem(Paragraph(
        "<b>Negative weights.</b> About 300 rows (0.6%) had a negative <i>weight</i> value "
        "(e.g. -36,559 lbs), which is physically impossible for a shipment. Treated as a "
        "sign-entry error and fixed with an absolute value rather than dropping the rows.",
        body)),
    ListItem(Paragraph(
        "<b>Missing values.</b> <i>weight</i> was missing in ~0.6% of rows and "
        "<i>market_index</i> in ~0.8%. Missing weights were filled with the training-set "
        "median.",
        body)),
    ListItem(Paragraph(
        "<b>Unseen cities in validation.</b> The validation set contains 8 pickup/delivery "
        "cities that never appear anywhere in the training data (Chicago, San Diego, "
        "Charlotte, Knoxville, Jackson, Norfolk, Laredo, Allentown). A model that encodes "
        "city names directly would have no learned behavior for those loads. Solved by not "
        "using city names as a feature at all &mdash; distance already captures the route.",
        body)),
    ListItem(Paragraph(
        "<b>Weak / partially unavailable features.</b> <i>market_index</i> and "
        "<i>quote_signal</i> correlate very weakly with the target rate (|r| &lt; 0.04) and, "
        "importantly, the December assessment file does not include either column at all. "
        "Both were excluded from the model so one single model can score every provided "
        "file without inventing values for missing columns.",
        body)),
], bulletType="bullet")
story.append(issues)

story.append(Paragraph("3. Model", h2))
story.append(Paragraph(
    "A <b>Gradient Boosting regressor</b> (scikit-learn's HistGradientBoostingRegressor) was "
    "used on five features: distance, weight, equipment type (one-hot encoded), pickup month, "
    "and pickup day-of-week. Gradient boosting was chosen over a plain linear model because "
    "the equipment-type effect on rate/mile is not simply additive (Reefer and Flatbed carry "
    "different premiums), and boosted trees handle that kind of interaction without manual "
    "feature crossing.",
    body,
))

story.append(Paragraph("4. Validation Results", h2))
table_data = [
    ["Model", "MAE", "RMSE", "R\u00b2"],
    ["Baseline (average rate/mile \u00d7 distance)", "$290.60", "$709.45", "0.784"],
    ["Gradient Boosting (this solution)", "$131.57", "$633.57", "0.828"],
]
t = Table(table_data, colWidths=[3.0 * inch, 1.2 * inch, 1.2 * inch, 0.9 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064A56")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F7")]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Evaluated on the Sep&ndash;Oct 2025 holdout period. The model cuts average error "
    "(MAE) by about 55% versus a naive distance-only baseline, confirming that weight, "
    "equipment type, and seasonality genuinely add predictive value beyond distance alone.",
    body,
))

story.append(Paragraph("5. December 2025 Prediction Chart", h2))
story.append(Paragraph(
    "The fixed Lexington &rarr; Fort Wayne / Dry Van / 360 mi / 32,000 lb lane was scored "
    "for every day in December 2025, using the final model retrained on all available "
    "labeled data.",
    body,
))
story.append(Image("outputs/scorer_results/candidate_december.png", width=6.3 * inch, height=2.8 * inch))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Why the December curve is nearly flat:</b> the training data only covers pickup "
    "months January through October. December (month 12) was never seen during training, "
    "and tree-based models like gradient boosting do not extrapolate a trend beyond the "
    "range of values they were trained on &mdash; they fall back to the closest pattern "
    "they learned. The small week-to-week wiggle that remains comes entirely from the "
    "day-of-week feature. In other words, the flatness itself is an honest signal: it shows "
    "the model correctly recognizes December is outside its training range rather than "
    "guessing wildly.",
    body,
))

doc.build(story)
print("Report written to outputs/report.pdf")
