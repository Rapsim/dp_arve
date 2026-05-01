from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

doc = SimpleDocTemplate("Peak_Flow_Analysis.pdf", pagesize=A4)
styles = getSampleStyleSheet()
content = []

def add_section(title, data, text):
    content.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    content.append(Spacer(1,10))

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.5,colors.black),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")
    ]))

    content.append(table)
    content.append(Spacer(1,10))
    content.append(Paragraph(text, styles["Normal"]))
    content.append(Spacer(1,20))


# ================================
# 0-6h
# ================================
add_section(
"0–6h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","93","-11","0.57","0.82"],
["Hydrique_Curve","56","-5","0.84","0.84"],
["SIG_CNR","53","-2","0.89","0.90"],
["OFEV","82","+0.7","0.78","0.71"]
],
"""All models show strong performance at short lead times.
NSE values above 0.5 indicate that models explain most of the variance of observed peak flows.
SIG_CNR performs best, with NSE close to 0.9 and very low bias (-2%), meaning it reproduces both magnitude and variability accurately."""
)

# ================================
# 6-12h
# ================================
add_section(
"6–12h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","200","-33","-1.08","0.49"],
["Hydrique_Curve","93","-12","0.54","0.75"],
["SIG_CNR","99","-8","0.38","0.72"],
["OFEV","151","+3.7","-0.82","0.11"]
],
"""Hydrique_ML has NSE < 0, meaning it performs worse than using the mean observed peak (climatology).
This is due to strong underestimation (~ -33%).
SIG_CNR remains usable, as NSE > 0 indicates it still captures part of the variability."""
)

# ================================
# 12-18h
# ================================
add_section(
"12–18h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","219","-30","-1.43","0.33"],
["Hydrique_Curve","132","-14","0.15","0.69"],
["SIG_CNR","81","-6","0.47","0.74"],
["OFEV","158","+1.6","-2.28","-0.09"]
],
"""Strong divergence appears between models.
Hydrique_ML and OFEV have strongly negative NSE values, indicating very poor predictive skill.
SIG_CNR remains the most reliable, with NSE ~0.47 meaning it still captures peak variability."""
)

# ================================
# 18-24h
# ================================
add_section(
"18–24h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","193","-27","-0.91","0.44"],
["Hydrique_Curve","139","-18","0.05","0.68"],
["SIG_CNR","120","-15","0.48","0.72"],
["OFEV","123","-2.7","0.43","0.69"]
],
"""Hydrique_Curve shows NSE close to 0, meaning it performs similarly to a simple mean estimate.
SIG_CNR remains the most robust model.
OFEV shows moderate performance with positive NSE."""
)

# ================================
# 24-36h
# ================================
add_section(
"24–36h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","242","-38","-1.44","0.33"],
["Hydrique_Curve","141","-20","0.20","0.73"],
["SIG_CNR","148","-21","-0.05","0.65"],
["OFEV","112","-9","0.49","0.54"]
],
"""Hydrique_ML has NSE < 0, indicating no predictive usefulness.
SIG_CNR begins to lose skill (NSE slightly negative).
OFEV becomes the best model in this window with positive NSE."""
)

# ================================
# 36-48h
# ================================
add_section(
"36–48h Lead Time",
[
["Model","RMSE","Bias %","NSE","KGE"],
["Hydrique_ML","248","-39","-1.57","0.30"],
["Hydrique_Curve","170","-27","-0.13","0.68"],
["SIG_CNR","154","-19","0.04","0.59"],
["OFEV","161","-16","-0.15","0.40"]
],
"""All models show NSE values near or below zero, indicating very low predictive skill.
Persistent negative bias shows systematic underestimation of flood peaks."""
)

# ================================
# CONCLUSION
# ================================
content.append(PageBreak())
content.append(Paragraph("<b>Overall Conclusion</b>", styles["Heading2"]))
content.append(Spacer(1,10))

content.append(Paragraph(
"""Model performance decreases significantly with increasing lead time.
Short-term forecasts are reliable, while long-term forecasts lose predictive skill.
Negative NSE values indicate performance worse than a climatological estimate.
SIG_CNR is the most robust model overall, while Hydrique_ML shows strong systematic underestimation.""",
styles["Normal"]
))

doc.build(content)

print("PDF generated: Peak_Flow_Analysis.pdf")