from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(data):

    file = "report.pdf"

    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("ECG REPORT", styles["Title"]))
    content.append(Paragraph(f"User: {data['username']}", styles["Normal"]))
    content.append(Paragraph(f"Prediction: {data['prediction']}", styles["Normal"]))
    content.append(Paragraph(f"Heart Rate: {data['heart_rate']}", styles["Normal"]))
    content.append(Paragraph(f"Stress: {data['stress']}", styles["Normal"]))
    content.append(Paragraph(f"Risk: {data['risk']}", styles["Normal"]))

    doc.build(content)

    return file