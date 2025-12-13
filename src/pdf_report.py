from fpdf import FPDF
import pandas as pd
import os

def generate_pdf_report(df, output_path="artifacts/NIDS_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "AI NIDS Prediction Report", ln=True, align="C")

    pdf.ln(10)

    # Summary
    pdf.set_font("Arial", "", 12)
    total = len(df)
    attacks = len(df[df['prediction'] != "normal"])
    normals = len(df[df['prediction'] == "normal"])

    pdf.cell(200, 10, f"Total Records: {total}", ln=True)
    pdf.cell(200, 10, f"Normal Traffic: {normals}", ln=True)
    pdf.cell(200, 10, f"Attack Traffic: {attacks}", ln=True)
    pdf.ln(10)

    # Table of 20 sample predictions
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Sample Predictions:", ln=True)
    pdf.set_font("Arial", "", 10)

    sample = df.head(20)

    for idx, row in sample.iterrows():
        pdf.cell(200, 8, f"{idx+1}. Prediction: {row['prediction']}", ln=True)

    # Ensure artifacts folder exists
    os.makedirs("artifacts", exist_ok=True)

    pdf.output(output_path)
    return output_path
