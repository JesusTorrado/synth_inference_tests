import sys

from synth_inference_tests.get_pdf import get_pdf

if __name__ == "__main__":
    # Build PDF
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass likelihood name as first arg, e.g. 'gaussian5'")
    pdf_name = sys.argv[1]
    pdf = get_pdf(pdf_name)
    pdf.triangle_plot()
