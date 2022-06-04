import pathlib
import pdf2image

pdf_files = pathlib.Path("pdfs").glob("*.pdf")
img_dir = pathlib.Path("out_img")

for i, pdf_file in enumerate(pdf_files):
    base = pdf_file.stem
