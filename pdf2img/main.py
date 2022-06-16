from typing import List
from pathlib import Path
import pdf2image
import PyPDF2
import os
import cv2


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    path_0_24 = f"{get_script_dir()}/pdf/0-24.pdf"
    path_25_49 = f"{get_script_dir()}/pdf/25-49.pdf"
    path_50_51 = f"{get_script_dir()}/pdf/50-51.pdf"

    path_tmp_pdf = f"{get_script_dir()}/tmp_pdf"
    path_tmp_jpg = f"{get_script_dir()}/tmp_jpg"
    path_img = f"{get_script_dir()}/img"

    def rm_dirs(path: str):
        for p in Path(path).glob("*"):
            if p.is_dir():
                rm_dirs(p)
            else:
                os.remove(p)
        if Path(path).is_dir():
            os.rmdir(path)
    rm_dirs(path_img)
    os.makedirs(path_tmp_pdf, exist_ok=True)
    os.makedirs(path_tmp_jpg, exist_ok=True)
    os.makedirs(path_img, exist_ok=True)
    split_to_pdf([path_0_24, path_25_49, path_50_51], path_tmp_pdf)
    pdf2images(path_tmp_pdf, path_tmp_jpg)
    jpg_resize(path_tmp_jpg, path_img)


def split_to_pdf(pdf_pathes: List[str], out_dir: str):

    image_index: int = 0
    for pdf_path in pdf_pathes:
        reader = PyPDF2.PdfFileReader(pdf_path)
        num_pages = reader.getNumPages()

        for i in range(num_pages):
            page = reader.getPage(i)
            writer = PyPDF2.PdfFileWriter()
            writer.addPage(page)
            with open(f"{out_dir}/{image_index}.pdf", mode="wb") as f:
                writer.write(f)
            image_index += 1


def pdf2images(pdf_dir: str, out_dir: str):
    pathes = [f"{pdf_dir}/{i}.pdf" for i in range(52)]
    out_format = "jpeg"
    for i, path in enumerate(pathes):
        image = pdf2image.convert_from_path(path, dpi=50, fmt=out_format)
        image = image[0]
        image.save(f"{out_dir}/{i}.jpg")


def jpg_resize(jpg_dir: str, out_dir: str):
    pathes = [f"{jpg_dir}/{i}.jpg" for i in range(52)]
    for i, path in enumerate(pathes):
        image = cv2.imread(path)
        dsize = (360, 590)
        image = cv2.resize(image, dsize)
        cv2.imwrite(f"{out_dir}/{i}.jpg", image)


if __name__ == "__main__":
    main()
