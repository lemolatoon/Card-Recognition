import openpyxl


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    book_name: str = "card_img"
    file_path: str = f"{get_script_dir()}/../{book_name}.xlsx"
    book = pd.ExcelFile(file_path)


if __name__ == "__main__":
    main()
