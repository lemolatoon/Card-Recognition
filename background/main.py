from icrawler.builtin import BingImageCrawler
import os
import shutil
from pathlib import Path


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    word: str = "floor"
    keywords = ["floor", "desk", "table", "ground"]
    crawler = BingImageCrawler(downloader_threads=8,
        storage={"root_dir": f"{get_script_dir()}/../images/background/{keywords[3]}"})
    crawler.crawl(keyword=keywords[3], max_num=1000, offset=0)
    crawler = BingImageCrawler(downloader_threads=8,
        storage={"root_dir": f"{get_script_dir()}/../images/background/{keywords[2]}"})
    crawler.crawl(keyword=keywords[2], max_num=1000, offset=0)
    crawler = BingImageCrawler(downloader_threads=8,
            storage={"root_dir": f"{get_script_dir()}/../images/background/{keywords[1]}"})
    crawler.crawl(keyword=keywords[1], max_num=1000, offset=0)
    crawler = BingImageCrawler(downloader_threads=8,
        storage={"root_dir": f"{get_script_dir()}/../images/background/{keywords[0]}"})
    crawler.crawl(keyword=keywords[0], max_num=1000, offset=0)

    os.makedirs(f"{get_script_dir()}/../images/background/root", exist_ok=True)
    base_dir = f"{get_script_dir()}/../images/background/"
    pathes = os.listdir(base_dir)
    count: int = 0
    for path in pathes:
        for img_path in Path(f"{base_dir}{path}").glob("*.jpg"):
            shutil.move(img_path, f"{base_dir}root/{count:06}.jpg")
            count += 1
    


if __name__ == "__main__":
    main()
