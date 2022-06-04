from icrawler.builtin import BingImageCrawler


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    crawler = BingImageCrawler(
        storage={"root_dir": f"{get_script_dir()}/../images/backgrounds"})
    crawler.crawl(keyword="background", max_num=10000)


if __name__ == "__main__":
    main()
