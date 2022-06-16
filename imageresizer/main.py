import cv2

def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir

def scale_box(img, width, height):
    """指定した大きさに収まるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)

    dst = cv2.resize(img, dsize=(nw, nh))

    return dst

def main():
    for i in range(52):
        cards_path: str = f"{get_script_dir()}/../pdf2img/img/"
        card = cv2.imread(f"{cards_path}{i}.jpg")
        dsize = (360, 585)
        card_img = cv2.resize(card, dsize)
        x = cv2.imwrite(f"{cards_path}/{i}.jpg", card_img)
        print(x)
if __name__ == "__main__":
    main()

