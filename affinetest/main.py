import cv2
import matplotlib.pyplot as plt
import numpy as np
def identity(image):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    affine = cv2.getAffineTransform(src, src)
    return cv2.warpAffine(image, affine, (w, h))

if __name__ == "__main__":
    image = cv2.imread("gorilla.jpg")[:,:,::-1]
    converted = identity(image)
    plt.imshow(converted)
    plt.title("Identity")
    plt.show()