import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

image_path = "E:\\2022work\experiment\\2-yolo\yolov4-infrared-master\yolov4-pytorch-master\img\\6.jpg"


def random_crop(image, crop_shape, padding=None):
    oshape = np.shape(image)

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)

        npad = ((padding, padding), (padding, padding), (0, 0))

        image_pad = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        image_crop = image_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]

        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image


if __name__ == "__main__":
    image_src = Image.open(image_path)
    crop_width = image_src.size[0] - 10
    crop_height = image_src.size[1] - 10
    image_dst_crop = random_crop(image_src, [crop_width, crop_height], padding=100)
    plt.imshow(image_dst_crop)
    plt.savefig('E:\\2022work\experiment\\2-yolo\yolov4-infrared-master\yolov4-pytorch-master\img\cropped.jpg')

    plt.figure()
    plt.subplot(221)
    plt.imshow(image_src)
    plt.title("oringin image")
    plt.subplot(222)
    plt.title("crop image")
    plt.show()