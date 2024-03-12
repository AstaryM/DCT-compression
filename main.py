import Encoder
import os
import Decoder
from utils import *

if __name__ == '__main__':
    # line = input("Please enter image path:\n")
    image = cv2.imread("test_image2.bmp", cv2.IMREAD_COLOR)
    length, width = image.shape[:2]
    result = cv2.copyMakeBorder(image, 0, BLOCK_LENGTH - (length % BLOCK_LENGTH), 0,
                                BLOCK_LENGTH - (width % BLOCK_LENGTH), cv2.BORDER_REPLICATE)
    length, width = result.shape[:2]
    # result = colorTreatment.color_treatment(result, length, width)
    # line = input("Please enter new image name:\n")
    Encoder.encode_image(result, length, width, file_name="temp")
    old_size = os.path.getsize("test_image2.bmp")
    new_size = os.path.getsize("temp")
    print(f"compression process ended, new file size is {np.float16(100 * new_size / old_size)}% of the old file size")
    Decoder.decoder("temp")
