import Encoder
import os
import Decoder
from utils import *
import sys

def encode(image_path):
    encoded_file_name = os.path.splitext(image_path)[0] + '.ast'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    old_size = os.path.getsize("test_image2.bmp")
    new_size = os.path.getsize("temp")
    print(f"compression process ended, new file size is {np.float16(100 * new_size / old_size)}% of the old file size")
    cv2.imwrite()


def decode(image_path):
    decoded_file_name = os.path.splitext(image_path)[0]+'.tsa'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    decoded_image = Decoder.decoder(image_path)
    cv2.imwrite(decoded_image)
    #cv2.imshow("temp", result_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    n = len(sys.argv)
    if len(sys.argv) != 3:
        print("error! invalid amount of arguments")
        exit
    if sys.argv[1] == "encode":
        encode(sys.argv[2])
    elif sys.argv[1] == "decode":
        decode(sys.argv[2])
    else:
        print("error! invalid command")


