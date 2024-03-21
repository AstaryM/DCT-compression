import Encoder
import os
import Decoder
from utils import *
import argparse


def encode(image_path):
    encoded_file_name = os.path.splitext(image_path)[0] + '.ast'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    encoded_image = Encoder.encoder(image)
    with open(encoded_file_name, "w"):
        encoded_image.tofile(encoded_file_name)
        old_size = os.path.getsize(image_path)
        new_size = os.path.getsize(encoded_file_name)
        print(
            f"compression process ended, new file size is {np.float16(100 * new_size / old_size)}% of the old file size")


def decode(image_path):
    decoded_file_name = os.path.splitext(image_path)[0] + '.tsa'
    encoded_image = np.fromfile(image_path, dtype=INTEGER_DTYPE_UNSIGNED)
    decoded_image = Decoder.decoder(encoded_image)
    decoded_image.tofile(decoded_file_name)
    cv2.imshow("temp", decoded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode or decode image using DCT based compression.', epilog="")
    parser.add_argument('file', type=str)
    parser.add_argument('action', choices=('encode', 'decode'), help='an action to be performed on the file')
    args = parser.parse_args()
    file_path = args.file
    action = args.action

    if action == 'encode':
        encode(file_path)
    if action == 'decode':
        decode(file_path)
# n = len(sys.argv)
# if len(sys.argv) != 3:
#    print(f"error! invalid amount of arguments. got {n} expected 3")
#    exit()
# if sys.argv[1] == "encode":
#    encode(sys.argv[2])
# elif sys.argv[1] == "decode":
#    decode(sys.argv[2])
# else:
#    print("error! invalid command")
#
