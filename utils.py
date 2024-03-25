import cv2
import numpy as np

QUANTIZATION_TABLE = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
)
#QUANTIZATION_TABLE = np.array([[16, 11, 10, 16, 24, 40, 51, 61, 16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55,12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56,4, 13, 16, 24, 40, 57, 69, 56],
#    [14, 17, 22, 29, 51, 87, 80, 62,14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77,18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92,24, 35, 55, 64, 81, 104, 113, 92],
#    [49, 64, 78, 87, 103, 121, 120, 101,49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99,72, 92, 95, 98, 112, 100, 103, 99],[16, 11, 10, 16, 24, 40, 51, 61, 16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55,12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56,4, 13, 16, 24, 40, 57, 69, 56],
#    [14, 17, 22, 29, 51, 87, 80, 62,14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77,18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92,24, 35, 55, 64, 81, 104, 113, 92],
#    [49, 64, 78, 87, 103, 121, 120, 101,49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99,72, 92, 95, 98, 112, 100, 103, 99]])

# QUANTIZATION_TABLE = [[1, 1, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2, 2]
#    , [1, 1, 1, 1, 2, 2, 3, 2], [1, 1, 1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 3, 4, 4, 3], [1, 1, 2, 3, 3, 4, 5, 4],
#                      [2, 3, 3, 3, 4, 5, 5, 4], [3, 4, 4, 4, 4, 4, 4, 4]]
BLOCK_LENGTH = 8
CHANNELS = 3
SCALE = 128.0
INTEGER_DTYPE_SIGNED = np.int8
INTEGER_DTYPE_UNSIGNED = np.uint8
FLOAT_DTYPE = np.float64
SECOND_BYTE_COEFFICIENT = 256
ONES = np.ones((BLOCK_LENGTH, BLOCK_LENGTH))
IMAGE_FORMAT = cv2.COLOR_BGR2YCrCb
INVERSE_IMAGE_FORMAT = cv2.COLOR_YCrCb2BGR


def get_incremented_coords_3dim(row, col, channel, width, length) -> tuple:
    channel = (channel + 1) % 3
    if channel == 0:
        col += 1
        if col == width:
            row += 1
            col = 0
    return (row, col, channel)


def get_incremented_coords_2dim(row, col, width) -> tuple:
    if col == width - 1:
        return row + 1, 0
    return row, col + 1


def is_on_edge(index) -> bool:
    return index == 0 or index == BLOCK_LENGTH - 1


def is_on_point(max_index, min_index) -> bool:
    return max_index == BLOCK_LENGTH - 1 and min_index == 0


def new_phase_values(phase, index) -> tuple:
    return phase * -1, index + 1
