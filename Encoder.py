import numpy as np

from utils import *

import numpy as np

from utils import *


def encoder(image, file_name):
    temp_length, temp_width = image.shape[:2]
    result = cv2.copyMakeBorder(image, 0, BLOCK_LENGTH - (temp_length % BLOCK_LENGTH), 0,
                                BLOCK_LENGTH - (temp_width % BLOCK_LENGTH), cv2.BORDER_REPLICATE)
    length, width = result.shape[:2]
    cv2.cvtColor(src=image, dst=image, code=IMAGE_FORMAT)

    result_image = np.empty((length, width, CHANNELS), dtype=INTEGER_DTYPE_SIGNED)
    for channel in range(CHANNELS):
        for i in range(int(length / BLOCK_LENGTH)):

            block_row_start = i * BLOCK_LENGTH
            block_row_end = i * BLOCK_LENGTH + BLOCK_LENGTH

            for j in range(int(width / BLOCK_LENGTH)):
                block_column_start = j * BLOCK_LENGTH
                block_column_end = j * BLOCK_LENGTH + BLOCK_LENGTH

                block = image[block_row_start: block_row_end, block_column_start: block_column_end, channel]
                result_image[block_row_start: block_row_end, block_column_start: block_column_end,
                channel] = encode_block(
                    block)
    RLE_to_file(result_image, length, width, file_name)


def encode_block(block):
    quantization_table = QUANTIZATION_TABLE
    processed_block = cv2.dct(np.asarray(block, dtype=FLOAT_DTYPE) / SCALE) * SCALE
    processed_block = np.divide(processed_block, quantization_table)
    processed_block = np.asarray(processed_block, dtype=INTEGER_DTYPE_SIGNED)
    processed_block = zigzager(processed_block)
    return processed_block


def zigzager(block):
    result = np.empty(shape=(BLOCK_LENGTH, BLOCK_LENGTH), dtype=INTEGER_DTYPE_SIGNED)
    phase = - 1
    i = 0
    j = 0
    col = 0
    row = 0
    while i < BLOCK_LENGTH != j < BLOCK_LENGTH:

        result[row, col] = block[i, j]
        row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)

        if is_on_edge(i) and not is_on_point(j, i):
            phase, j = new_phase_values(phase, j)
            result[row, col] = block[i, j]

            row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)
        elif is_on_edge(j) and not is_on_point(i, j):
            phase, i = new_phase_values(phase, i)
            result[row, col] = block[i, j]
            row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)

        i += phase
        j -= phase

    return result


def RLE_to_file(image, length, width, file_name):
    counter = 1
    pos = 4
    # with open(file_name, "wb") as file:
    result = np.empty(length * width)
    result[0:2] = length.to_bytes(2, byteorder="little")
    result[2:4] = width.to_bytes(2, byteorder="little")
    image = image.flatten()
    for i in range(image.shape[0]):
        if i < image.shape[0] - 1 and image[i] == image[i + 1]:
            counter += 1
        else:
            result[pos] = image[i]
            pos += 1
            if counter > 1:
                result[pos] = image[i]
                pos += 1
                result[pos:pos + 2] = counter.to_bytes(2, byteorder="little")
                pos += 2
            counter = 1
    return result
