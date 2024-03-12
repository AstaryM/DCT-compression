from utils import *


def decoder(image_file_name):
    with open(image_file_name, 'rb') as compressed_file:
        image_data = compressed_file.read()

        length = image_data[0] + image_data[1] * SECOND_BYTE_COEFFICIENT
        width = image_data[2] + image_data[3] * SECOND_BYTE_COEFFICIENT

        image_data = image_data[4:]

        result_image = np.empty((length, width, CHANNELS), dtype=INTEGER_DTYPE_UNSIGNED)
        encoded_image = anti_RLE(image_data, length, width)
        for channel in range(CHANNELS):
            for i in range(int(length / BLOCK_LENGTH)):
                block_row_start = i * BLOCK_LENGTH
                block_row_end = i * BLOCK_LENGTH + BLOCK_LENGTH

                for j in range(int(width / BLOCK_LENGTH)):
                    block_column_start = j * BLOCK_LENGTH
                    block_column_end = j * BLOCK_LENGTH + BLOCK_LENGTH

                    block = encoded_image[block_row_start: block_row_end, block_column_start: block_column_end, channel]
                    result_image[block_row_start: block_row_end, block_column_start: block_column_end,
                    channel] = decode_block(block)
        cv2.cvtColor(src=result_image, dst=result_image, code=INVERSE_IMAGE_FORMAT)
        cv2.imshow("temp", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def decode_block(block):
    quantization_table = QUANTIZATION_TABLE
    block = anti_zigzager(block)
    block = np.multiply(block, quantization_table)
    block = cv2.idct(block.astype(FLOAT_DTYPE) / SCALE) * SCALE
    block = np.clip(block, a_min=0, a_max=None)
    block = np.asarray(block, dtype=INTEGER_DTYPE_UNSIGNED)
    return block


def anti_RLE(image_data, length, width):
    i = 0
    row = 0
    col = 0
    channel = 0
    encoded_image = np.zeros(shape=(length, width, CHANNELS), dtype=INTEGER_DTYPE_SIGNED)
    while i < len(image_data):
        encoded_image[row, col, channel] = INTEGER_DTYPE_SIGNED(image_data[i])
        row, col, channel = get_incremented_coords_3dim(row, col, channel, width, length)
        if i < len(image_data) - 1 and image_data[i] == image_data[i + 1]:
            amount = image_data[i + 2] + SECOND_BYTE_COEFFICIENT * image_data[i + 3]
            for k in range(amount - 1):
                encoded_image[row, col, channel] = INTEGER_DTYPE_SIGNED(image_data[i])
                row, col, channel = get_incremented_coords_3dim(row, col, channel, width, length)
            i += 3
        i += 1
    return encoded_image


def anti_zigzager(block):
    result = np.empty(shape=(BLOCK_LENGTH, BLOCK_LENGTH), dtype=INTEGER_DTYPE_SIGNED)
    phase = - 1
    i = 0
    j = 0
    col = 0
    row = 0
    while i < BLOCK_LENGTH != j < BLOCK_LENGTH:

        result[i, j] = block[row, col]
        row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)

        if is_on_edge(i) and not is_on_point(j, i):
            phase, j = new_phase_values(phase, j)
            result[i, j] = block[row, col]

            row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)
        elif is_on_edge(j) and not is_on_point(i, j):
            phase, i = new_phase_values(phase, i)
            result[i, j] = block[row, col]
            row, col = get_incremented_coords_2dim(row, col, BLOCK_LENGTH)

        i += phase
        j -= phase

    return result
