import cv2
import numpy as np
import math

# import zigzag functions;
from zigzag import *

# function to create run-length code of the input image;
def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:
            stream.append((image[i], skip))
            bitstream = bitstream + str(image[i]) + " " + str(skip) + " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1
    return bitstream


# defining standard block size for Quanitization process;
block_size = 8

# Quantization Matrix
# This quantization matrix is standardised by research and used in most major DCT-Based compression algorithmd;
QUANTIZATION_MAT = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [
                            18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

# reading image in grayscale and getting size;
img = cv2.imread('penguins.jpg', cv2.IMREAD_GRAYSCALE)
h = float()
w = float()
[h, w] = img.shape

# No of blocks needed;
height = h
width = w
h = np.float32(h)
w = np.float32(w)

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)
nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)

# The image may need to be padded with data to handle cases where size is not divisible by block size
# size of padded image = block size * number of blocks in height or width

# height of padded image
H = block_size * nbh
# width of padded image
W = block_size * nbw

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H, W))

# copy the values of img into padded_img[0:h,0:w]
for i in range(height):
        for j in range(width):
                pixel = img[i,j]
                padded_img[i,j] = pixel

cv2.imwrite('uncompressed_padded.bmp', np.uint8(padded_img))

# Encoding process:
# TODO: Check block creation again
# TODO: Look into zig zag reading of stream

# 1. Divide image into 8x8 blocks
# 2. Apply 2D DCT block by block
# 3. Read and reorder DCT coefficients in zig-zag order
# 4. Reshape into 8x8 blocks

for i in range(nbh):
        # start and end index for row
    row_ind_1 = i*block_size
    row_ind_2 = row_ind_1+block_size

    for j in range(nbw):
        # Start & end index for column
        col_ind_1 = j*block_size
        col_ind_2 = col_ind_1+block_size

        block = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]
        
        # traversing l2r and t2b across blcxks to quantize; 
        DCT = cv2.dct(block)
        DCT_normalized = np.divide(DCT, QUANTIZATION_MAT).astype(int)
        # reorder DCT coefficients in zig zag order by calling zigzag function
        reordered = zigzag(DCT_normalized)
        reshaped = np.reshape(reordered, (block_size, block_size))

        # copying reshaped matrix into padded_img on current blocks index
        padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = reshaped

cv2.imshow('Encoded Image', np.uint8(padded_img))

arranged = padded_img.flatten()
# Write encoded RLE to file;
bitstream = get_run_length_encoding(arranged)

# semicolon is delimiter;
bitstream = str(padded_img.shape[0]) + " " + \
    str(padded_img.shape[1]) + " " + bitstream + ";"

# Written to image_compressed.txt
file1 = open("image_compressed.txt", "w")
file1.write(bitstream)
file1.close()

cv2.waitKey(0)
cv2.destroyAllWindows()
