# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:16:50 2016

@author: debasmit
"""

import random
import math
import BitVector
## UNCOMMENT THE TEXTURE TYPE YOU WANT:
texture_type = ’random’ #(A1)
#texture_type = ’vertical’ #(A2)
#texture_type = ’horizontal’ #(A3)
#texture_type = ’checkerboard’ #(A4)
#texture_type = None #(A5)
IMAGE_SIZE = 8 #(A6)
#IMAGE_SIZE = 4 #(A6)
GRAY_LEVELS = 6 #(A7)
R = 1 # the parameter R is radius of the circular pattern #(A8)
P = 8 # the number of points to sample on the circle #(A9)
image = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(B1)
if texture_type == ’random’: #(B2)
image = [[random.randint(0,GRAY_LEVELS-1)
for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(B3)
elif texture_type == ’diagonal’: #(B4)
image = [[GRAY_LEVELS - 1 if (i+j)%2 == 0 else 0
for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(B5)
elif texture_type == ’vertical’: #(B6)
image = [[GRAY_LEVELS - 1 if i%2 == 0 else 0
for i in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(B7)
elif texture_type == ’horizontal’: #(B8)
image = [[GRAY_LEVELS - 1 if j%2 == 0 else 0
for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(B9)
elif texture_type == ’checkerboard’: #(B10)
image = [[GRAY_LEVELS - 1 if (i+j+1)%2 == 0 else 0
for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(B11)
else: #(B12)
image = [[1, 5, 3, 1],[5, 3, 1, 4],[4, 0, 0, 0],[2, 3, 4, 5]] #(B13)
IMAGE_SIZE = 4 #(B14)
GRAY_LEVELS = 3 #(B15)
print "Texture type chosen: ", texture_type #(C1)
print "The image: " #(C2)
for row in range(IMAGE_SIZE): print image[row] #(C3)