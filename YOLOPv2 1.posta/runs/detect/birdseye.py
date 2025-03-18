import cv2
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import time
# To open matplotlib in interactive mode


# Load the image
img = cv2.imread('aa.jpg')

# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)
plt.show()


# All points are in format [cols, rows]
pt_A = [25, 540]
pt_B = [25, 830]
pt_C = [1820, 830]
pt_D = [1820, 540]

width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))


input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
cm= 65
#görüş düzgün yamuğunun oranları a,b ve h
a=10
b=50
h=18
output_pts = np.float32([[0, 0],
                        [(b-a)*cm/2, h*cm],
                        [(b+a)*cm/2, h*cm],
                        [b*cm, 0]])

# Compute the perspective transform M
# Compute the perspective transform M
t1 = time.time()
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# Apply the perspective transformation to the image
out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
t2 = time.time()
# Display the transformed image
print(f"Time taken: {t2-t1:.6f} seconds")

plt.imshow(out)
plt.show()