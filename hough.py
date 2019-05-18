# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:52:58 2018

@author: aswinmanohar
"""

import numpy as np
import sys
from DATAExtract.training import *
import matplotlib as pyplot

def hough_transform(img):
  # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos



#Input image to binary image 
if len(sys.argv) >= 2:
   D=pfddata(sys.argv[1])

y=pfddata.time_vs_phase(D)

img=np.zeros_like(y)

img[np.arange(len(y)),y.argmax(1)]=1

accumulator, thetas, rhos = hough_transform(img)

#  finding the peak based on max votes


idx = np.argmax(accumulator)

rho = rhos[idx / accumulator.shape[1]]
theta = thetas[idx % accumulator.shape[1]]
print "rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta))
print idx
width, height = img.shape
print width,height

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Input binary image')
ax[0].axis('Image')

ax[1].imshow(accumulator, cmap='jet',extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('Image')
    
plt.show()
