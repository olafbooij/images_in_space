from PIL import Image
import numpy
from transformations import concatenate_matrices, euler_matrix, translation_matrix
import matplotlib.pyplot as plt
from urllib.request import urlopen

img_in = Image.open(urlopen("https://python-pillow.org/images/pillow-logo-light-text-1280x640.png"))

# Create a "pose" (translation and rotation) of the image-plane relative
# to the camera expressed as a 4x4 transformation matrix.
pose = concatenate_matrices(translation_matrix((-.5, 0, 1)),
                            euler_matrix(0., 0.8, 0.2, 'rxyz'))

# From this transformation matrix create a 3x3 "Homography matrix". This
# requires some "magic": adding the 4rd column (translation) to the 3rd
# column (z-axis rotation), and keep only upper-left 3x3 part.
pose[:,2] += pose[:,3]
homography = pose[0:3, 0:3]

# To compose a new image, we create camera calibration matrices K_in and K_out given the image centers (cx, cy) and focal lengths f.
imsize_out = (1000, 1000)
cx_out = imsize_out[0]//2
cy_out = imsize_out[1]//2
f_out = imsize_out[0]
K_out = numpy.array([[f_out, 0    , cx_out],
                     [0    , f_out, cy_out],
                     [0    , 0    , 1     ]]);

cx_in = img_in.size[0]//2
cy_in = img_in.size[1]//2
f_in = img_in.size[0]
K_in = numpy.array([[f_in, 0   , cx_in],
                    [0   , f_in, cy_in],
                    [0   , 0   , 1    ]]);

homography = numpy.dot(K_out, homography)
homography = numpy.dot(homography, numpy.linalg.inv(K_in))

# The call to transform, actually requires the transformation of the
# output image pixels to the input image pixels. This is the inverse of
# the homography we created.
invhomography = numpy.linalg.inv(homography)
# Mathematically scale of the homography does not matter. For the
# transform function it should be scaled such that the last element is 1.
invhomography /= invhomography[2,2]

# Image.transform will apply the perspective transformation of the input
# image.
img_out = img_in.transform(imsize_out, Image.Transform.PERSPECTIVE, invhomography.flatten())


# The 3x3 matrix "homography" transforms points from the input image space
# to the output image space. Below we transform a square (borders of the
# input image) to pixel locations of the output image.
points = [numpy.array([0,              0             ]),
          numpy.array([0,              img_in.size[1]]),
          numpy.array([img_in.size[0], img_in.size[1]]),
          numpy.array([img_in.size[0], 0             ]),
          numpy.array([0,              0             ])]

def project_point(homography, point):
  # To apply a homography we need to use homogeneous coordinates.
  point_homogenious = [point[0], point[1], 1.]
  transformed_point_homogenious = numpy.dot(homography, point_homogenious)
  transformed_point = transformed_point_homogenious[0:2] / transformed_point_homogenious[2]
  return transformed_point

transed = [project_point(homography, point) for point in points]

# And finaly use some matplotlib to show the results:
plt.imshow(img_out)
x_coordinate, y_coordinate = zip(*transed)
plt.plot(x_coordinate, y_coordinate)
plt.show()

