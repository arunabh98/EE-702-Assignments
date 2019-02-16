# Import packages
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
# First we construct the depthmap of the sphere.
sphere_radius = 40
source_vector = [0, 0, 1]
source_noise = 0
radiance_noise = 0
image_radius = 0.25*sphere_radius
num_iter = 1000
lambda_value = 100
depth_num_iter = 1000

# Compute the meshgrid - grid of x and y.
[cols,rows] = np.meshgrid(range(0, sphere_radius),range(0, sphere_radius))

depth_map = np.zeros((sphere_radius, sphere_radius))
roi = np.zeros((sphere_radius, sphere_radius)) # Region of interest of sphere

for i in range(0, sphere_radius):
    for j in range(0, sphere_radius):
        depth_map[i][j] = image_radius**2 - math.pow(cols[i][j] - sphere_radius/2 , 2) -\
        math.pow(rows[i][j] - sphere_radius/2 , 2)
        if depth_map[i][j] > 0:
            roi[i][j] = 1

depth_map = np.sqrt(depth_map * roi);

# Compute the gradient field.
p = np.zeros((sphere_radius, sphere_radius))
q = np.zeros((sphere_radius, sphere_radius))
for i in range(1, sphere_radius - 1):
    for j in range(1, sphere_radius - 1):
        p[i][j] = depth_map[i][j] - depth_map[i][j-1]
        q[i][j] = depth_map[i][j] - depth_map[i-1][j] 

# Only take the gradient in the region of interest.
p = p * roi
q = q * roi

# Add noise to source_vector.
source_vector = source_vector + np.random.normal(0,1,3)*source_noise
source_magnitude = (source_vector[0]**2+source_vector[1]**2 + 1)**0.5

# Calculate the image radiance = n.s
image_radiance = np.zeros((sphere_radius, sphere_radius))
for i in range(0, sphere_radius):
    for j in range(0, sphere_radius):
        # If within region of interest
        if roi[i][j]:
            gradient_magnitude = (p[i,j]**2 + q[i,j]**2 + 1)**0.5
            image_radiance[i][j] = max((p[i,j]*source_vector[0] + q[i,j]*source_vector[1] + 1)/(source_magnitude*gradient_magnitude), 0)

# Extract the boundary of an image.
boundary_map = np.zeros((sphere_radius, sphere_radius))
roi_radiance = image_radiance > 0 # region of interest where the randiance is greater than 0

# Algorithm presented to extract the borundary is inspired by 
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

kernel = np.ones((3,3), np.uint8)
# cv2.erode extracts the part without the boundary. So to extract the boundary we need
# to subtract the result given by cv.erode from roi_radiance.
boundary_map = roi_radiance - cv2.erode(roi_radiance.astype(np.uint8), kernel, 5)

intersection_roi = roi_radiance * roi

# Preserve the gradient of the final region of interest.
p = p * intersection_roi
q = q * intersection_roi

# Calculate the noise added to the radiance.
noise_vector = np.random.normal(0, 1, sphere_radius * sphere_radius)
noise_matrix = noise_vector.reshape(sphere_radius, sphere_radius)*radiance_noise
image_radiance = image_radiance + noise_matrix

# Calculate the boundary at the gradient.
gradX, gradY       = np.array(image_radiance,copy=True),np.array(image_radiance,copy=True)
gradX[:][1:-1]     = (  gradX[:][2:]   - gradX[:][:-2]) * 0.5 
gradY[1:-1][:]     = (  gradY[:-2][:]  - gradY[2:][:] ) * 0.5

boundary_p = np.array(gradX * boundary_map.astype(bool),copy=True)
boundary_q = np.array(gradY * boundary_map.astype(bool),copy=True)

next_p = np.array(boundary_p,copy=True)
next_q = np.array(boundary_p,copy=True)
p_estimated,q_estimated = np.array(boundary_p,copy=True),np.array(boundary_q,copy=True)   
# TODO: To be changed.

# Start the shape from shading iterative solution.
for k in range(0, num_iter):
    for i in range(1, sphere_radius - 1):
        for j in range(1, sphere_radius - 1):
            if roi_radiance[i][j]:
                # Current value of radiance estimated by our model.
                gradient_magnitude = (p_estimated[i,j]**2 + q_estimated[i,j]**2 + 1)**0.5
                current_estimated_radiance = (p_estimated[i,j]*source_vector[0] + q_estimated[i,j]*source_vector[1] + 1)/\
                (source_magnitude*gradient_magnitude)
                
                # Compute the gradient of R(p, q) with respect to p and q.
                denominator_gradient = source_magnitude*math.pow((1 + p_estimated[i][j]**2 + q_estimated[i][j]**2), 1.5)
                Rp = (p_estimated[i][j] + (p_estimated[i][j]**2)*source_vector[1] - p_estimated[i][j] - \
                    q_estimated[i][j]*p_estimated[i][j]*source_vector[1])/denominator_gradient
                Rq = (q_estimated[i][j] + (p_estimated[i][j]**2)*source_vector[0] - q_estimated[i][j] - \
                    q_estimated[i][j]*p_estimated[i][j]*source_vector[0])/denominator_gradient
                
                # Compute the next q and q estimates.
                next_p[i][j] = 0.25*(p_estimated[i+1][j] + p_estimated[i][j+1] + p_estimated[i-1][j] + p_estimated[i][j-1]) +\
                 lambda_value*(current_estimated_radiance - image_radiance[i][j])*Rp
                next_q[i][j] = 0.25*(q_estimated[i+1][j] + q_estimated[i][j+1] + q_estimated[i-1][j] + q_estimated[i][j-1]) +\
                 lambda_value*(current_estimated_radiance - image_radiance[i][j])*Rq

    # Update the gradient values, except the ones on the boundary.
    p_estimated = next_p*roi_radiance*(1-boundary_map) + boundary_p*roi_radiance*boundary_map
    q_estimated = next_q*roi_radiance*(1-boundary_map) + boundary_q*roi_radiance*boundary_map

# Now that the gradient has been estimated at each point. Calculate the depth.
# Calculate gradient of p with respect to x and of q with respect to y.
gradient_p_x = np.zeros((sphere_radius, sphere_radius))
gradient_q_y = np.zeros((sphere_radius, sphere_radius))
gradient_p_x[:][1:-1] = p_estimated[:][2:] - p_estimated[:][:-2]
gradient_q_y[:][1:-1] = q_estimated[:][2:] - q_estimated[:][:-2]
Z = np.zeros((sphere_radius, sphere_radius))
next_Z = np.zeros((sphere_radius, sphere_radius))

for k in range(0, depth_num_iter):
    for i in range(1, sphere_radius - 1):
        for j in range(1, sphere_radius - 1):
            if roi_radiance[i][j]:
                next_Z[i][j] = 0.25*(Z[i-1][j] + Z[i+1][j] + Z[i][j-1] + Z[i][j+1]) + abs(gradient_p_x[i][j]) + abs(gradient_q_y[i][j])

    Z = roi_radiance*next_Z

final_Z = Z*roi_radiance

plt.imshow(final_Z)


