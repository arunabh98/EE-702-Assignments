import numpy as np
import math

def create_depth_map_of_sphere(radius):
	image_radius = radius*0.3
	depth_map = np.zeros((radius, radius))
	for i in range(0, radius):
	    for j in range(0, radius):
	        depth = np.sqrt(max(image_radius**2 - math.pow(i - radius/2 , 2) -\
	        math.pow(j - radius/2 , 2), 0))
	        depth_map[i][j] = depth

	return depth_map
