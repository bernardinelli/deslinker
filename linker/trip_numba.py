import numpy as np 

import numba 
from numba.pycc import CC

cc = CC('triplet_parallelogram')
# Uncomment the following line to print out the compilation steps
cc.verbose = True

@cc.export('triplet_parallelogram', 'b1[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:], b1[:,:])')

@numba.jit('b1[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:], b1[:,:])', nopython=True, parallel = True, cache = True)
def triplet_parallelogram(theta_1, theta_2, origin, scale, det_x, det_y, trips):
 	n = len(det_x)
 	for i in numba.prange(n):
 		shift_x = det_x[i] - origin[0]
 		shift_y = det_y[i] - origin[1]

 		mu = ( theta_2[1] * shift_x - theta_2[0] * shift_y)/scale
 		nu = (-theta_1[1] * shift_x + theta_1[0] * shift_y)/scale

 		inside = np.logical_and(np.abs(mu) < 1.1, np.abs(nu) < 1.1)

 		trips[i] = inside

 	return trips

@cc.export('triplet_inverse', 'b1[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:], b1[:,:])')

@numba.jit('b1[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:], b1[:,:])', nopython = True, parallel=True, cache = True)
def triplet_inverse(theta_1, theta_2, origin, scale, det_x, det_y, trips):
 	n = len(scale)
 	for i in numba.prange(n):
 		shift_x = det_x - origin[0][i]
 		shift_y = det_y - origin[1][i]

 		mu = ( theta_2[1][i] * shift_x - theta_2[0][i] * shift_y)/scale[i]
 		nu = (-theta_1[1][i] * shift_x + theta_1[0][i] * shift_y)/scale[i]

 		inside = np.logical_and(np.abs(mu) < 1.1, np.abs(nu) < 1.1)

 		trips[i] = inside

 	return trips
	
if __name__ == "__main__":
    cc.compile()