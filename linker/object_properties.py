import astropy.table as tb
import numpy as np
from scipy import spatial
from  skyfield.api import load, Topos
from astropy.time import Time
from itertools import chain



class Exposure:
	'''
	Corresponds to an exposure, with the correspondent catalog, covariance/correlation matrix, Earth position and kD tree
	'''
	def __init__(self, table, coordinate_system):
		self.expnum = table['EXPNUM']
		self.ra_center = table['RA']
		self.dec_center = table['DEC']
		#self.mjd = table['MJD_OBS']
		self.mjd_mid = table['MJD_MID']
		self.mjd = self.mjd_mid
		self.band = table['BAND']
		self.cs = coordinate_system
		self.earth_position = self.cs.telescope_centric_r(self.mjd_mid)[0]
		self.theta = np.array([self.cs.theta_def(self.ra_center, self.dec_center)])
		self.covariance = np.array([[0,0],[0,0]])
		self.sigma = coordinate_system.propagate_error_matrix(self.covariance, self.ra_center, self.dec_center)
		#self.catalog = []

	def __str__(self):
		'''
		Sometimes we wanna see if the thing works, right?
		'''
		st = 'Exposure ' + str(self.expnum) + ' taken at MJD ' + str(self.mjd) + ', centered on (' + str(self.ra_center) + ', ' + str(self.dec_center) + ')'
		return st

	def find_catalog(self, catalog):
		'''
		Finds all catalog in the exposures in a given list of catalog
		'''
		self.catalog = catalog[catalog['EXPNUM'] == self.expnum]

	def make_kdtree(self, catalog):
		'''
		Builds the cKDTree object for the exposure
		'''
		#self.get_points()
		if len(catalog) > 0:
			return spatial.cKDTree(catalog)
		else: 
			return None

	def det_kdtree(self):
		'''
		Builds the cKDTree object for the exposure
		'''
		#self.get_points()
		if len(self.catalog) > 0:
			theta = np.asarray([self.catalog['THETA_X'], self.catalog['THETA_Y']])
			self.kdtree = spatial.cKDTree(theta.T, leafsize=64)
		else: 
			self.kdtree = None

	def query_exposure(self, point, radius):
		'''
		Queries the exposure for a detection on the ball :math:`\\mathcal{C}(\\mathrm{point}, \\mathrm{radius})`, returns the matches in this ball
		'''
		if self.kdtree is not None:
			res = self.kdtree.query_ball_point(point, radius)
			res = list(chain(*res))
			if len(res) > 0:
				return self.det_id[res]
			else:
				return []

		else:
			return np.array([])

	def query_tree_exposure(self, tree, radius):
		'''
		Queries the exposure for a detection on the ball against the other input tree, returns the matches in this ball
		'''
		if self.kdtree is not None:
			res = tree.query_ball_tree(self.kdtree, radius, eps = 0.1)
			res = list(chain(*res))
			if len(res) > 0:
				return self.det_id[res]
			else:
				return []
		else:
			return np.array([])
