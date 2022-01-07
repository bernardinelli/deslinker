from __future__ import print_function

import object_properties as op
import orbital_motion as om

import astropy.table as tb
import numpy as np
#from pixmappy import DESMaps
import pixmappy 
from scipy.spatial import cKDTree
from scipy.optimize import minimize          
from functools import reduce
from itertools import chain
import gc
import os 


import vaex
import numba 
from numba.typed import List 

YEAR = 365.25


def construct_exposures(exposure_list, catalog, coordinate_system):
	'''
	Builds all kD trees for exposures and detections, with no definition of gamma
	'''
	pmc = pixmappy.DESMaps()

	exposures = {}
	for i in exposure_list:
		exposures[i['EXPNUM']] = op.Exposure(i, coordinate_system, pmc)
		exposures[i['EXPNUM']].find_catalog(catalog)
		exposures[i['EXPNUM']].det_id = np.int32(exposures[i['EXPNUM']].catalog['OBJID'])
		exposures[i['EXPNUM']].catalog['SIGMA_POS'] = exposures[i['EXPNUM']].catalog['SIGMA_POS'] + exposures[i['EXPNUM']].sigma
		#exposures[i['EXPNUM']].det_kdtree()

	del pmc 
	return exposures


def build_trees(gamma, dgamma, coordinate_system, exposures):
	'''
	Builds, for all exposures and a given gamma and coordinate system, all trees with the parallax-removed catalog for a certain coordinate system. 
	'''

	#dgamma = dgamma_finder(gamma, 90)
	deparallaxed_catalog = []

	for i in exposures:
		det = exposures[i].catalog
		exposures[i].det_kdtree()

		theta = np.array([det['THETA_X'], det['THETA_Y']]).T
		det_parallax_0 = om.deparallax_catalog(theta, gamma, exposures[i].earth_position)
		#det.sort('OBJID')
		deparallaxed_catalog.append(det)
		det_parallax_p = om.deparallax_catalog(theta, gamma + dgamma*gamma, exposures[i].earth_position)
		det_parallax_m = om.deparallax_catalog(theta, gamma - dgamma*gamma, exposures[i].earth_position)
		det['THETA_X_Z'] = det_parallax_0.T[0]
		det['THETA_Y_Z'] = det_parallax_0.T[1]
		det['THETA_X_P'] = det_parallax_p.T[0]
		det['THETA_Y_P'] = det_parallax_p.T[1]
		det['THETA_X_M'] = det_parallax_m.T[0]
		det['THETA_Y_M'] = det_parallax_m.T[1]


		exposures[i].tree_0 = exposures[i].make_kdtree(det_parallax_0)
		exposures[i].tree_p = exposures[i].make_kdtree(det_parallax_p)
		exposures[i].tree_m = exposures[i].make_kdtree(det_parallax_m)


		exposures[i].theta_p = om.deparallax_catalog(exposures[i].theta, gamma, exposures[i].earth_position)


	deparallaxed_catalog = tb.vstack(deparallaxed_catalog)
	deparallaxed_catalog.sort('OBJID')

	return exposures, deparallaxed_catalog

@numba.jit
def id_pair_checker(pairs, exp_1, exp_2):
	'''
	Finds the ids of the pairs i,j of objects i in exp_1 and j in exp_2
	'''
	pp = zip(range(len(pairs)), pairs)
	id_list = []

	for i, j in pp:
		for k in j:
			id_list.append([exp_1[i], exp_2[k]])

	return id_list

#@numba.njit
def pair_id_checker(pairs, pair_bools):
	#pair = np.zeros((size_1, size_2), dtype=bool)

	for i in range(len(pairs)):
		for j in pairs[i]:
			pair_bools[i,int(j)] = True 
	return pair_bools

def find_pairs(exposures, gamma, dgamma):
	'''
	Pair finding algorithm with a dual tree search implementation. The idea is that, after building the three parallax-subtracted 
	trees for the Delta gamma range in `build_trees`, finds all pairs by searching future (< 90 days) trees for all exposures 
	in the input dataset.
	'''
	keys = np.sort(np.array(list(exposures.keys())))
	pairs = []

	for i in np.arange(0,len(keys)):
		for j in np.arange(i+1, len(keys)):
			if exposures[keys[j]].mjd - exposures[keys[i]].mjd > 365:
				break
			elif exposures[keys[j]].mjd - exposures[keys[i]].mjd < 1:
				continue
			else:
				d_0 = 1.1*om.max_proper(gamma, exposures[keys[i]].mjd, exposures[keys[j]].mjd)
				d_p = 1.1*om.max_proper(gamma + dgamma*gamma, exposures[keys[i]].mjd, exposures[keys[j]].mjd)
				d_m = 1.1*om.max_proper(gamma - dgamma*gamma, exposures[keys[i]].mjd, exposures[keys[j]].mjd)
				#if om.compute_distance(exposures[keys[i]].theta_p, exposures[keys[j]].theta_p) <= 2*om.r_decam + d_p:
				pairs_0 = exposures[keys[i]].tree_0.query_ball_tree(exposures[keys[j]].tree_0, d_0,)
				pairs_p = exposures[keys[i]].tree_p.query_ball_tree(exposures[keys[j]].tree_p, d_p,)
				pairs_m = exposures[keys[i]].tree_m.query_ball_tree(exposures[keys[j]].tree_m, d_m,)
				pair_bools = np.zeros((len(exposures[keys[i]].det_id), len(exposures[keys[j]].det_id)), dtype = bool)
				pair_bools = pair_id_checker(pairs_0, pair_bools)
				pair_bools = pair_id_checker(pairs_m, pair_bools)
				pair_bools = pair_id_checker(pairs_p, pair_bools)

				#pair_bools = pair_id_checker(pairs_expo, pair_bools)

				pairids = np.where(pair_bools)
				pairs_ids = np.zeros((2,len(pairids[0])))
				pairs_ids[0] = exposures[keys[i]].det_id[pairids[0]]
				pairs_ids[1] = exposures[keys[j]].det_id[pairids[1]]
				
				'''pairs_expo = List()

				for k in range(len(pairs_0)):
					pairs_expo.append(reduce(np.union1d, (pairs_0[k], pairs_p[k], pairs_m[k])).astype('int32'))

				pairs_ids = id_pair_checker(pairs_expo, exposures[keys[i]].det_id, exposures[keys[j]].det_id)'''


				#pairs_ids = np.unique(np.array(pairs_ids), axis=1)
				if len(pairs_ids) > 0:
					pairs.append(pairs_ids)
				#del pairs_expo, pairs_0, pairs_p, pairs_m, pairs_ids
				del pairids, pairs_ids, pair_bools, pairs_0, pairs_p, pairs_m


		del exposures[keys[i]].tree_0, exposures[keys[i]].tree_p, exposures[keys[i]].tree_m
	#print(pairs)
	if len(pairs) == 0:
		return np.array([])
	pp = np.hstack(pairs)
	return np.array(pp).astype('int32')
					


#@vaex.delayed
def compute_velocities(catalog, pairs, coordinate_system, gamma, dgamma, exposures):
	'''
	Computes the relevant pair information (alphadot, betadot, variance on delta gamma, Delta t, gamma dot bind, ...)
	'''
	#obj1 = tb.Table(catalog[pairs[0]])
	#obj2 = tb.Table(catalog[pairs[1]])


	params = vaex.from_arrays(OBJ_1 = pairs[0], OBJ_2 = pairs[1])
	#del pairs
	
	params['OBJ_1'] = params['OBJ_1'].astype('i4')
	params['OBJ_2'] = params['OBJ_2'].astype('i4')

	params['ALPHA'] = catalog['THETA_X_Z'][pairs[0]]
	params['BETA'] = catalog['THETA_Y_Z'][pairs[0]]

	params['ALPHA_P'] = catalog['THETA_X_P'][pairs[0]]
	params['BETA_P'] = catalog['THETA_Y_P'][pairs[0]]

	#this should be MJD_MID!!!!
	params['MJD_OBS_1'] = catalog['MJD_MID'][pairs[0]]
	params['MJD_OBS_2'] = catalog['MJD_MID'][pairs[1]]

	params['DELTA_T'] = (params['MJD_OBS_2'] - params['MJD_OBS_1'])/YEAR
	#params['DELTA_T'] = params['DELTA_T'].astype('f4')

	params['ALPHA_DOT'] = (catalog['THETA_X_Z'][pairs[1]] - catalog['THETA_X_Z'][pairs[0]])
	params['ALPHA_DOT'] = params['ALPHA_DOT']/params['DELTA_T']

	params['BETA_DOT'] = (catalog['THETA_Y_Z'][pairs[1]] - catalog['THETA_Y_Z'][pairs[0]])
	params['BETA_DOT'] = params['BETA_DOT']/params['DELTA_T']

	params['ALPHA_DOT_P'] = (catalog['THETA_X_P'][pairs[1]] - catalog['THETA_X_P'][pairs[0]])
	params['ALPHA_DOT_P'] = params['ALPHA_DOT_P']/params['DELTA_T']

	params['BETA_DOT_P'] = (catalog['THETA_Y_P'][pairs[1]] - catalog['THETA_Y_P'][pairs[0]])
	params['BETA_DOT_P'] = params['BETA_DOT_P']/params['DELTA_T']
	
	#obj1.remove_columns(['THETA_X_Z', 'THETA_Y_Z', 'THETA_X_P', 'THETA_Y_P', 'THETA_X_M', 'THETA_Y_M', 'MJD_MID'])
	#obj2.remove_columns(['THETA_X_Z', 'THETA_Y_Z', 'THETA_X_P', 'THETA_Y_P', 'THETA_X_M', 'THETA_Y_M', 'MJD_MID'])

	cosbeta = -(params['ALPHA'] * coordinate_system.r0[0] + params['BETA'] * coordinate_system.r0[1] + coordinate_system.r0[2])
	cosbeta /= np.linalg.norm(coordinate_system.r0)*np.sqrt(params['ALPHA']**2 + params['BETA']**2 + 1)

	gdot = om.gamma_dot_bind(gamma, params['ALPHA_DOT'], params['BETA_DOT'], cosbeta).values
	del cosbeta
	#gdot = np.array(params['GAMMA_DOT_BIND'])
	#params['GAMMA_DOT_BIND'].fill_value = 2.*np.pi*np.sqrt(2)*gamma**(3./2)/np.sqrt(3)
	#params['GAMMA_DOT_BIND'] = params['GAMMA_DOT_BIND'].filled()
	gdot[np.logical_not(np.isfinite(gdot))] = 2.*np.pi*np.sqrt(2)*gamma**(3./2)/np.sqrt(3)
	params['GAMMA_DOT_BIND'] = gdot
	del gdot

	#params.remove_column('COS_BETA')
	#del gdot
	#hack for when gdot is undefined

	
	params['ERR_1_x_x'] = catalog['SIGMA_POS'][pairs[0],0,0]
	params['ERR_1_y_y'] = catalog['SIGMA_POS'][pairs[0],1,1]
	params['ERR_1_x_y'] = catalog['SIGMA_POS'][pairs[0],0,1]

	params['ERR_2_x_x'] = catalog['SIGMA_POS'][pairs[1],0,0]
	params['ERR_2_y_y'] = catalog['SIGMA_POS'][pairs[1],1,1]
	params['ERR_2_x_y'] = catalog['SIGMA_POS'][pairs[1],0,1]

	#del obj1, obj2 
	del pairs 

	return params


#@vaex.delayed
def triplet_estimator(params, exposure, gamma, dgamma):
	'''
	Finds the possible locations for a given pair in the future according to varying gamma dot and Delta gamma
	'''
	z_plus = 1 + params['GAMMA_DOT_BIND'] * params['T_3'] - gamma * exposure.earth_position[2]

	z_p = 1 - (1 + dgamma)*gamma * exposure.earth_position[2]

	z_zero = 1 - gamma * exposure.earth_position[2]

	x = params['ALPHA'] + params['ALPHA_DOT'] * params['T_3'] - gamma * exposure.earth_position[0]
	y = params['BETA'] + params['BETA_DOT'] * params['T_3'] - gamma * exposure.earth_position[1]

	x_plus = params['ALPHA_P'] + params['ALPHA_DOT_P'] * params['T_3'] - (1 + dgamma) * gamma * exposure.earth_position[0]
	y_plus = params['BETA_P'] + params['BETA_DOT_P'] * params['T_3'] - (1 + dgamma) * gamma * exposure.earth_position[1]


	params['THETA_X'] = x/z_zero
	params['THETA_Y'] = y/z_zero

	params['THETA_PLUS_X'] = x_plus/z_p
	params['THETA_PLUS_Y'] = y_plus/z_p

	params['THETA_BIND_PLUS_X'] = x/z_plus
	params['THETA_BIND_PLUS_Y'] = y/z_plus

	return params


def sample_space(pair):
	'''
	Defines the search points and search radius for a given pair
	'''
	
	if pair['BOOL']:
		n = pair['N_GAMMA']
		theta_x = np.linspace(pair['THETA_MINUS'][0], pair['THETA_PLUS'][0], int(n), dtype=None)
		theta_y = np.linspace(pair['THETA_MINUS'][1], pair['THETA_PLUS'][1], int(n), dtype=None)
		r = pair['R_BIND']
	else:
		n = pair['N_BIND']
		theta_x = np.linspace(pair['THETA_BIND_MINUS'][0], pair['THETA_BIND_PLUS'][0], int(n), dtype=None)
		theta_y = np.linspace(pair['THETA_BIND_MINUS'][1], pair['THETA_BIND_PLUS'][1], int(n), dtype=None)
		r = pair['R_GAMMA']
	return theta_x, theta_y, n, r


def triplet_ids(triplets, pairs, catalog):
	'''
	Similar to pp.id_pair_checker, transforms the list of (pair_ids, matches) into a list of object IDs for the triplets
	'''
	trip = []
	for i, j in triplets:
		for k in j:
			trip.append([pairs[i]['OBJ_1'], pairs[i]['OBJ_2'], catalog[k]['OBJID']])

	trip = np.array(trip)

	return trip


def compute_search(pairs_for_tree):
	pairs_for_tree['DIST_GAMMA'] = np.sqrt((pairs_for_tree['THETA_PLUS_X'] - pairs_for_tree['THETA_MINUS_X'])**2 + (pairs_for_tree['THETA_PLUS_Y'] - pairs_for_tree['THETA_MINUS_Y'])**2)
	pairs_for_tree['DIST_BIND'] = np.sqrt((pairs_for_tree['THETA_BIND_PLUS_X'] - pairs_for_tree['THETA_BIND_MINUS_X'])**2 + (pairs_for_tree['THETA_BIND_PLUS_Y'] - pairs_for_tree['THETA_BIND_MINUS_Y'])**2)

	pairs_for_tree['COV'] = om.covariance_analytic(pairs_for_tree['T_3'][:,None, None], pairs_for_tree['DELTA_T'][:,None, None], pairs_for_tree['ERR_1'], pairs_for_tree['ERR_2'])

	pairs_for_tree['SEMI_MAJOR'] = 0.5*(pairs_for_tree['COV'][:,0,0] + pairs_for_tree['COV'][:,1,1] + np.sqrt((pairs_for_tree['COV'][:,0,0] - pairs_for_tree['COV'][:,1,1])**2 + 4*pairs_for_tree['COV'][:,0,1]**2))
	pairs_for_tree['SEMI_MAJOR'] = 1.5*np.sqrt(pairs_for_tree['SEMI_MAJOR'])

	del pairs_for_tree['COV']

	pairs_for_tree['N_BIND'] = 2*np.ceil(pairs_for_tree['DIST_BIND']/pairs_for_tree['DIST_GAMMA']) + 1
	pairs_for_tree['N_GAMMA'] = 2*np.ceil(pairs_for_tree['DIST_GAMMA']/pairs_for_tree['DIST_BIND']) + 1

	pairs_for_tree['R_GAMMA'] = 0.525*pairs_for_tree['DIST_GAMMA']
	pairs_for_tree['R_BIND'] = 0.525*pairs_for_tree['DIST_BIND']

	pairs_for_tree['BOOL'] = pairs_for_tree['DIST_GAMMA'] > pairs_for_tree['DIST_BIND']
	pairs_for_tree['BOOL'] = pairs_for_tree['BOOL'].astype('i1')

	pairs_for_tree['SEARCH_RADIUS'] = (pairs_for_tree['BOOL'] * pairs_for_tree['R_BIND'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['R_GAMMA']) + pairs_for_tree['SEMI_MAJOR']

	pairs_for_tree['N_SEARCH'] = pairs_for_tree['BOOL'] * pairs_for_tree['N_GAMMA'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['N_BIND']
	pairs_for_tree['N_SEARCH'] = pairs_for_tree['N_SEARCH'].astype('i4')

	del pairs_for_tree['DIST_BIND', 'DIST_GAMMA', 'R_GAMMA', 'R_BIND', 'N_GAMMA', 'N_BIND']

	pairs_for_tree['THETA_SEARCH_MINUS_X'] = pairs_for_tree['BOOL'] * pairs_for_tree['THETA_MINUS_X'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['THETA_BIND_MINUS_X']
	pairs_for_tree['THETA_SEARCH_MINUS_Y'] = pairs_for_tree['BOOL'] * pairs_for_tree['THETA_MINUS_Y'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['THETA_BIND_MINUS_Y']

	pairs_for_tree['THETA_SEARCH_PLUS_X'] = pairs_for_tree['BOOL'] * pairs_for_tree['THETA_PLUS_X'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['THETA_BIND_PLUS_X']
	pairs_for_tree['THETA_SEARCH_PLUS_Y'] = pairs_for_tree['BOOL'] * pairs_for_tree['THETA_PLUS_Y'] + (1 - pairs_for_tree['BOOL']) * pairs_for_tree['THETA_BIND_PLUS_Y']

	return pairs_for_tree['PAIR_ID', 'THETA_SEARCH_PLUS_X', 'THETA_SEARCH_MINUS_X', 'THETA_SEARCH_PLUS_Y', 'THETA_SEARCH_MINUS_Y', 'N_SEARCH', 'SEARCH_RADIUS']




