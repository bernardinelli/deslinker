import pair_finder as pp
import object_properties as op
import orbital_motion as om

import numpy as np
import astropy.table as tb 
import gc
import os
import sys
import triplet_finder as tf 
import glob 

import vaex

try:
	subsplit = bool(int(os.getenv('SUBSPLIT')))
except:
	subsplit = False



def define_constants(catalog):
	'''
	Defines constants for the catalog: RA_0, DEC_0 and t_0.
	'''
	RA_0 = (np.max(catalog['RA']) + np.min(catalog['RA']))/2
	DEC_0 = (np.max(catalog['DEC']) + np.min(catalog['DEC']))/2
	t0 = (np.max(catalog(['MJD_OBS'])) + np.min(catalog['MJD_OBS']))/2
	return om.CoordinateSystem(t0, RA_0, DEC_0)

def prepare_data(catalog, exposure_list, coord = None):
	'''
	Prepares the dataset (i.e. exposures and detections) for the pair matching section
	'''

	#catalog = 
	if coord == None:
		coord = define_constants(catalog)

	exposure_list = exposure_list[np.isin(exposure_list['EXPNUM'], catalog['EXPNUM'])]

	catalog = tb.join(catalog, exposure_list['EXPNUM', 'MJD_MID'])

	catalog['THETA_X'], catalog['THETA_Y'] = coord.theta_def(catalog['RA'], catalog['DEC'])
	catalog['OBJID'] = range(len(catalog))
	

	pos_err = []
	for i in catalog:
		det_err = np.array([[i['ERRAWIN_WORLD']**2, 0],[0, i['ERRAWIN_WORLD']**2]])*(np.pi/180)**2
		pos_err.append(coord.propagate_error_matrix(det_err, i['RA'], i['DEC']))

	catalog['SIGMA_POS'] = pos_err
	exposures = pp.construct_exposures(exposure_list, catalog, coord)

	del pos_err


	return exposures, catalog, coord

def pair_finder(exposures, gamma, dgamma, coord, output):
	'''
	For a given exposure set and gamma, finds all pairs, and computes the parameter set we have
	'''
	
	exp, det = pp.build_trees(gamma, dgamma, coord, exposures)
	if not os.path.exists('triplets/' + output.split('_')[0] + '/' + output + '_pairs.hdf5'):
		if not os.path.exists('triplets/' + output.split('_')[0] + '/' + output + '_pairs.npy'):
			print('Finding pairs')
			
			pairids = pp.find_pairs(exposures, gamma, dgamma)
			np.save('triplets/' + output.split('_')[0] + '/' + output + '_pairs.npy', pairids)
		else:
			print("Loading previously saved pairs")
			pairids = np.load('triplets/' + output.split('_')[0] + '/' + output + '_pairs.npy')

		if len(pairids) == 0:
			return det, None
		print("Computing parameters")
		pairs = pp.compute_velocities(det, pairids, coord, gamma, dgamma, exp) 
		del pairids
		pairs.export_hdf5('triplets/' + output.split('_')[0] + '/' + output + '_pairs.hdf5')
		del pairs
	else:
		print("Loading pairs HDF5 file")

	## make sure vaex dataframe is out-of-memory!
	pairs = vaex.open('triplets/' + output.split('_')[0] + '/' + output + '_pairs.hdf5')
	return det, pairs


def triplet_finder(catalog, pairs, gamma, dgamma, exposures, output):
	'''
	Finds triplets for the input pairs
	'''
	if not subsplit:
		tf.triplet_finder(catalog, pairs, gamma, dgamma, exposures, output)
	else:
		tf.triplet_finder_chunks(catalog, pairs, gamma, dgamma, exposures, output)

def linker(catalog, exposure_list, output, gamma, dgamma, ra_0 = None, dec_0 = None, mjd_0 = None):
	'''
	Finds triplets for a given choice of catalog and year.
	'''
	triplets = {}
	print('Preparing data')
	#catalog = catalog[(catalog['MJD_OBS'] < years_end[year]) & (catalog['MJD_OBS'] > years_start[year])]
	if ra_0 != None:
		coord = om.CoordinateSystem(mjd_0, ra_0, dec_0)
	else:
		coord = None

	exposures, catalog, coord = prepare_data(catalog, exposure_list, coord)

	print(1/gamma, gamma) 
	gc.collect()

	
	catalog, pairs = pair_finder(exposures, gamma, dgamma, coord, output)

	if pairs == None:
		print('No pairs found!')
		a = tb.Table(names=('ORBITID', 'TRANSIENTID'))
		a.meta['GAMMA0'] = gamma
		a.meta['DGAMMA'] = dgamma*gamma

		a.meta['RA0'] = coord.ra_0 
		a.meta['DEC0'] = coord.dec_0 
		a.meta['MJD0'] = coord.t0
		a.meta['TDB0'] = coord.tdb0

		a.write('triplets/'+  output.split('_')[0] + '/'  + output + '_triplets.fits', overwrite=True)
		sys.exit()
	
		
	gc.collect()
			
	files = glob.glob('triplets/' + output.split('_')[0] + '/'  + output + '_triplets.*.npy')

	if len(files) == 0:
		print('First processing of this ini file')
	else:
		already_done = [i.split('.')[1] for i in files]
		for key in already_done:
			try:
				del exposures[int(key)]
			except:
				pass
	gc.collect()
	


	if bool(exposures):
		print('Finding triplets')
		catalog = catalog['CATALOG_ID', 'OBJID']
		triplet_finder(catalog, pairs, gamma, dgamma, exposures, output)
	print("Cleaning up")

	del catalog, pairs, exposures

	files = glob.glob('triplets/' + output.split('_')[0] + '/' + output + '_triplets.*.npy')
	triplets = []
	orb_offset = 0
	for f in files:
		try:
			trip = np.load(f)
			if len(trip) > 0:
				a = tb.Table()
				a['ORBITID'] = trip[:,0]
				a['ORBITID'] += orb_offset
				a['TRANSIENTID'] = trip[:,1]
				#a.meta.clear()
				orb_offset = np.max(a['ORBITID']) + 1
				triplets.append(a)
		except:
			print(f)
			
	if len(triplets) == 0:
		a = tb.Table(names=('ORBITID', 'TRANSIENTID'))
		a.meta['GAMMA0'] = gamma
		a.meta['DGAMMA'] = dgamma*gamma

		a.meta['RA0'] = coord.ra_0 
		a.meta['DEC0'] = coord.dec_0 
		a.meta['MJD0'] = coord.t0
		a.meta['TDB0'] = coord.tdb0

		a.write('triplets/'+  output.split('_')[0] + '/'  + output + '_triplets.fits', overwrite=True)
		sys.exit()

	try:
		triplets = tb.vstack(triplets)

	except TypeError:
		triplets = tb.Table(names=['ORBITID', 'TRANSIENTID'])

	del files, trip

	triplets['ORBITID'] = triplets['ORBITID'].astype('i8')
	triplets['TRANSIENTID'] = triplets['TRANSIENTID'].astype('i8')

	triplets.meta['GAMMA0'] = gamma
	triplets.meta['DGAMMA'] = dgamma*gamma

	triplets.meta['RA0'] = coord.ra_0 
	triplets.meta['DEC0'] = coord.dec_0 
	triplets.meta['MJD0'] = coord.t0
	triplets.meta['TDB0'] = coord.tdb0

	triplets.write('triplets/' + output.split('_')[0] + '/'  + output + '_triplets.fits', overwrite=True)



