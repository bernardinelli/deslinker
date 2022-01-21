import trip_numba as tp
import astropy.table as tb 
import numpy as np
import orbital_motion as om 

import gc 
import pair_finder as pp
import os 
import vaex
import numba


try:
	chunksize = int(os.getenv('CHUNK'))
except:
	chunksize = 5000000


#@vaex.delayed
def compute_triplets(evolve, exposure, gamma, dgamma):
	
	evolve['T_3'] = (exposure.mjd_mid - evolve['MJD_OBS_1'])/365.25

	evolve = pp.triplet_estimator(evolve, exposure, gamma, dgamma)
    #print(theta_zero, evolve)

	evolve['COV_SCALE'] = ((evolve['T_3'] - 1)/evolve['DELTA_T'])**2

	evolve['COV_0_0'] = evolve['ERR_1_x_x'] + evolve['COV_SCALE'] * evolve['ERR_2_x_x']
	evolve['COV_1_1'] = evolve['ERR_1_y_y'] + evolve['COV_SCALE'] * evolve['ERR_2_y_y']
	evolve['COV_0_1'] = evolve['ERR_1_x_y'] + evolve['COV_SCALE'] * evolve['ERR_2_x_y']

	evolve['SEMI_MAJOR'] = 0.5*(evolve['COV_0_0'] + evolve['COV_1_1'] + np.sqrt((evolve['COV_0_0'] - evolve['COV_1_1'])**2 + 4*evolve['COV_0_1']**2))
	evolve['SEMI_MAJOR'] = 1.5*np.sqrt(evolve['SEMI_MAJOR'])

	evolve['THETA_X_1'] =  evolve['THETA_PLUS_X']  - evolve['THETA_X']
	evolve['THETA_Y_1'] =  evolve['THETA_PLUS_Y']  - evolve['THETA_Y']

	pref_theta_1 = evolve['SEMI_MAJOR']/np.sqrt(evolve['THETA_X_1']**2 + evolve['THETA_Y_1']**2)
	evolve['THETA_X_1'] *= np.sqrt((1 + pref_theta_1))
	evolve['THETA_Y_1'] *= np.sqrt((1 + pref_theta_1))
	del pref_theta_1

	evolve['THETA_X_2'] =  evolve['THETA_BIND_PLUS_X'] - evolve['THETA_X']
	evolve['THETA_Y_2'] =  evolve['THETA_BIND_PLUS_Y'] - evolve['THETA_Y']

	pref_theta_2 = evolve['SEMI_MAJOR']/np.sqrt(evolve['THETA_X_2']**2 + evolve['THETA_Y_2']**2)
	evolve['THETA_X_2'] *= np.sqrt((1 + pref_theta_2))
	evolve['THETA_Y_2'] *= np.sqrt((1 + pref_theta_2))
	del pref_theta_2

	evolve['DET'] = evolve['THETA_X_1'] * evolve['THETA_Y_2']  - evolve['THETA_X_2']  * evolve['THETA_Y_1'] 
	#evolve.remove_columns(['DIST_CENTER', 'T_3', 'COV', 'SEMI_MAJOR', 'THETA_PLUS_X', 'THETA_PLUS_Y', 'THETA_BIND_PLUS_X', 'THETA_BIND_PLUS_Y'])

	evolve['DIST'] = np.sqrt((evolve['THETA_X'] - exposure.theta[0,0])**2 + (evolve['THETA_Y'] - exposure.theta[0,1])**2)

	
	return evolve


def generate_triplet_output(trip, obj1, obj2, exposure, cat_ids):
	
	trips = np.where(trip)
	ntrips = len(trips[0])

	trnid = np.zeros(ntrips * 3, dtype='int')
	orbid = np.zeros_like(trnid)

	trnid[0::3] = obj1[trips[0]]
	trnid[1::3] = obj2[trips[0]]
	trnid[2::3] = exposure[trips[1]]
	orbid[0::3] = np.arange(ntrips)
	orbid[1::3] = np.arange(ntrips)
	orbid[2::3] = np.arange(ntrips)

	trnid = cat_ids[trnid]

	return orbid, trnid

def triplet_finder(catalog, pairs, gamma, dgamma, exposures, output):
	'''	
	For all pairs, finds the location of a future detection according to a fiducial model (fixed gamma, delta gamma = 0, gammadot = 0)
	and searches the corresponding exposure for another detection for this pair, with the region for this search defines by
	 gamma_search = [gamma - dgamma, gamma + dgamma] and gammadot^2 <= gammadot_bind^2
	'''

	keys = np.array(list(exposures.keys()))
	

	#pairs['DELTA_T'] =   (pairs['MJD_OBS_2'] - pairs['MJD_OBS_1'])/365.25
	search_pairs = pairs[pairs['DELTA_T'] > 0.08/365.25]
	del pairs 

	gc.collect()



	for i in keys:
		print(i)
		#cond = (search_pairs['MJD_OBS_2'] < exposures[i].mjd_mid) & (search_pairs['MJD_OBS_1'] > exposures[i].mjd_mid - 365)
		#cond = (cond) & (search_pairs['MJD_OBS_2'] > exposures[i].mjd_mid - 365)
		evolve = compute_triplets(search_pairs, exposures[i], gamma, dgamma)
		#evolve.execute()

		if len(evolve) == 0:
			#t = tb.Table(names=('ORBITID', 'TRANSIENTID'))
			#t.write('triplets/' + output.split('_')[0] + '/' + output + '_triplets.{}.fits'.format(i))
			np.save('triplets/' + output.split('_')[0] + '/' + output + '_triplets.{}.npy'.format(i), np.array([]))
			continue
		else:
			pass

		gc.collect()
		#print(evolve)

		print('Searching triplets: ' + str(len(evolve)) + ' pairs')

		origin  = np.array([evolve['THETA_X'].values,   evolve['THETA_Y'].values])
		theta_1 = np.array([evolve['THETA_X_1'].values, evolve['THETA_Y_1'].values]) 
		theta_2 = np.array([evolve['THETA_X_2'].values, evolve['THETA_Y_2'].values])
		#det = evolve['THETA_X_1'] * evolve['THETA_Y_2']  - evolve['THETA_X_2']  * evolve['THETA_Y_1'] 
		trip = np.zeros((len(evolve), len(exposures[i].catalog)), dtype = bool)
		trip = tp.triplet_inverse(theta_1, theta_2, origin, np.array(evolve['DET'].values), np.array(exposures[i].catalog['THETA_X']), np.array(exposures[i].catalog['THETA_Y']), trip)

		#print(evolve['OBJ_1', 'OBJ_2'])

		orbid, trnid = generate_triplet_output(trip, evolve['OBJ_1'].values, evolve['OBJ_2'].values, exposures[i].catalog['OBJID'], catalog['CATALOG_ID'])
		
		print('Found {} triplets for exposure {}'.format(len(orbid), i))

		if len(orbid) == 0:
			np.save('triplets/' +  output.split('_')[0] + '/' + output + '_triplets.{}.npy'.format(i), np.array([]))
			continue


		#triplets.write('triplets/' +  output.split('_')[0] + '/' + output + '_triplets.{}.fits'.format(i))
		# Faster write with numpy than with FITS!

		np.save('triplets/' +  output.split('_')[0] + '/' + output + '_triplets.{}.npy'.format(i), np.array([orbid, trnid]).T)

		del trip, orbid, trnid, exposures[i], origin, theta_1, theta_2, evolve

		gc.collect()



def triplet_finder_chunks(catalog, pairs, gamma, dgamma, exposures, output, chunksize = chunksize):
	'''	
	Same as `triplet_finder`, but splits the catalog into chunks, thus making the RAM requirement smaller
	'''

	keys = np.array(list(exposures.keys()))
	

	#pairs['DELTA_T'] =   (pairs['MJD_OBS_2'] - pairs['MJD_OBS_1'])/365.25
	#pairs.execute()

	search_pairs = pairs[pairs['DELTA_T'] > 0.08/365.25]
	#search_pairs.execute()
	del pairs

	#search_pairs.add_index('EXPNUM_1')

	for i in keys:
		print(i)
		cond = (search_pairs['MJD_OBS_2'] < exposures[i].mjd_mid) & (search_pairs['MJD_OBS_1'] > exposures[i].mjd_mid - 365)
		cond = (cond) & (search_pairs['MJD_OBS_2'] > exposures[i].mjd_mid - 365)
		
		chunk_pairs = search_pairs[cond]
		del cond 

		chunk_pairs = compute_triplets(chunk_pairs, exposures[i], gamma, dgamma)


		if len(chunk_pairs) == 0:
			np.save('triplets/' + output.split('_')[0] + '/' + output + '_triplets.{}.npy'.format(i), np.array([]))
			continue

		divisor = 1

		if len(exposures[i].catalog) > 4000:
			divisor = 2

		chunk_start = np.arange(0, len(chunk_pairs), chunksize//divisor)
		chunk_end   = np.arange(chunksize//divisor, len(chunk_pairs) + chunksize//divisor, chunksize//divisor)
		chunk_end[-1] = len(chunk_pairs)
		chunks = range(len(chunk_start))

		for j in chunks:

			print('Chunk {}'.format(j))
			if not os.path.exists('triplets/' + output.split('_')[0] + '/' + output + '_triplets.{}_{}.npy'.format(i, j)):
				
				#evolve = compute_triplets(chunk_pairs[chunk_start[j] : chunk_end[j]], exposures[i], gamma, dgamma)
				evolve = chunk_pairs[chunk_start[j] : chunk_end[j]]				
				gc.collect()

				print('Searching triplets: ' + str(len(evolve)) + ' pairs')

				origin  = np.array([evolve['THETA_X'].values,   evolve['THETA_Y'].values])
				theta_1 = np.array([evolve['THETA_X_1'].values, evolve['THETA_Y_1'].values]) 
				theta_2 = np.array([evolve['THETA_X_2'].values, evolve['THETA_Y_2'].values])
				#det = evolve['THETA_X_1'] * evolve['THETA_Y_2']  - evolve['THETA_X_2']  * evolve['THETA_Y_1'] 

				trip = np.zeros((len(evolve), len(exposures[i].catalog)), dtype = bool)
				trip = tp.triplet_inverse(theta_1, theta_2, origin, np.array(evolve['DET'].values), np.array(exposures[i].catalog['THETA_X']), np.array(exposures[i].catalog['THETA_Y']), trip)

				
				orbid, trnid = generate_triplet_output(trip, evolve['OBJ_1'].values, evolve['OBJ_2'].values, 
								np.array(exposures[i].catalog['OBJID']), np.array(catalog['CATALOG_ID']))

				print('Found {} triplets for exposure {}'.format(len(orbid), i))

				if len(orbid) == 0:
					np.save('triplets/' +  output.split('_')[0] + '/' + output + '_triplets.{}_{}.npy'.format(i, j), np.array([]))
					continue


				np.save('triplets/' +  output.split('_')[0] + '/' + output + '_triplets.{}_{}.npy'.format(i,j), np.array([orbid, trnid]).T)

				del trip, orbid, trnid, evolve, origin, theta_1, theta_2

		del exposures[i], chunk_pairs
		## cheat to avoid having to re-check exposure when we come back to subsplitting
		np.save('triplets/' + output.split('_')[0] + '/' + output + '_triplets.{}.npy'.format(i), np.array([]))

		gc.collect()
