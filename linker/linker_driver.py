
'''
The idea of this file is to control the linker and include the C++ steps as system commands
'''
import numpy as np

from linker import linker
import configparser
import sys
import numpy as np
from astropy.time import Time
import subprocess
import os
import numpy as np 
import astropy.table as tb


if __name__ == '__main__':	
	
	config = configparser.ConfigParser()
	config.sections()
	config.read(sys.argv[1])
	
	orbitspp = os.getenv('ORBITSPP')

	output = config['LINKER']['OUTPUT']

	if os.path.exists('triplets/' + output.split('_')[0] + '/'  + output + '_triplets.fits'):
		print(output + ' triplets already done')
		if os.path.exists('orbits/' + output.split('_')[0] + '/' + output + '_orbits.fits'):
			print(output + ' orbits already done')
			sys.exit()
		else:
			subprocess.call([orbitspp+'/GrowOrbits', '-transientFile=' + config['LINKER']['CATALOG'], 
				'-tripletFile=triplets/' + output.split('_')[0] + '/' + output + '_triplets.fits', 
				'-orbitFile=orbits/' + output.split('_')[0] + '/'  + output + '_orbits.fits',
				'-maxFPR=1', '-exposureFile=y6a1.exposures.positions.fits'])

			sys.exit()


	ra_center = float(config['LINKER']['RA0'])
	dec_center = float(config['LINKER']['DEC0'])
	try:
		tdb0 = float(config['LINKER']['TDB0'])
		mjd0 = Time(tdb0 + 2000, format='byear', scale='tdb').utc.mjd
	except:
		mjd0 = float(config['LINKER']['MJD0'])
		tdb0 = Time(mjd0, format='mjd').tdb.byear - 2000

	gamma = float(config['LINKER']['GAMMA0'])
	dgamma = float(config['LINKER']['DGAMMA0'])

	exposure_list = tb.Table.read(config['LINKER']['EXPOSURE_LIST'])

	input_exp = 'EXPOSURES/' + output.split('_')[0] + '/' + config['LINKER']['INPUT_EXPOSURES']

	## Call orbitspp/GetExposurePool, save it into the input_exposures file
	#orbitspp = os.getenv('ORBITSPP')
	if not os.path.exists(input_exp):
		with open(input_exp, 'w') as f:
			print(' '.join([orbitspp+'/GetExposurePool', str(ra_center), str(dec_center), 
				str(3.5), str(tdb0), str(gamma), str(gamma*dgamma), '-exposureFile=' + config['LINKER']['EXPOSURE_LIST']]))
			subprocess.call([orbitspp+'/GetExposurePool', str(ra_center), str(dec_center), str(3.5), 
				str(tdb0), str(gamma), str(gamma*dgamma),  '-exposureFile=' + config['LINKER']['EXPOSURE_LIST']], stdout=f)

	print("Retrieving exposures")
	exposures, tdb, astrometry = np.loadtxt(input_exp, unpack = True, skiprows = 3)
	exposures = exposures[np.where(astrometry == 1)]
	tdb = tdb[np.where(astrometry == 1)]
	exposures = exposures[np.where(np.abs(tdb) < 0.6)] #this means that exposures have to be < 8 months from the reference point. If we use the middle of the season, this gets us the year

	#sometimes (survey edges) we have no exposures, need a safety net for that:
	if len(exposures) == 0:
		print('No exposures to check!')
		t = tb.Table(names=('ORBITID', 'TRANSIENTID'))
		t.write('triplets/' + output.split('_')[0] + '/' + output + '_triplets.fits')
		sys.exit()
	## Loads the catalog
	print("Loading the catalog")
	catalog = tb.Table.read(config['LINKER']['YEAR_CATALOG'], 1)
	#catalog = catalog['RA', 'DEC', 'EXPNUM', 'ERRAWIN_WORLD', 'CATALOG_ID']
	exposure_list = exposure_list[np.isin(exposure_list['EXPNUM'], exposures)]
	print("Reducing the catalog")
	catalog = catalog[np.isin(catalog['EXPNUM'], exposures)]
	if len(catalog) == 0:
		print('No exposures to check!')
		t = tb.Table(names=('ORBITID', 'TRANSIENTID'))
		t.write('triplets/' + output.split('_')[0] + '/' + output + '_triplets.fits')
		sys.exit()

	linker(catalog, exposure_list, output, gamma, dgamma, ra_center, dec_center, mjd0)


	print(sys.argv[1] + ' triplets done')

	subprocess.call([orbitspp+'/GrowOrbits', '-transientFile=' + config['LINKER']['CATALOG'], 
					'-tripletFile=triplets/' + output.split('_')[0] + '/'  + output + '_triplets.fits', 
					'-orbitFile=orbits/' + output.split('_')[0] + '/'  + output + '_orbits.fits',
					'-maxFPR=1', '-exposureFile=y6a1.exposures.positions.fits'])

