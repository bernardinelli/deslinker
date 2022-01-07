
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

	try:
		run_grow = bool(int(os.getenv('RUN_GROW')))
	except:
		run_grow = False

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
				'-maxFPR=1', '-exposureFile=' + config['LINKER']['EXPOSURE_LIST']])

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



	## Loads the catalog
	print("Loading the catalog")
	catalog = tb.Table.read(config['LINKER']['YEAR_CATALOG'], 1)
	#catalog = catalog['RA', 'DEC', 'EXPNUM', 'ERRAWIN_WORLD', 'CATALOG_ID']


	linker(catalog, exposure_list, output, gamma, dgamma, ra_center, dec_center, mjd0)


	print(sys.argv[1] + ' triplets done')

	if run_grow:
		subprocess.call([orbitspp+'/GrowOrbits', '-transientFile=' + config['LINKER']['CATALOG'], 
						'-tripletFile=triplets/' + output.split('_')[0] + '/'  + output + '_triplets.fits', 
						'-orbitFile=orbits/' + output.split('_')[0] + '/'  + output + '_orbits.fits',
						'-maxFPR=1', '-minUnique=6', '-exposureFile=' + config['LINKER']['EXPOSURE_LIST']])

