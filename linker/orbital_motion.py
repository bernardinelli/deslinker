import astropy.table as tb
import numpy as np
from scipy import spatial
from  skyfield.api import load, Topos
from astropy.time import Time
import gc 
import os 
import itertools as it
from itertools import chain
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, curve_fit
from  scipy.linalg import block_diag


'''
---------------------------- Constants ---------------------------- 
'''
epsilon = 23.4392811*np.pi/180 #obliquity
GM = 4*np.pi**2

planets = load('de421.bsp')
earth = planets['earth']
ctio = Topos('-30.16606 N','70.81489 E', elevation_m=2215)
observer = earth + ctio 

time_skyfield = load.timescale(0)  
one_arc_sec_sq = (np.pi/180)**2*(1./3600)**2
r_decam = np.sqrt(3.3*np.pi)/180 #A = pi r^2 => r = sqrt(A/pi) * pi/180

'''
---------------------------- Coordinate system ---------------------------- 
'''
class CoordinateSystem:
	'''
	Fixes a coordinate system for the tangent plane projection according to a b0, l0, t0, and r0.
	This class has some utility functions to make things easier as needed, like angle conversions
	'''
	def __init__(self, t0, ra_0 = None, dec_0 = None, b0 = None, l0 = None, r0 = None, latitude = None, longitude = None, elevation = None):
		if b0 == None:
			b0, l0 = b_l(np.pi*ra_0/180, np.pi*dec_0/180)
			self.ra_0 = ra_0
			self.dec_0 = dec_0
		self.b0 = b0
		self.l0 = l0
		self.t0 = t0
		if latitude != None:
			self.telescope = Topos(latitude, longitude, elevation_m = elevation)
			self.observer = earth + self.telescope
		else:
			self.telescope = ctio
			self.observer = observer
		if r0 == None:
			self.r0 = self.earth_position(self.t0)
		else:
			self.r0 = r0

		self.tdb0 = Time(self.t0, format='mjd').tdb.byear - 2000
		
	def earth_position(self, t):
		'''
		Finds the barycentric position of the Earth at time t (in MJD)
		'''
		return earth_position(t, self.b0, self.l0, observer = self.observer)

	def telescope_centric_r(self, t):
		'''
		Finds the telescope-centric r vector at mjd t
		'''
		if np.isscalar(t):
			t = [t]
		t = np.array(t)
		return (self.earth_position(t) - self.r0)

	def theta_def(self, ra, dec):
		'''
		Given a (RA, DEC) pair in degrees, returns the tangent plane ecliptic projection with this coordinate system
		'''
		ra =  np.pi*ra/180
		dec = np.pi*dec/180
		b, l = b_l(ra, dec)
		return theta_def(b, l, self.b0, self.l0)

	def ra_dec(self, theta):
		'''
		Given a :math:`\\theta` vector, returns the correspondent (RA, DEC)
		'''
		b, l = theta_to_b_l(theta, self.b0, self.l0)
		ra, dec = ra_dec(b, l)
		return 180*ra/np.pi, 180*dec/np.pi

	def propagate_error_matrix(self, Sigma, ra, dec):
		'''
		Propagates the error matrix from (RA, DEC) to Theta at a given reference system.
		'''
		ra = ra*np.pi/180
		dec = dec*np.pi/180
		b, l = b_l(ra, dec)
		J_ec = J_eq_to_ec(ra, dec)
		#J_tp = J_ec_to_tp(b, l, self.b0, self.l0)

		error = np.linalg.multi_dot([J_ec, Sigma, J_ec.T])

		return error

	def ecliptic_to_telescope(self, x, vel = np.array([0,0,0])):
		'''
		Transforms both a position and a velocity from the ecliptic coordinate system to the telescope-centric coordinate system
		'''
		l0 = self.l0
		b0 = self.b0
		R_bl = np.array([[-np.sin(l0), np.cos(l0), 0],[-np.cos(l0)*np.sin(b0), -np.sin(l0)*np.sin(b0), np.cos(l0)],[np.cos(l0)*np.cos(b0), np.sin(l0)*np.cos(b0), np.sin(l0)]])

		x = R_bl.dot(x)
		vel = R_bl.dot(vel)

		return x - self.r0, vel


'''
---------------------------- Coordinate transformations ---------------------------- 
'''
def b_l(alpha, delta):
	'''
	For a given :math:`(\\alpha,\\delta)` (RA, DEC) measurement, finds the corresponding :math:`(b,\\ell)` angle
	'''
	b = np.arcsin(np.cos(epsilon) * np.sin(delta) - np.sin(epsilon) * np.cos(delta) * np.sin(alpha))
	l = np.arctan2((np.cos(epsilon) * np.cos(delta) * np.sin(alpha) + np.sin(epsilon) * np.sin(delta)), (np.cos(delta) * np.cos(alpha)))
	return b, l

def ra_dec(b, l):
	'''
	For a given :math:`(b,\\ell)` measurement, finds the corresponding :math:`(RA,DEC)` angle
	'''
	dec = np.arcsin(np.cos(epsilon) * np.sin(b) + np.sin(epsilon) * np.cos(b) * np.sin(l))
	ra = np.arctan((np.cos(epsilon) * np.cos(b) * np.sin(l) - np.sin(epsilon) * np.sin(b))/(np.cos(b) * np.cos(l)))
	return ra, dec

def theta_def(b, l, b0, l0):
	'''
	For a given :math:`(b,\\ell)` angle, finds the corresponding :math:`\\vec{\\theta}` angle 
	as measured according to a reference :math:`(b_0,\\ell_0)`
	'''
	den = np.sin(b0) * np.sin(b) + np.cos(b0) * np.cos(b) * np.cos(l - l0)
	theta_x = np.cos(b) * np.sin(l - l0)/den
	theta_y = (np.cos(b0) * np.sin(b) - np.sin(b0) * np.cos(b) * np.cos(l - l0))/den
	return np.array([theta_x, theta_y])

def theta_to_b_l(theta, b0, l0):
	'''
	For a given :math:`\\boldsymbol{\\theta}`, returns the correspondent :math:`(b,\\ell)` coordinates
	'''	
	den = np.sqrt(1 + theta[0]**2 + theta[1]**2)
	sin_b = (np.sin(b0) + theta[1]*np.cos(b0))/den
	sin_l = theta[0]/(np.cos(b0) * den)
	return np.arcsin(sin_b), np.arcsin(sin_l) + l0


def earth_position(date, b0, l0, observer = observer):
	'''
	Uses the skyfield library to read a JPL ephemeris and find the position of the telescope according to a reference frame centered at the Sun and rotated
	by an angle :math:`(b_0, \\ell_0)`
	'''
	time =  time_skyfield.from_astropy(Time(date, format='mjd', scale='utc'))
	R_bl = np.array([[-np.sin(l0), np.cos(l0), 0],[-np.cos(l0)*np.sin(b0), -np.sin(l0)*np.sin(b0), np.cos(l0)],[np.cos(l0)*np.cos(b0), np.sin(l0)*np.cos(b0), np.sin(l0)]])
	
	
	return np.einsum('ij, j...', R_bl, observer.at(time).ecliptic_position().au)


def theta_parallax(alpha, beta, gamma, r, r0):
	'''
	Calculates the parallax-only evolution of the :math:`\\theta` vector, defined by 

	:math:`\\theta_x(t) = \\frac{\\alpha - \\gamma x_E(t)}{1-\\gamma z_E(t)},`

	:math:`\\theta_y(t) = \\frac{\\beta  - \\gamma y_E(t)}{1-\\gamma z_E(t)}.`
	'''			
	r = r - r0
	theta_x = (alpha - gamma*r[0])/(1-gamma*r[2])
	theta_y = (beta  - gamma*r[1])/(1-gamma*r[2])
	return theta_x, theta_y

def theta_parallax_vec(theta, gamma, r, r0):
	'''
	Calculates the parallax-only evolution of the :math:`\\theta` vector, defined by 

	:math:`\\theta_x(t) = \\frac{\\alpha - \\gamma x_E(t)}{1-\\gamma z_E(t)},`

	:math:`\\theta_y(t) = \\frac{\\beta  - \\gamma y_E(t)}{1-\\gamma z_E(t)}.`

	This version is vectorized and accepts a vector as an input.
	'''	

	r = r - r0
	theta_n = (theta - gamma*r[:,:2])/(1 - gamma*r[2])
	return theta_n

def deparallax_catalog(theta, gamma, r):
	'''
	Calculates the parallax-only evolution of the :math:`\\theta` vector, defined by 

	:math:`\\theta_x(t) = \\frac{\\alpha - \\gamma x_E(t)}{1-\\gamma z_E(t)},`

	:math:`\\theta_y(t) = \\frac{\\beta  - \\gamma y_E(t)}{1-\\gamma z_E(t)}.`
	'''			
	theta_n = (1 - gamma * r[2]) * theta + gamma*r[:2]

	return theta_n.reshape(-1,2)

def theta_real(alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot, t, coord_system = None, b0 = None, l0 = None, t0 = None, r0 = None):
	'''
	Calculates the full evolution of the :math:`\\theta` vector without the gravitational contribution, given by

	:math:`\\theta_x(t) = \\frac{\\alpha + \\dot{\\alpha} t- \\gamma x_E(t)}{1 + \\dot{\\gamma} t-\\gamma z_E(t)}`

	:math:`\\theta_y(t) = \\frac{\\beta  + \\dot{\\beta}  t- \\gamma y_E(t)}{1 + \\dot{\\gamma} t-\\gamma z_E(t)}`
	'''
	if coord_system != None:
		r = coord_system.telescope_centric_r(t)
		t = (t - coord_system.t0)/(365.25)
	else:	
		r = earth_position(t, b0, l0)
		r = r - r0[:, None]
		t = (t - t0)/365.25
	
	x = alpha + alpha_dot * t - gamma * r[:, 0]
	y = beta + beta_dot * t - gamma * r[:, 1]
	z = 1 + gamma_dot * t - gamma * r[:, 2]
	return np.array([x/z, y/z])


def theta_gravity(alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot, t, coord_system = None, b0 = None, l0 = None, t0 = None, r0 = None):
	'''
	Calculates the full evolution of the :math:`\\theta` vector with an approximate form of gravitational contribution, given by

	:math:`\\theta_x(t) = \\frac{\\alpha + \\dot{\\alpha} t + \\gamma g_x(t)- \\gamma x_E(t)}{1 + \\dot{\\gamma} t + \\gamma g_z(t)-\\gamma z_E(t)}`

	:math:`\\theta_y(t) = \\frac{\\beta  + \\dot{\\beta}  t + \\gamma g_y(t) - \\gamma y_E(t)}{1 + \\dot{\\gamma} t + \\gamma g_z(t) -\\gamma z_E(t)}`

	Here, :math:`\\gamma \\mathbf{g}(t)` is given by Gauss functions as defined in Holman et al 2018:
	:math:`\\gamma \\mathbf{g}(t) = -\\frac{1}{2} \\boldsymbol{\\alpha} \\sigma t^2 - \\frac{1}{6} \\sigma t^3 (\\boldsymbol{\\dot{\\alpha}} - 3 \\tau \\boldsymbol{\\alpha})`
	'''
	if coord_system != None:
		r = coord_system.telescope_centric_r(t)
		t = (t - coord_system.t0)/(365.25)
	else:	
		r = earth_position(t, b0, l0)
		r = r - r0[:, None]
		t = (t - t0)/365.25
	

	sigma = GM * gamma**3
	tau = gamma_dot
	gamma_g = -0.5*sigma*t**2*np.array([alpha, beta, 1])[:, None] - 1./6 * sigma * t**3 * (np.array([alpha_dot, beta_dot, gamma_dot])[:,None] -3*np.array([alpha, beta, 1])[:,None] * tau)

	x = alpha + alpha_dot * t + gamma_g[0] - gamma * r[:,0]
	y = beta + beta_dot * t + gamma_g[1] - gamma * r[:,1]
	z = 1 + gamma_dot * t + gamma_g[2] - gamma * r[:,2]
	return x/z, y/z


def sun_position(date, cs):
	'''
	Uses the skyfield library to read a JPL ephemeris and find the position of the Sun rotated by an angle :math:`(b_0, \\ell_0)`
	'''
	time =  time_skyfield.from_astropy(Time(date, format='mjd', scale='utc'))
	#R_bl = np.array([[-np.sin(l0), np.cos(l0), 0],[-np.cos(l0)*np.sin(b0), -np.sin(l0)*np.sin(b0), np.cos(l0)],[np.cos(l0)*np.cos(b0), np.sin(l0)*np.cos(b0), np.sin(l0)]])

	x = cs.ecliptic_to_telescope(sun.at(time).ecliptic_position().au)[0]
	return x


def grav_monopole(t, x, cs):
	'''
	Computes the gravitational monopole of the sun
	'''
	sun_x = sun_position(t, cs)	
	r = np.linalg.norm(x - sun_x)

	return - GM * (x - sun_x)/r**3

def f(x, t, x0, v0, cs):
	'''
	The integrated function!
	'''
	dx = x[:3]
	dv = x[3:]
	dvdt = grav_monopole(365.25*t, x0 + v0*(t-t0) + dx, t0, b0, l0)
	dxdt = dv

	return np.array([dxdt, dvdt]).flatten()


def gravitational_term(t, cs, x0, v0):
	'''
	Returns the gravitational monopole integral for a TNO
	'''
	'''if t_start == t_end:
		return [0,0,0]'''

	#dt = np.linspace(t_start, t_end, 1000)
	x = np.array([0,0,0,0,0,0])
	x[:3] = x0
	x[3:] = v0

	integral = solve_ivp(f, x, t, args=(x0, v0, cs))
	return integral

def theta_integral(t_end, t_start, params, cs, n_steps):
	
	t = np.linspace(t_start, t_end, n_steps)
	r = coord_system.telescope_centric_r(t)
	t = (t - coord_system.t0)/(365.25)
	

	x0 = np.array([params[0], params[1], 1])/params[2]
	v0 = np.array([params[3], params[4], params[5]])/params[2]

	g = gravitational_term(t, cs, x0, v0)

	x = params[0] + params[3]*t + g[0]*params[2] - params[2]*r[0]
	y = params[1] + params[4]*t + g[1]*params[2] - params[2]*r[1]

	z = 1 + params[5]*(t) + g[2]*params[2] - params[2]*r[2]

	return np.array([x/z, y/z])


def max_proper(gamma, t1, t2):
	'''
	Max distance that proper motion can make an object travel and still be in a bound orbit:

	:math:`d_{\\max} = 2 \\sqrt{2\\gamma^3}\\Delta t`
	'''
	return 2*np.pi*np.sqrt(2*np.power(gamma, 3))*(t2 - t1)/365.25

def alpha_est(theta_1, theta_2, t1, t2, gamma, coord = None, t0 = None, r0 = None):
	'''
	Given two measurements :math:`\\vec{\\theta}_{1,2}`, finds the :math:`\\vec{\\alpha}` parameter that best describes it, for a fixed :math:`\\gamma`
	'''
	if coord != None:
		r_vec = coord.telescope_centric_r(np.array([t1, t2]))
		r1 = r_vec.T[0]
		r2 = r_vec.T[1]
		t2 = (t2 - coord.t0)/365.25
		t1 = (t1 - coord.t0)/365.25
	else:
		r1 = earth_position(t1, b0, l0)
		r2 = earth_position(t2, b0, l0)

		r1 = r1 - r0
		r2 = r2 - r0 
		t1 = (t1 - t0)/365.25
		t2 = (t2 - t0)/365.25

	M = np.array([[1, 0, t1, 0],[0, 1, 0, t1], [1, 0, t2, 0], [0, 1, 0, t2]])
	M_inv = np.linalg.inv(M)

	theta1_p = theta_1*(1 - gamma*r1[2]) + gamma*r1[:2]
	theta2_p = theta_2*(1 - gamma*r2[2]) + gamma*r2[:2]

	theta_v = np.bmat([theta1_p, theta2_p])

	params = M_inv.dot(theta_v.T)
	return params

def alpha_est_pair(pair, gamma, exposures):
	'''
	Pair version of the :math:`(\\alpha,\\beta,\\dot{\\alpha},\\dot{\\beta})` fit, given two measurements :math:`\\vec{\\theta}_{1,2}`, 
	finds the :math:`\\vec{\\alpha}` parameter that best describes it, for a fixed :math:`\\gamma`
	'''
	r1 = exposures[pair.det1.expnum].r
	r2 = exposures[pair.det2.expnum].r

	dr = r2 - r1
	dt = pair.det2.mjd - pair.det1.mjd
	M = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [1, 0, dt, 0], [0, 1, 0, dt]])
	M_inv = np.linalg.inv(M)

	theta1_p = pair.det1.theta
	theta2_p = pair.det2.theta*(1 - gamma*dr[2]) + gamma*dr[:2]

	theta_v = np.bmat([theta1_p, theta2_p])

	params = M_inv.dot(theta_v.T)
	return params

def J_eq_to_ec(alpha, delta):
	'''
	Jacobian matrix of the :math:`(\\alpha,\\delta)` to :math:`(b,\\ell)` transformation
	'''
	den = np.sqrt(1 - (np.cos(epsilon) * np.sin(delta) - np.cos(delta) * np.sin(alpha) * np.sin(epsilon))**2.)
	J00 = -np.cos(alpha) * np.cos(delta) * np.sin(epsilon)/den
	J01 = (np.cos(delta) * np.cos(epsilon) + np.sin(alpha) * np.sin(delta) * np.sin(epsilon))/den
	den = 1 + (np.cos(epsilon) * np.tan(alpha) + (1./np.cos(alpha)) * np.sin(epsilon) * np.tan(delta))**2.
	J10 = (1./(np.cos(alpha)) * (np.cos(epsilon)/np.cos(alpha) + np.sin(epsilon) * np.tan(alpha) * np.tan(delta)))/den
	den = 1 + (np.cos(epsilon) * np.tan(alpha))**2. + np.tan(alpha)/np.cos(alpha) * np.sin(2*epsilon) * np.tan(delta) + np.sin(epsilon)**2. * np.tan(delta)**2/np.cos(alpha)**2
	J11 = np.sin(epsilon)/(np.cos(alpha) * np.cos(delta)**2. * den)
	return np.array([[J00, J01],[J10, J11]])

def J_ec_to_tp(b, l, b0, l0):
	'''
	Jacobian matrix of the :math:`(b,\\ell)` to :math:`(\\theta_x,\\theta_y)` transformation, for a given reference :math:`(b_0, \\ell_0)`
	'''
	den = (np.cos(b0) * np.cos(b) * np.cos(l - l0) + np.sin(b0) * np.sin(b))**2.
	J00 = - np.sin(b0) * np.sin(b - l0)/den 
	J01 = np.cos(b) * (-np.cos(b0) * np.cos(b) + np.cos(l-l0) * np.sin(b0) * np.sin(b))/den
	J10 = -(3 + np.cos(2*l - 2*l0))*np.sin(2*b0)/(4*den)
	J11 = ((np.cos(b)*np.cos(b0))**2 + (np.sin(b)*np.sin(b0))**2)*np.sin(l - l0)/(den)
	return np.array([[J00, J01],[J10, J11]])

def compute_distance(theta_1, theta_2):
	'''
	Computes 
	:math:`||\\vec{\\theta}_1  - \\vec{\\theta}_2||`
	for a list of vectors
	'''
	return np.linalg.norm(theta_1 - theta_2, axis=1)

def solar_elongation(pair, exposures):
	'''
	Computes the cosine of the angle :math:`\\beta_0` for a pair, corresponding to the solar elongation of the target at the time of the first detection:

	:math:`\\cos\\beta_0 = -\\frac{(\\alpha, \\beta, 1) \\cdot \\mathbf{r}_0}{r_0 \\sqrt{\\alpha^2 + \\beta^2 + 1}}`
	'''
	r0 = -exposures[pair.det1.expnum].r
	vec = np.array([pair.det1.x, pair.det1.y, 1])
	prod = np.inner(r0, vec)
	r0 = np.linalg.norm(r0)
	vec = np.linalg.norm(vec)

	return prod/(r0*vec)

def gamma_dot_bind(gamma, alpha_dot, beta_dot, cos_beta):
	'''
	Computes a limiting value for :math:`\\dot{\\gamma}`, given by 
	
	:math:`\\dot{\\gamma}_{\\text{bind}}^2 = 8 \\pi^2 \\gamma^3(1 + \\gamma^2 - 2 \\gamma \\cos\\beta_0)^{-1/2} - \\dot{\\alpha}^2 - \\dot{\\beta}^2`
	'''
	r = 1 + gamma**2. - 2 * cos_beta * gamma 
	gamma_bind_sq = 8 * np.pi**2. * gamma**3./np.sqrt(r) - alpha_dot**2 - beta_dot**2

	return np.sqrt(gamma_bind_sq)

def covariance_measurement(t3, t2, t1, Sigma_tp_1, Sigma_tp_2):
	'''
	Finds the error matrix in the :math:`\\boldsymbol{\\alpha}` parameters given :math:`\\Delta t \\equiv t_2 - t_1` and the measurements. Then, proceeds to 
	find the error matrix in a position :math:`t_3`.
	'''
	Sigma = np.bmat([[Sigma_tp_1, np.zeros_like(Sigma_tp_1)],[np.zeros_like(Sigma_tp_2), Sigma_tp_2]])
	dt = (t2 - t1)/365.25 

	M = np.matrix([[1,0,0,0],[0,1,0,0],[-1/dt,0,1/dt,0],[0,-1/dt,0,1/dt]])

	Sigma_params = np.einsum('ij,jk,mk', M, Sigma, M)

	Dt = (t3 - t1)/365.25
	N = np.matrix([[1,0,Dt,0],[0,1,0,Dt]])

	Sigma_measurement = np.einsum('ij,jk,mk',N, Sigma_params, N)

	return Sigma_measurement

def covariance_analytic(t3, Delta_t, Sigma_tp_1, Sigma_tp_2):
	'''
	Finds the error matrix in the :math:`\\boldsymbol{\\alpha}` parameters given :math:`\\Delta t \\equiv t_2 - t_1` and the measurements. Then, proceeds to 
	find the error matrix in a position :math:`t_3`. Analytic solution
	'''
	return Sigma_tp_1 + ((t3 - 1)/Delta_t)**2 * Sigma_tp_2




