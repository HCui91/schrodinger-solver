############################################
#   Unsquare potential & Schrodinger Eqn   #
############################################

### Import essential modulus ###
import numpy as np
import matplotlib.pyplot as plt              
# Inline style graphs run faster on my PC

### Universal physics constants ###
m = 9.10938356 * 10**(-31)  # electron mass in kilogram
hbar =  1.0545718 * 10**(-34) # angular Planck's constant in Js
e = 1.60217662 * 10**(-19)   # electron charge in Coulomb

def V(x):
	"""Finite square well potential function
	Input : x : position for x
	Output: Vx: Potential at position x"""
	
	##########################################################################
	### Because I want to plot the potential with input as an array          #
	### Python gives error when doing "if" to an array                       #
	### so I added this grand "if" to distinguish an array or a single number#
	##########################################################################
	
	### Different Output type (single float or array) depends on input type ###
	if np.size(x)==1:
		### For a single number ###
		if np.abs(x)>=a/2:
			return V_0
		else:
			return 0
	else:          # For an array, comparing every element
		V=np.zeros(np.size(x))    # An zeroes array with the same size with x
		for i in range(np.size(x)):
			if np.abs(x[i])>=a/2: # Using absolute value for both sides
				V[i]=V_0          # Finite potential at two sides
			else:
				V[i]=0
		return V
	print("Error, np.size(x)=",np.size(x))      # Debug if error
	return 0
### This function is updatable later if different potential is required ###

### Seperated Schrodinger equation ###
def schro(r,x,E):
	"""1D Time-independent Schrodinger Equation
	Input: r : wavefunction and energy correspond to x position
		   x: the position of electron
		   E: energy of the wavefunction
	Output: dpsidx: first derivative of wavefunction w.r.t. x
			dphidx: second derivative of wavefunction w.r.t. x"""
	psi = r[0]
	phi = r[1]      
	dpsidx = phi                           # Equation (2)
	dphidx = 2*m*(V(x)-E)*psi/(hbar**2)    # Equation (3)
	return np.array([dpsidx,dphidx])
	
### forth-order Runge Kutta function ###
def RungeKutta2d(r,xpoints,E):
	'''Fourth-order Runge-Kutta for two differential equations
	Inputs: r: initial guess of [wavefunction psi, first derivative phi]
			xpoints: array of x coordinates.
			E : Energy of the wave function
	Outputs: [psi, phi]: solutions for psi(x) and phi(x), numpy arrays one longer than xpoints'''
	psi = [] # initialise empty arrays
	phi = []
	for x in xpoints: # loops over all xpoints 
		psi.append(r[0])
		phi.append(r[1])
		# This RK function solves shcrodinger equation
		k1 = h*schro(r,x,E) 
		k2 = h*schro(r+0.5*k1, x+0.5*h,E)
		k3 = h*schro(r+0.5*k2, x+0.5*h,E)
		k4 = h*schro(r+k3, x+h,E)
		r = r + (k1 + 2*k2 + 2*k3 + k4)/6
	# these next two lines calculate for the point at x = xend (Adding up the last element)
	psi.append(r[0])
	phi.append(r[1])
	return np.array([psi, phi]) # convert output to numpy array with 2 rows and N+1 columns
	
### Setting up position array ###
a = 5*10**(-11)       # half-width of the potential well
xstart = -a           # start of the well
xend = a              # end of the well
N = 1000              # number of points for Runge-Kutta
h = (xend - xstart)/N # size of Runge-Kutta steps
xpoints = np.arange(xstart, xend, h)       # x position array of the well 

def secand(E1,E2):
	"""Secand method to find specific wavefunction solution w.r.t. 2 initial guesses
	and normalise the result wavefunction
	Input: E1, E2: 2 intial guesses for the energy
	Output: psi :  normalised wavefunction array"""
	#Initial guess
	phi = 1
	wave1 = RungeKutta2d(np.array([0,phi]),xpoints,E1)[0,N]      # Solve equation with initial guess and extrate last wavefuction(psi) component
	wave2 = RungeKutta2d(np.array([0,phi]),xpoints,E2)[0,N]
	tolerance = e/1000                 # set the tolerance
	err = 1                          # initialise the error variable
	while err > tolerance:
		E3 = E2 - wave2*(E2-E1)/(wave2-wave1)
		err = abs(E2-E1) 
		# reset initial phi for the next iteration
		E1 = E2 
		E2 = E3 
		# and obtain wavefunction at the end
		wave1 = RungeKutta2d(np.array([0, phi]),xpoints,E1)[0,N]
		wave2 = RungeKutta2d(np.array([0, phi]),xpoints,E2)[0,N]
	psi = RungeKutta2d(np.array([0, phi]),xpoints,E1)[0,]
	### Normalising ###
	I = h*(0.5*psi[N]**2+0.5*psi[0]**2+np.sum(psi[1:N-1]**2))   # Use trapezium rule to integrate wavefunction
	psi_n = psi/np.sqrt(I)        # Normalising original wavefunction
	print("Energy is",E1/e,"eV")  # Print out the result energy 
	return psi_n     # Returning normalised wave function array

### Finite potential value ###
V_0 = 500*e      # The step depth of finite potential is 500 eV

### Ploat a graph of potential ###
plt.figure()
plt.plot(xpoints,V(xpoints)/e,label="Potential")
plt.ylabel("Potential Value (in eV)")
plt.xlabel("x (m)")
plt.title("Potential with respect to position")
plt.ylim(-50,550)
plt.savefig("potential.png")

### Plot a graph of many different energy level wavefunctions ###
plt.figure(figsize=(9,4))
plt.plot(xpoints,secand(0*e,50*e)[0:N])
plt.plot(xpoints,secand(150*e,100*e)[0:N])           # Higher initial guesses approch to higher quantum number
plt.plot(xpoints,secand(400*e,350*e)[0:N])
plt.plot(xpoints,secand(650*e,600*e)[0:N])
plt.xlabel("x (m)")
plt.ylabel(r"$\psi(x)$")
plt.title("Different energy states of wavefunctions for infinite square well")
plt.savefig("wave.png")