"""
hw6.py
Name(s): Pranay Pherwani
NetId(s): prp12
Date: 4/9/20
"""

import math
import numpy
import matplotlib.pyplot as plt

"""
FE: Forward Euler
"""
def FE(w0, z, m, w, x0, T, N):
	# calculate matrix A and delta t
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	# Initialize the vector x
	xV = [0]*(N+1)
	xV[0]=x0

	# Initialize x representing displacement
	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	# Initialize t representing time
	t = [0]*(N+1)

	# Fill t with the timesteps
	for s in range(N+1):
		t[s] = s*dt

	# Calculate b and use the update rule to calculate the next vector x
	for n in range(N):
		b = numpy.matrix([[0],[math.cos(w*t[n])]])
		xV[n+1] = xV[n]+ dt*A*xV[n]+dt*b

		# Get the displacement x from the vector x
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
BE: Backward Euler
"""
def BE(w0, z, m, w, x0, T, N):
	# calculate matrix A and delta t
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	# Initialize the vector x
	xV = [0]*(N+1)
	xV[0]=x0

	# Initialize x representing displacement
	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	# Initialize t representing time
	t = [0]*(N+1)

	# Fill t with the timesteps
	for s in range(N+1):
		t[s] = s*dt

	# Calculate b and use the update rule to calculate the next vector x
	for n in range(N):
		b = numpy.matrix([[0],[math.cos(w*t[n+1])]])
		xV[n+1] = (numpy.matrix([[1,0],[0,1]])-dt*A).I*(xV[n]+dt*b)
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
CN: Crank-Nicolson
"""
def CN(w0, z, m, w, x0, T, N):
	# calculate matrix A and delta t
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	# Initialize the vector x
	xV = [0]*(N+1)
	xV[0]=x0

	# Initialize x representing displacement
	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	# Initialize t representing time
	t = [0]*(N+1)

	# Fill t with the timesteps
	for s in range(N+1):
		t[s] = s*dt

	# Calculate b and use the update rule to calculate the next vector x
	for n in range(N):
		b1 = numpy.matrix([[0],[math.cos(w*t[n])]])
		b2 = numpy.matrix([[0],[math.cos(w*t[n+1])]])
		xV[n+1] = ((numpy.matrix([[1,0],[0,1]])-(dt/2)*A).I*((numpy.matrix([[1,0],[0,1]])+(dt/2)*A)*xV[n]+dt*(b1+b2)/2))
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
RK4: fourth order Runge-Kutta
"""
def RK4(w0, z, m, w, x0, T, N):
	# calculate matrix A and delta t
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	# Initialize the vector x
	xV = [0]*(N+1)
	xV[0]=x0

	# Initialize x representing displacement
	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	# Initialize t representing time
	t = [0]*(N+1)

	# Fill t with the timesteps
	for s in range(N+1):
		t[s] = s*dt

	# Calculate b and use the update rule to calculate the next vector x
	for n in range(N):
		b1 = numpy.matrix([[0],[math.cos(w*t[n])]])
		k1 = dt*A*xV[n]+dt*b1

		b2 = numpy.matrix([[0],[math.cos(w*(t[n]+dt/2))]])
		k2 = dt*A*(xV[n]+k1/2)+dt*b2

		b3 = numpy.matrix([[0],[math.cos(w*(t[n]+dt/2))]])
		k3 = dt*A*(xV[n]+k2/2)+dt*b3

		b4 = numpy.matrix([[0],[math.cos(w*(t[n]+dt))]])
		k4 = dt*A*(xV[n]+k3)+dt*b4

		xV[n+1] = xV[n] + k1/6 + k2/3 + k3/3 + k4/6
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
main
"""
if __name__ == '__main__':

	# Find the actual value for step 3
	correct = (1/2)*(math.sin(10)-10*(math.e**(-10)))

    # Create the range of N values.
	N = [10**p for p in range(2,4)]

	# Initialize values lists
	FEValues = []
	BEValues = []
	CNValues = []
	RK4Values = []

	# Initialize errors lists
	FEErrors = []
	BEErrors = []
	CNErrors = []
	RK4Errors = []

	# For each N, calculate the value for each rule and the error
	for n in N:
		# calculate the values for each rule
		fe = FE(1,1,1,1,numpy.matrix([[0],[0]]),10,n)[0][-1]
		be = BE(1,1,1,1,numpy.matrix([[0],[0]]),10,n)[0][-1]
		cn = CN(1,1,1,1,numpy.matrix([[0],[0]]),10,n)[0][-1]
		rk4 = RK4(1,1,1,1,numpy.matrix([[0],[0]]),10,n)[0][-1]

		# append the values to the lists
		FEValues.append(fe)
		BEValues.append(be)
		CNValues.append(cn)
		RK4Values.append(rk4)

		# calculate the errors and append them to the errors lists
		FEErrors.append(abs((fe-correct)/correct))
		BEErrors.append(abs((be-correct)/correct))
		CNErrors.append(abs((cn-correct)/correct))
		RK4Errors.append(abs((rk4-correct)/correct))

	# plot the errors vs N for FE
	plt.figure()
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.plot(N, FEErrors, label = 'FE errors')
	legend = ax.legend(loc='upper left')
	plt.title('FE Errors vs N')
	plt.xlabel('N')
	plt.ylabel('Error')
	plt.savefig('FE.png', bbox_inches='tight') 
	plt.close('all')

	# plot the errors vs N for BE
	plt.figure()
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.plot(N, BEErrors, label = 'BE errors')
	legend = ax.legend(loc='upper left')
	plt.title('BE Errors vs N')
	plt.xlabel('N')
	plt.ylabel('Error')
	plt.savefig('BE.png', bbox_inches='tight') 
	plt.close('all')

	# plot the errors vs N for CN
	plt.figure()
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.plot(N, CNErrors, label = 'CN errors')
	legend = ax.legend(loc='upper left')
	plt.title('CN Errors vs N')
	plt.xlabel('N')
	plt.ylabel('Error')
	plt.savefig('CN.png', bbox_inches='tight') 
	plt.close('all')

	# plot the errors vs N for RK4
	plt.figure()
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.plot(N, RK4Errors, label = 'RK4 errors')
	legend = ax.legend(loc='upper left')
	plt.title('RK4 Errors vs N')
	plt.xlabel('N')
	plt.ylabel('Error')
	plt.savefig('RK4.png', bbox_inches='tight') 
	plt.close('all')

	# Part 4

	# Use CN to calculate the values for w=0.8,0.9,1
	(x1,times) = CN(1,0,1,0.8,numpy.matrix([[0],[0]]),100,100)
	x2 = CN(1,0,1,0.9,numpy.matrix([[0],[0]]),100,100)[0]
	x3 = CN(1,0,1,1,numpy.matrix([[0],[0]]),100,100)[0]

	# plot x vs t for w=0.8
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(times, x1, label = 'x(t)')
	legend = ax.legend(loc='upper left')
	plt.title('x vs t for w=0.8')
	plt.xlabel('t')
	plt.ylabel('x')
	plt.savefig('x1.png', bbox_inches='tight') 
	plt.close('all')

	# plot x vs t for w=0.9
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(times, x2, label = 'x(t)')
	legend = ax.legend(loc='upper left')
	plt.title('x vs t for w=0.9')
	plt.xlabel('t')
	plt.ylabel('x')
	plt.savefig('x2.png', bbox_inches='tight') 
	plt.close('all')

	# plot x vs t for w=1
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(times, x3, label = 'x(t)')
	legend = ax.legend(loc='upper left')
	plt.title('x vs t for w=1')
	plt.xlabel('t')
	plt.ylabel('x')
	plt.savefig('x3.png', bbox_inches='tight') 
	plt.close('all')

	# Part 5

	# Initialize w list and max displacement list
	wValues = []
	maxValues = []

	# Set w list to 0.1 to 10 with 0.1 increments
	for i in range(1,101):
		wValues.append(i/10)

	# Calculate max values from displacement lists using CN for each w
	for w in wValues:
		x = CN(1,1/10,1,w,numpy.matrix([[0],[0]]),100,100)[0]
		absValues =[abs(n) for n in x]
		maxValues.append(max(absValues))

	# plot the maximum displacement vs w
	plt.figure()
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.plot(wValues, maxValues, label = 'max displacement')
	legend = ax.legend(loc='upper left')
	plt.title('Maximum displacement vs w')
	plt.xlabel('w')
	plt.ylabel('max displacement')
	plt.savefig('Displacement.png', bbox_inches='tight') 
	plt.close('all')










