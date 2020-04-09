"""
hw6.py
Name(s):
NetId(s):
Date:
"""

import math
import numpy
import matplotlib.pyplot as plt

"""
FE: Forward Euler
"""
def FE(w0, z, m, w, x0, T, N):
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	xV = [0]*(N+1)
	xV[0]=x0

	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	t = [0]*(N+1)
	for s in range(N+1):
		t[s] = s*dt
	for n in range(N):
		b = numpy.matrix([[0],[math.cos(w*t[n])]])
		xV[n+1] = xV[n]+ dt*A*xV[n]+dt*b
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
BE: Backward Euler
"""
def BE(w0, z, m, w, x0, T, N):
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	xV = [0]*(N+1)
	xV[0]=x0

	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	t = [0]*(N+1)
	for s in range(N+1):
		t[s] = s*dt
	for n in range(N):
		b = numpy.matrix([[0],[math.cos(w*t[n+1])]])
		xV[n+1] = (numpy.matrix([[1,0],[0,1]])-dt*A).I*(xV[n]+dt*b)
		x[n+1]=xV[n+1].A[0][0]

	return (x,t)

"""
CN: Crank-Nicolson
"""
def CN(w0, z, m, w, x0, T, N):
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	xV = [0]*(N+1)
	xV[0]=x0

	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	t = [0]*(N+1)
	for s in range(N+1):
		t[s] = s*dt
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
	A = numpy.matrix([[0,1],[-(w0**2),-2*z*w0]])	
	dt = T/N

	xV = [0]*(N+1)
	xV[0]=x0

	x = [0]*(N+1)
	x[0] = x0.A[0][0]

	t = [0]*(N+1)
	for s in range(N+1):
		t[s] = s*dt
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

	correct = (1/2)*(math.sin(10)-10*(math.e**(-10)))

    # Create the range of N values.
	N = [10**p for p in range(2,5)]

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

	# plot the errors vs N for left point rule
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

	# plot the errors vs N for left point rule
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

	# plot the errors vs N for left point rule
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

	# plot the errors vs N for left point rule
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






