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
	return (x,t)

"""
RK4: fourth order Runge-Kutta
"""
def RK4(w0, z, m, w, x0, T, N):
	return (x,t)

"""
main
"""
if __name__ == '__main__':
	print('FE')
	print(FE(1,1,1,1,numpy.matrix([[0],[0]]),10,10))
	print('BE')
	print(BE(1,1,1,1,numpy.matrix([[0],[0]]),10,10))





