# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:47:24 2018

solution for the Gross-Pitaievskii equation for a null external potential and the 
interaction term equal to 0

@author: Rosa
"""

import numpy as np
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import time

start_time = time.time()

#define the spacing and time interval
limits=10
dz=0.01 #spacing
Nz=(limits-(-limits))/dz #number of points
z=np.linspace(-limits,limits,Nz) #position vector, from -10 to 10 Nz points
dt=0.01 #time interval

#function to evolve
def gaussian(x,mu,sigma):
    return np.exp(-(x - mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))
    
def bright(z,t,v,n,z0):
    "solution for a bright soliton whith Vext=0"
    imag= 0.0 + 1j
    arg1=(z*v/np.sqrt(2))
    arg2=((0.5 -v**2/4)*t)
    psi=np.sqrt(n)*(1/np.cosh((z-z0)/np.sqrt(2) -v*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    return psi
    
def grey(z,t,v,n,z0):
    "solution for a dark soliton whith Vext=0"
    imag=0.0 +1j
    psi=np.sqrt(n)*(imag*v + np.sqrt(1-v**2)*np.tanh((z-z0)/np.sqrt(2)-v*t)*np.sqrt(1-v**2))
    return psi
    

#normalitzation function
def Normalitzation(array,h):
    """
    Computes the normalitzation constant using the simpson's method for a 1D integral
    the function to integrate is an array, h is the spacing between points 
    and returns a float
    """
    constant=0.0
    for i in range(len(array)):
        if i == 0:
            constant+=array[i]*array[i].conjugate()
        elif i == len(array):
            constant+=array[i]*array[i].conjugate()
        else:
            if (i % 2) ==0:
                constant += 2*array[i]*array[i].conjugate()
            else:
                constant += 4*array[i]*array[i].conjugate()
    constant=(h/3.)*constant
    return np.sqrt(1/np.real(constant))
    
#sytem at time t, it has to include all the boundary conditions
time_0=[]
for position in z:
    time_0.append(grey(position,0,0,1,0))

time_0=np.asanyarray(time_0) #turs to an ndarray (needed for the tridiag solver)

#normalitzation, we compute it with the Simpson's method:

time_0=time_0*Normalitzation(time_0,dz)

norm_0=Normalitzation(time_0,dz)


#system at time t+1
time_1=time_0

#matrixs for the Crank-Nicholson method
#first [] the numbers to plug in the diagonals, second [] position of the 
#diagonals (0 is the main one), shape: matrix's shape
r= (1j*dt)/(dz**2) #parameter of the method
A=diags([-r,2*(1+r),-r],[-1,0,1], shape=(len(time_0),len(time_0)))
A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
B=diags([r,2*(1-r),r],[-1,0,1], shape=(len(time_0),len(time_0)))

#plot of the square modulus of phy at t=0
plt.plot(z,np.real(time_0*time_0.conjugate()))

energy=0
t=0
dif_norm=0.

while t < 5:
    #ndarray b product of B and the system's state at time t
    prod=B.dot(time_0)
    #solver of a tridiagonal problem
    time_1=linalg.spsolve(A,prod)
    #we store the maximum diference between the norm_0 and the evolved one
    dif_norm=max(dif_norm,abs(norm_0 - Normalitzation(time_1,dz)))
    #redefine each matrix
    time_0=time_1
    t += dt

#plot the evolved function
plt.plot(z,np.real(time_1*time_1.conjugate()))
print(dif_norm)

# your code
elapsed_time = time.time() - start_time
print(elapsed_time)

"""
#try to see how the function matrix.dot() works
C=diags([1,2,3],[-1,0,1], shape=(3,3))
print(C.toarray())
x=[4,5,6]
x=np.asanyarray(x)
print(C.dot(x))
"""
"""
#chek if the linalg.spsolve works well with an analytic problem
D=scipy.sparse.csr_matrix(diags([1,2,3],[-1,0,1], shape=(3,3)))
e=np.asanyarray([23,32,17])
print(linalg.spsolve(D,e))
"""

