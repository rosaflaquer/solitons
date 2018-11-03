# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:47:24 2018; edited on Fri Nov  2 23:10:54 2018

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
limits=15
dz=0.1 #spacing
Nz=(limits-(-limits))/dz #number of points
z=np.linspace(-limits,limits,Nz) #position vector, from -10 to 10 Nz points
dt=0.009 #time interval

#function to evolve
def gaussian(x,mu,sigma):
    return np.exp(-(x - mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))
    
#normalitzation function
def Normalitzation(array,h):
    """
    Computes the normalitzation constant using the simpson's method for a 1D integral
    the function to integrate is an array, h is the spacing between points 
    and returns a float
    """
    constant=0.0
    for i in range(len(array)):
        if (i == 0 or i==len(array)):
            constant+=array[i]*array[i].conjugate()
        else:
            if (i % 2) == 0:
                constant += 2.0*array[i]*array[i].conjugate()
            else:
                constant += 4.0*array[i]*array[i].conjugate()
    constant=(h/3.)*constant
    return np.sqrt(1/np.real(constant))
    
#sytem at time t, it has to include all the boundary conditions
time_0=[]
for position in z:
    time_0.append(gaussian(position,0,1))

time_0=np.asanyarray(time_0) #turs to an ndarray (needed for the tridiag solver)

#normalitzation, we compute it with the Simpson's method:
time_0=time_0*Normalitzation(time_0,dz)

#we store the norm at t=0 as it will be useful for cheking if it is preserved during
#the time evolution.
norm_0=Normalitzation(time_0,dz)


#system at time t+1
time_1=time_0

#matrixs for the Crank-Nicholson method
#first [] the numbers to plug in the diagonals, second [] position of the 
#diagonals (0 is the main one), shape: matrix's shape
r_free= (1j*dt)/(2*dz**2) #parameter of the method for free space
A=diags([-r_free,2*(1+r_free),-r_free],[-1,0,1], shape=(len(time_0),len(time_0)))
A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
B=diags([r_free,2*(1-r_free),r_free],[-1,0,1], shape=(len(time_0),len(time_0)))

#plot of the square modulus of phy at t=0
plt.ylabel('$|\psi(\~z)|^2$')
plt.xlabel('$\~z =z\sqrt{m/\hbar}$')
plt.plot(z,np.real(time_0*time_0.conjugate()))

energy=0
t=0
dif_norm=0.

while t < 1:
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
print('norm_diference:',dif_norm)
print('r',r_free)
# your code
elapsed_time = time.time() - start_time
print('computing time=',elapsed_time)

