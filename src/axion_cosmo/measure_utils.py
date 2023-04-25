"""
utility functions for identifying interesting features in the simulation
given complex field values in the box,
return the locations of strings, domain walls, and overdensities
"""


import numpy as np


def tempcut(dtheta):
    '''
    making sure the angle change is smaller than pi
    '''
    return (dtheta - 2*np.pi)*(dtheta > np.pi) + \
           (dtheta + 2*np.pi)*(dtheta < -np.pi) + \
           (dtheta)*(dtheta < np.pi)*(dtheta > -np.pi)

def plaq(theta,i,j):
    '''
    calculate the total change of angle around a plaquette (in field space)
    '''

    return tempcut(np.roll(theta,-1,axis=i) - theta) + \
            tempcut(np.roll(np.roll(theta,-1,axis=i),-1,axis=j) - np.roll(theta,-1,axis=i)) + \
            tempcut(np.roll(theta,-1,axis=j) - np.roll(np.roll(theta,-1,axis=i),-1,axis=j)) + \
            tempcut(theta - np.roll(theta,-1,axis=j))



def stringid(field):
    '''
    give the locations of where the strings penetrate a plaquette
    by calculating the total angle around a plaquette

    see jupyter notebook for how to visualize the data it returns
    '''

    strings = []
    theta = np.angle(field)
    
    p01 = plaq(theta,0,1)
    p02 = plaq(theta,0,2)
    p12 = plaq(theta,1,2)
    
    i,j,k = np.where(np.abs(p01) > 6)
    for c in zip(list(i),list(j),list(k)):
        strings.append(([c[0]+0.5,c[1]+0.5,c[2]-0.5],[c[0]+0.5,c[1]+0.5,c[2]+0.5]))
    i,j,k = np.where(np.abs(p02) > 6)
    for c in zip(list(i),list(j),list(k)):
        strings.append(([c[0]+0.5,c[1]-0.5,c[2]+0.5],[c[0]+0.5,c[1]+0.5,c[2]+0.5]))
    i,j,k = np.where(np.abs(p12) > 6)
    for c in zip(list(i),list(j),list(k)):
        strings.append(([c[0]-0.5,c[1]+0.5,c[2]+0.5],[c[0]+0.5,c[1]+0.5,c[2]+0.5]))
    
    return strings


def domainid(field):
    '''
    give the locations of strings and domain walls 
    by calculating the signed crossing of the real axis of a plaquette
    '''

    def temp(i):
        cut   = (np.imag(field)*np.imag(np.roll(field,-1,axis=i)) < 0)
        dr    = 2*(np.imag(field*np.roll(np.conj(field),-1,axis=i)) < 0) - 1
        return cut*dr
    sc0 = temp(0)
    sc1 = temp(1)
    sc2 = temp(2)
    
    a01 = sc0 + np.roll(sc1,-1,axis=0) - np.roll(sc0,-1,axis=1) - sc1
    a02 = sc0 + np.roll(sc2,-1,axis=0) - np.roll(sc0,-1,axis=2) - sc2
    a12 = sc1 + np.roll(sc2,-1,axis=1) - np.roll(sc1,-1,axis=2) - sc2
    
    def temp2(i):
        return (np.imag(field)*np.imag(np.roll(field,-1,axis=i)) < 0)*\
                (np.imag(field*np.roll(np.conj(field),-1,axis=i))*\
                 np.imag(field-np.roll(field,-1,axis=i)) < 0)
    
    p0 = temp2(0)
    p1 = temp2(1)
    p2 = temp2(2)

    return [a01,a02,a12],[p0,p1,p2]


def energy_density(field,fieldp,ma,fa,eta,scale):
    '''
    gives energy density after the QCD phase transition
    '''
    def grad(field,dir):
        # give gradient in one direction
        return (-np.roll(field,-1,axis=dir) + \
                np.roll(field,1,axis=dir))
    def sum_gradsq_scaled(field,eta,fa,scale): 
        # give gradient squared
        axes = len(np.shape(field))
        result = np.zeros(np.shape(field))
        for i in range(axes):
            result += grad(field,i)**2 * (scale*eta/fa)**2  # H1=fa
            return result
    def prime_atan(A,B,Ap,Bp):
        # take derivative of arc tangent
        return ( Ap/B - A*Bp/B**2 ) / (1 + (A/B)**2)

    psi1, psi2 = np.real(field), np.imag(field)
    psi1p, psi2p = np.real(fieldp), np.imag(fieldp)

    theta = np.angle(field)

    return fa**4 / (2* eta**2) * (prime_atan(psi1,psi2,psi1p,psi2p)**2 + sum_gradsq_scaled(theta,eta,fa,scale)) + fa**2 * ma**2 *(1-np.cos(theta))



