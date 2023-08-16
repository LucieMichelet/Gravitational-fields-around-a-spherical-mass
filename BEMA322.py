# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:04:51 2022

@author: Lucie
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Bibilothèque 

def A(c,K,r):
    return (c**2)*(1+(K/r))

def B(c,K,r):
    return (1+(K/r))**(-1)

def Aprim(c,K,r):
    return ((-c**2)*K)/(r**2)

def Bprim(c,K,r):
    return (K/(r**2))/((1+K/r)**2)

def F(y,c,K):
    """
    Parameters
    ----------
    y : Matrice de paramètres initiaux.
    c : célérité
    K : constante

    Returns
    -------
    dy : dérivé de la matrice y

    """
    dy=np.zeros(len(y))
    dy[0] = y[4]
    dy[1] = y[5]
    dy[2] = y[6]
    dy[3] = y[7]
    dy[4] = -(Aprim(c,K,y[1])*y[4]*y[5])/A(c,K,y[1])
    dy[5] = -(Aprim(c,K,y[1])*(y[4]**2))/(2*B(c,K,y[1]))-(Bprim(c,K,y[1])*(y[5]**2))/(2*B(c,K,y[1]))+(y[1]/B(c,K,y[1]))*(y[6]**2)+(y[1]*(np.sin(y[2])**2)*(y[7]**2))/B(c,K,y[1])
    dy[6] = -(2/y[1])*y[6]*y[5]+np.sin(y[2])*np.cos(y[2])*(y[7]**2)
    dy[7] = -(2/y[1])*y[7]*y[5]-2*(np.cos(y[2])/np.sin(y[2]))*y[6]*y[7]
    
    return dy


def RS(G,M,c): #calcul du rayon de Schwarzschild
    return (2*G*M)/(c**2)
    

def rg4(f,t0,tf,y0,N,c,K,rs):
    
    b = np.size(y0)
    h = (tf-t0)/N
    y = np.zeros((b,N))
    y[:,0] = y0
    
    for n in tqdm(range(1,N)):
        #La boucle tourne tant que le rayon (distance photon-astre) est plus grand que le rayon de Schwarzschild de l'astre
        if (y[1,n-1]>rs):
            k1 = h*f(y[:,n-1],c,K)
            k2 = h*f(y[:,n-1]+k1/2,c,K)
            k3 = h*f(y[:,-1]+k2/2,c,K)
            k4 = h*f(y[:,n-1]+k3,c,K)
            y[:,n] = y[:,n-1]+(1/6)*(k1 + 2*k2 + 2*k3 + k4)
            
        else:
            print("Arret de l'algorithme, R < Rs")
            break
    
    #On supprime les colonnes de 0 restantes apres le break
    y= np.delete(y,np.where(~y.any(axis=0))[0], axis=1)
    return y



def Traj_photon(y0,h,itermax,F,c,K,rs):
    """
    Parameters
    ----------
    y0 : vecteur - Les données initiales. 
    h : int - Le pas de la méthode.
    itermax : int - Le nombre d’itération maximal.
    c : float - célérité
    r : float - rayon

    Returns
    -------  
    P : matrice de taille (8xN), avec N le nombre d'itteration de l'algo de runge kutta.
    
    """
    
    t0 = 0
    tf = 20
    n = np.size(y0)
    P = np.zeros((n,itermax))
    P[:,0] = y0 
    P = rg4(F, t0, tf, y0, itermax, c, K,rs)
        
    return P



def Visualisation(P,rs,lim) :
    """
    Parameters 
    ----------
    P : Matrice P de taille (8,N).
    rs : Rayon de Schwarzschild du corps.
    lim : limite des axes de la figure.
    
    Returns
    -------
    Affiche la trajectoire du photon.
    
    """
    
    n,p = np.shape(P)
    x = np.zeros(p).T
    y = np.zeros(p).T
    z = np.zeros(p).T
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    #Affichage de la sphere de rayon rs
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    X = rs*np.cos(u) * np.sin(v)
    Y = rs*np.sin(u) * np.sin(v)
    Z = rs*np.cos(v)
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)
    
    #Affichage de la trajectoire
    for i in range(0,p):
        x[i] = P[1,i]*np.sin(P[2,i])*np.cos(P[3,i])
        y[i] = P[1,i]*np.sin(P[2,i])*np.sin(P[3,i])
        z[i] = P[1,i]*np.cos(P[2,i])
        
    ax.plot(x,y,z)
    
    plt.show()    
    
    return 


def alpha1(P):
    return np.arctan((-P[1,-1])/P[1,0])

def alpha2(P):
    return np.arctan(P[5,-1]/(P[1,-1]*P[-1,-1]))

def ecart(A1,A2):
    return abs(A1 - A2)


#%%
#----------------------------------------------Partie1-------------------------------------------------
#
#%% Question 3 et 4

c = 1
G = 1
M = 1
K = (-2*G*M)/(c**2)
rs = RS(G,M,c)

h = 10**(-2)
itermax = 10**3
Y0 = np.array([1,10,np.pi/2,0,0,-1.5,0,-0.1]).T


#Calcul de P et affichage de la trajectoire du photon
P = Traj_photon(Y0,h,itermax,F,c,K,rs)   
Visualisation(P,rs,10)

#%% Trouvons phi qui rebrousse chemin

itermax = 10**3
Phi = -0.038
Y0 = np.array([1,10,np.pi/2,0,0,-1.5,0,Phi]).T

Ptest = Traj_photon(Y0,h,itermax,F,c,K,rs)   
Visualisation(Ptest,rs,10)

#%% Calcul des alpha et de leur ecart en seconde d'arc
Alpha_un = alpha1(P)
Alpha_deux = alpha2(P)

E = ecart(Alpha_un,Alpha_deux)

    
#%% Question 6

#Importation des données
c = 632415
M = 1
G = 4*(np.pi**2)
K = (-2*G*M)/(c**2)
rs = RS(G,M,c)


h = 10**(-3)
itermax = 10**3
Y0 = np.array([1,0.1,np.pi/2,0,0,-1,0,-0.75]).T


#Calcul de P et affichage de la trajectoire du photon
P2 = Traj_photon(Y0,h,itermax,F,c,K,rs)   
Visualisation(P2,rs,0.1)


#Calcul des alpha et de leur ecart en seconde d'arc
Alpha_un = alpha1(P2)
Alpha_deux = alpha2(P2)

E = ecart(Alpha_un,Alpha_deux)


#%%
#------------------------------------------------Partie2----------------------------------------------
#
#%%


def Newton_Cotes4(f,a,b,n):
    coeff = [7,32,12,32,7]
    h = (b-a)/len(coeff)
    result = 0
    for i in range(0,len(coeff)):
        x = a + (i*(b-a))/(len(coeff)-1)
        result += coeff[i]*f(x)
    result = result*(((b-a)*2*h)/45)

    return result


def f(x):
    return (np.pi/2)*np.sin((np.pi/2)*x)

#%%
a = 0
b = 1
n = 500

result = Newton_Cotes4(f,a,b,n)
