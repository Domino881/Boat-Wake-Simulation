# Boat wake simulation (06.11.2022)
# Author: Dominik Kuczynski

# Code for generating 3d plots of boat wake patterns.
# A point source of surface waves of different wavelengths ('lambdas')
# travels at speed 'v', ending up in a set place. 
# To speed up computation time, some values are first stored in large arrays
# during preprocessing. Other factors that affect time are t_nsteps
# and n_lbdas.

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter

# initiate the axes
ymax = 40
xmax = ymax
pixelsize = 0.4
proportion = 8.0/9.0
x = np.arange(-xmax, xmax, pixelsize)
y = np.arange(0, 2*ymax, pixelsize)
X, Y = np.meshgrid(x, y)
Z = np.ones(np.shape(X))


############################### set system properties #########################
v = 4.13 # speed of boat
A = 5e-5 # amplitude of displacement

################################ set time variables ###########################
tmax = (2*ymax*proportion+5.0)/v
t_nsteps=400
tstep = 2*tmax/t_nsteps

################################ set lambda variables #########################
lbdamin = 0.423
lbdamax = 4.5
n_lbdas = 100
lbda_step = (lbdamax-lbdamin)/n_lbdas
lbdas = np.arange(lbdamin, lbdamax, lbda_step)


# shift Y axis to place boat in the right place
shift = v*tmax - 2*proportion*ymax
Y = Y+shift

# dispersion relation for deep water waves
def gen_omega(k):
    return np.sqrt(9.81*k) # cp approx. 3.13

def preprocess():
    # do preprocessing
    xsize = int(np.ceil(2*xmax/pixelsize))
    ysize = int(np.ceil(2*ymax/pixelsize))
    
    r = np.zeros((t_nsteps, ysize, xsize))
    # r - 3d array of distances from the points source at given time
    
    checks = np.full((n_lbdas, t_nsteps, ysize, xsize), False)
    # checks - 4d array that stores boolean values indicating which
    # points should be affected by a wave starting at a given time and 
    # propagating at speed cp.
    
    for t in range(0, t_nsteps):
        r[t] = (np.sqrt(X**2+(Y-v*t*tstep)**2))
        print("\r"+"preprocessing: %.0f" %(100.0*float(t+1)/t_nsteps) + "%", end='')
        for lbda in range(len(lbdas)):
            k = 2*np.pi/lbdas[lbda]
            cp = gen_omega(k)/k
            checks[lbda][t] = (r[t] <= cp*(tmax-t*tstep))
    print("")
    return checks, r
    
checks, r = preprocess()

# run simulation
for lbda in range(len(lbdas)):        
    k = 2*np.pi/lbdas[lbda]
    omega = gen_omega(k)
    cp = omega/k

    for t in range(t_nsteps): 
        Z += A*checks[lbda][t]*np.exp(-0.02*r[t]*(tmax-t*tstep))*\
            np.cos((k*r[t]-omega*(tmax-t*tstep)))
        # add to Z a damped cosine surface wave originating at (0,vt) and
        # propagating for tmax-t time
        
    print('\rlambda = %.3f  - %d/%d' %(lbdas[lbda], lbda+1, len(lbdas)),
          end='', flush=True)
print("")

# create the figure
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=35, azim=110)
ax.set(zlim=[0,2])
ax.set_title("%d lambdas in (%.2f, %.2f), " %(n_lbdas, lbdamin, lbdamax) +\
             "ymax=%d, pixelsize=%.2f, t_nsteps=%d"%(ymax,pixelsize,t_nsteps)+\
             ", v=%.2f"%(v))

# plot the surface
ax.plot_surface(X,Y,Z, cmap = plt.cm.bone, rcount=400, ccount=400,
                antialiased=False)