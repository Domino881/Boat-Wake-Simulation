# Code for generating 3d plots of boat wake patterns.
# A point source of surface waves of different wavelengths ('lambdas')
# travels at speed 'v' and always ends up in the same place of the graph -
# x=0, y=proportion*ymax. 
# Some values are first stored in large arrays during preprocessing to
# speed dup computation time. Other factors that affect time are t_nsteps
# and n_lbdas. Gen_omega is the dispersion relation for deep water waves.

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111,projection='3d')
ax.view_init(elev=90, azim=270)
ax.set(zlim=[0,2])

proportion = 8.0/9.0
ymax = 60
# xmax = 2*np.tan(19.47*360/(2*np.pi))*proportion*ymax
xmax = 60
pixel = 0.1

x = np.arange(-xmax, xmax, pixel)
y = np.arange(0, 2*ymax, pixel)
X, Y = np.meshgrid(x, y)
Z = np.zeros(np.shape(X))


v=8.0
A = 0.002

#set time variables
tmax = (2*ymax*proportion+5.0)/v
t_nsteps=50
tstep = tmax/t_nsteps

#set the shift
shift = v*tmax - 2*proportion*ymax
Y = Y+shift

#set lambda variables
lbdamin = 0.5
lbdamax = 5.0
n_lbdas = 30
lbda_step = (lbdamax-lbdamin)/n_lbdas
lbdas = np.arange(lbdamin, lbdamax, lbda_step)

def gen_omega(k):
    return np.sqrt(9.81*k)

def preprocess():
    # do preprocessing
    xsize = int(np.ceil(2*xmax/pixel))
    ysize = int(np.ceil(2*ymax/pixel))
    
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

# run code
for lbda in range(len(lbdas)):
    print('\rlambda = %.3f  - %d/%d' %(lbdas[lbda], lbda, len(lbdas)),
          end='', flush=True)
        
    k = 2*np.pi/lbdas[lbda]
    omega = gen_omega(k)
    cp = omega/k

    for t in range(t_nsteps): 
        Z += A*checks[lbda][t]*np.cos((k*r[t]+omega*(tmax-t*tstep)))


ax.plot_surface(X,Y,Z, cmap = plt.cm.cividis)
