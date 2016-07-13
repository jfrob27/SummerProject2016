import scipy
from scipy import interpolate
import numpy as np
import math


#added imatrix output to look at individual scales
def perlin2d(n,p):
    total = 0
    fmax=(2**(n))
    imatrix = np.zeros((fmax,fmax,n-2))
    coord=np.arange(fmax)
    for i in range(2,n):
        f=(2**(i))
        a=(p**(i))
        randz=np.random.uniform(-1,1,size=(f,f))
        #randy=np.random.uniform(-1,1,size=(f))
        x=np.linspace(0,fmax-1,f)
        y=np.linspace(0,fmax-1,f)

        #x=np.linspace(0,fmax,f+1)
        #y=np.linspace(0,fmax,f+1)
        #x,y= np.meshgrid(x,y)
        #zmat=np.arange(0,fmax,fmax/f)
        #coord=np.arange(fmax)
        g=scipy.interpolate.RectBivariateSpline(y,x,randz)
        intmat=g(coord,coord)*a
        total=(intmat)+total
        imatrix[:,:,i-2]=intmat
    return total, imatrix

