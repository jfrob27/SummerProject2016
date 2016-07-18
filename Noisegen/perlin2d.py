import scipy.interpolate
import numpy as np

def perlin2d(n,p):

    total = 0
    fmax = (2**(n))
    imatrix = np.zeros((fmax,fmax,n-2))
    coord = np.arange(fmax)

    for i in range(2,n):
        f = (2**(i))
        a = (p**(i))
        randz = np.random.uniform(-1,1,size=(f,f))
        x=np.linspace(0,fmax-1,f)
        y=np.linspace(0,fmax-1,f)
        g=scipy.interpolate.RectBivariateSpline(y,x,randz)
        intmat=g(coord,coord)*a
        total=(intmat)+total
        imatrix[:,:,i-2]=intmat
        
    return total, imatrix

