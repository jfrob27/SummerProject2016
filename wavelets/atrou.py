import numpy as np
import scipy
from scipy import signal

def atrou(data, scale):
    '''
    '''
    kernel_val = np.array([1./16, 1./8., 1./16., 1./8., 1./4., 1./8., 1./16., 1./8, 1./16])
    result=np.complex_(np.zeros((data.shape[0],data.shape[1],scale)))
    tempo = data
    
    for i in range(scale):
        n=int(2*(2**i)+1)
        kernel = np.zeros(n*n)
        dx = (n-1)/2.
        lc = (n**2 - n)/2.
        indice = np.array([0, dx, 2*dx, lc, lc+dx, lc+2*dx, 2*lc, 2*lc+dx, 2*(lc+dx)] )
        for j in range(indice.shape[0]):
            kernel[indice[j]] = kernel_val[j]
        kernel=kernel.reshape(n,n)
        conv=scipy.signal.convolve2d(tempo, kernel, mode='same')
        result[:,:, i]=tempo - conv
        tempo = conv

    return result
