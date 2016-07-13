import numpy as np
import scipy
from scipy import signal

def atrou(data, scale):
    kernel_val = np.array([1./16, 1./8., 1./16., 1./8., 1./4., 1./8., 1./16., 1./8, 1./16])
    result=np.complex_(np.zeros((data.shape[0],data.shape[1],scale)))
    tempo = data
    
    for i in range(scale):
        n=int(2*(2**i)+1)
        #x=np.arange(-n/2+1,n/2+1)
        #x,y=np.meshgrid(x,x)
        #kernel=(3/(n*np.sqrt(2*np.pi))*np.exp(-(9*(x**2+y**2)/(2*n**2))))
        kernel = np.zeros(n*n)
        dx = (n-1)/2.
        lc = (n**2 - n)/2.
        indice = np.array([0, dx, 2*dx, lc, lc+dx, lc+2*dx, 2*lc, 2*lc+dx, 2*(lc+dx)] )
        for j in range(indice.shape[0]):
            kernel[indice[j]] = kernel_val[j]
        kernel=kernel.reshape(n,n)
        conv=scipy.signal.convolve2d(tempo, kernel, mode='same')
        #tempofft=(np.fft.fft2(tempo))
        #kernelfft=(np.fft.fft2(kernel))
        #conv=np.fft.ifft2((kernelfft*tempofft))
        result[:,:, i]=tempo - conv
        tempo = conv

    return result
