import numpy as np
import math
import matplotlib.pyplot as plt

def fbm2d(exp ,nx ,ny):

#definitions
    exp = float(exp)
    nx = float(nx)
    ny = float(ny)
    if ( nx % 2 ) != 0:
        nx_half = (nx-1.)/2.
        odd_x = 1.
    else:
        nx_half = nx/2.
        odd_x = 0.
    if ( ny % 2 ) != 0:
        ny_half = (ny-1.)/2.
        odd_y = 1.
    else:
        ny_half = ny/2.
        odd_y = 0.
 
#phase  
   # phase=np.random.uniform(-np.pi,np.pi,size=(nx,ny))
#Setting Phase Using Array Projection
    #if odd_y == 1:
     #   phase[1:,1:][nx_half:,:ny][::-1,::-1] = -phase[1:,1:][:nx_half,:ny]
        ##phase[(nx_half+1.):,:ny][::-1,::-1] = -phase[:nx_half,:ny]
    #else:
     #   phase[1:,1:][(nx_half):,:ny][::-1,::-1] = -phase[1:,1:][:nx_half-1,:ny]
      #  phase[1:,1:][(ny_half-1),:(nx_half-1)][::-1]= -phase[1:,1:][(ny_half-1),nx_half:]
#phase[nx_half:,:ny][::-1,::-1] = -phase[:nx_half,:ny]

#old method like in IDL
    phase = np.zeros((nx,ny))
    phase[:]=-599
    for j in range(int(ny)):
        j2 = 2*ny_half - j
        for i in range(int(nx)):
            i2=2*nx_half-i
            if phase [i,j] == -599:
                tempo = np.random.uniform(-np.pi,np.pi)
                phase[i,j]=tempo
                if (i2 < nx and j2 < ny): 
                    phase[i2,j2]= -1.*tempo
#    phase= np.roll(phase, int(nx_half+odd_x), axis=1)
#    phase= np.roll(phase, int(ny_half+odd_y-1), axis=0)


    phase = np.fft.ifftshift(phase)

#k matrix
    xmap = np.zeros((nx,ny))
    ymap = np.zeros((nx,ny))
    for i in range(int(nx)):
        xmap[i,:]=(i-nx_half)/nx
    for i in range(int(ny)):
        ymap[:,i]=(i-ny_half)/ny
    kmat= np.sqrt(xmap**2. + ymap ** 2.)
    kmat[nx_half,ny_half]=1.

#amplititude

    amplitude = kmat**(exp/2.)
    amplitude[nx_half,ny_half]=0.


    #amplitude = np.roll(amplitude, int(nx_half+odd_x), axis=1)
    #amplitude = np.roll(amplitude, int(ny_half+odd_y-1), axis=0)
    amplitude = np.fft.ifftshift(amplitude)
    
    imRE = amplitude * np.cos(phase)
    imIM = amplitude * np.sin(phase)
    imfft = 1.j*imIM+imRE
    image = np.fft.ifft2(imfft)

    image=(image.real)
    
    #image = image / np.std(image)
    return image
