import numpy as np
from scipy import scipy.interpolate

__all__=["fbm2d", "perlin2d", "powerlawmod"]

# Defines functions used for creating and modifying fractal noise that simulates
# ISM behavior. Currently (7/13/16) all are 2d only.
# includes fbm(GRF), perlin, and modification of powerlaw 



##########################################################################



def fbm2d(exp ,nx ,ny):
    '''
    Generates an image using a power spectrum with slope 'exp' of size nx, ny
    Returns an image as np array
    '''

#--------------definitions--------------#
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
 
#---------------phase------------------#  


    #phase=np.random.uniform(-np.pi,np.pi,size=(nx,ny))
    #---Setting Phase Using Array Projection---#
    #if odd_y == 1:
    #   phase[1:,1:][nx_half:,:ny][::-1,::-1] = -phase[1:,1:][:nx_half,:ny]
    #   phase[(nx_half+1.):,:ny][::-1,::-1] = -phase[:nx_half,:ny]
    #else:
    #   phase[1:,1:][(nx_half):,:ny][::-1,::-1] = -phase[1:,1:][:nx_half-1,:ny]
    #  phase[1:,1:][(ny_half-1),:(nx_half-1)][::-1]= -phase[1:,1:][(ny_half-1),
    #    nx_half:]
    #phase[nx_half:,:ny][::-1,::-1] = -phase[:nx_half,:ny]

    #method like in IDL
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
    #phase= np.roll(phase, int(nx_half+odd_x), axis=1)
    #phase= np.roll(phase, int(ny_half+odd_y-1), axis=0)


    phase = np.fft.ifftshift(phase)

#-----------------k matrix-----------------------#
    xmap = np.zeros((nx,ny))
    ymap = np.zeros((nx,ny))
    for i in range(int(nx)):
        xmap[i,:]=(i-nx_half)/nx
    for i in range(int(ny)):
        ymap[:,i]=(i-ny_half)/ny
    kmat= np.sqrt(xmap**2. + ymap ** 2.)
    kmat[nx_half,ny_half]=1.

#------------------amplititude-------------------#

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
    

    return image



##############################################################################



def perlin2d(n,p):
    '''
    image size n x n with 'perturbance' "p"
    '''
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


###############################################################################



def powerlawmod(wt, wtC, tab_k,  wherestart, slope, S1ac, S1a, wtN):   
   
    '''
   ##function used to modify power law of non-gaussian part of image. wt is
   ## original image wavelet transform. wtC is coherent part of wavelet 
   ##transform. where start is the point that will be the interesection of 
   ##the two powerlaws. Modificiation is done by multiplication 
   ## by a constant
   Returns Modified wavelets of coherent part, wtCmod.
   '''



#-----------Definitions and Everything as Magnitudes--------#    
    Wc=np.sum(wtC[:], axis=3)
    Wc=abs(Wc)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    wtmod=abs(Wc.copy())
    #wNmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)
    power=np.log(np.mean((abs(wt)**2.), axis=(0,1)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(0,1)))
    end=wtmod.shape[2]


#Power of wavelets corresponding to slope input calculated#
    for i in range(end):

        wtfori=abs(Wc[:,:,i])

        difference = slope * ( x[i] - x[wherestart] )
             - powernew[i] + power[wherestart]
        constant= np.sqrt(np.exp(difference))

        wtmod[:,:,i]=wtfori*constant
                
    return wtmod




######################################################################



#seems not useful#
def powerlawmod2(wt, wtC, tab_k,  wherestart, slope, S1ac, S1a,):   


    '''
    ##function used to modify power law of non-gaussian part of image. wt is
    ## original image wavelet transform. wtC is coherent part of wavelet 
    ##transform. where start is the point that will be the interesection of 
    ##the two powerlaws. Modificiation is done by addition 
    ## of a constant
    Returns Modified wavelets of coherent part, wtCmod.
    '''

    #S1a=np.mean(abs(wt)**2., axis=(0,1))
    
    Wc=np.sum(wtC[:], axis=3)
    #S1ac=np.mean(abs(Wc)**2., axis=(0,1))
    Wc=abs(Wc)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2], wtC.shape[3]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)
    #power=np.log(S1a)

    power=np.log(np.mean((abs(wt)**2.), axis=(0,1)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(0,1)))
    #powernew=np.log(S1ac)
    end=wtmod.shape[2]
    #wtC=np.sum(wtC[:], axis =3)
    for i in range(wherestart-4, end):
        test=0
        ctest=0
        wtfori=abs(Wc[:,:,i])

        difference = slope * ( x[i] - x[wherestart] ) - powernew[i] +
                  power[wherestart]

        constant= ( (-2*np.mean(wtfori)) - np.sqrt(4*((np.mean(wtfori)**2.))-
                  4*np.mean(wtfori**2.)*(1-np.exp(difference))))/(2)
        
        #wtmod[:,:,i]=wtfori+constant
        for j in range(awt.shape[3]):
            #wtmod[:,:,i,j]=wtC[:,:,i,j]+constant*(np.sum(awt[:,:,i,j].flatten()
            #)/np.sum(wtfori.flatten()))
            a=np.sum(awt[:,:,i,j])/np.sum(wtfori)
            test=test+a
            ctest=constant*a+ctest
            wtmod[:,:,i,j]=awt[:,:,i,j]+constant*a
        
   
    wtmod=np.sum(wtmod[:], axis=3)
    return wtmod
