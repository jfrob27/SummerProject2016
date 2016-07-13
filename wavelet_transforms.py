import numpy as np
from scipy import scipy.signal

__all__ = ["atrou", "fan_transform", "fan_inverse", "halo_transform", "halo_inverse"]


#definition of atrou, fan, halo wavelet functions




#############################################################################



def atrou(data, scale):
    '''
    description
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


###########################################################################


def fan_transform(image, scale='scale'):
    '''
    Performs fan transform on 'image' input. If scale is not specified returns
    largest number of scales. Right now scale option is broken. 

    Returns wt, tab_k and S1a where:
    wt   is an image cube of wavelets of size image with depth number of scales.
    tab_k is scales
    S1a is powerlaw
    '''


#--------------------Definitions----------------------#
    ko= 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko
    na=float(image.shape[0])
    nb=float(image.shape[1])


#--------------Spectral Logarithm--------------------#
    M=int(np.log(nb)/delta)
    a2= np.zeros(M)
    a2[0]=np.log(nb)

    for i in range(M-1):
        a2[i+1]=a2[i]-delta

    a2=np.exp(a2)
    tab_k = 1. / (a2)
    if scale != 'scale':
        a2=np.log(a2)
        a=np.linspace(a2[0],a2[M-1],int(scale))
        a2=np.exp(a)
        tab_k=1./(a2)
        M=scale

#-----------------UVPlane--------------#
    x=np.arange(nb)
    y=np.arange(na)
    x,y=np.meshgrid(x,y)
    if (nb % 2) == 0:
        x=(1.*x - (nb)/2.)/nb
        shiftx = (nb)/2.
    else:
       x= (1.*x - (nb-1)/2.)/nb 
       shiftx=(nb-1.)/2.+1

    if (na % 2) == 0:
        y=(1.*y-(na/2.))/na
        shifty=(na)/2.
    else:
        y= (1.*y - (na-1)/2.)/ na
        shifty=(na-1.)/2.+1

    S1 = np.zeros((na,nb,M))
    wt= np.zeros((na,nb,M), dtype=complex)

    S1a=np.zeros(M)

    a= ko*a2
    N=int(np.pi/delta)



#----------------Fourier Domain------------------------#
    imageFT= np.fft.fft2(image)
    #imageFT=np.fft.fftshift(image)
    imageFT= np.roll(imageFT,int( shiftx), axis=1)
    imageFT= np.roll(imageFT,int(shifty), axis=0)

    for j in range(M):
        for i in range(N):
            uv=0.
            t=float(delta*i)

            uv=np.exp( -.5*((a[j]*x - ko*np.cos(t))**2. + (a[j]*y - ko*np.sin(t))**2.))

            uv=uv* a[j]


            W1FT=imageFT*uv
            W1F2=np.roll(W1FT,int(shiftx), axis =1)
            W1F2=np.roll(W1F2,int(shifty),axis=0)

            W1=np.fft.ifft2(W1F2)
            wt[:,:,j]=wt[:,:,j]+ W1
            S1[:,:,j]= S1[:,:,j] + abs(W1)**2.

        S1a[j]=np.mean(S1[:,:,j])*delta/ float(N)

    return wt, tab_k, S1a




def fan_inverse(wt, tab_k): 
#intend to add option for scale so image can be reconstructed at certain scales
    '''
    Performs the inverse fan wavelet transform on a set of wavelets wt
    scale tab_k. Returns an image of size wt.shape[0] , wt.shape[1]
    '''


    ko= 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko
    na = float(wt.shape[0])
    nb = float(wt.shape[1])

    M = tab_k.shape[0]
    interval = [tab_k[0],tab_k[M-1]]  #Default if no interval input#

#------------------------------------------------------------------

    x= np.arange(nb)
    y= np.arange(na)
    x,y = np.meshgrid(x,y)
    if (nb % 2) == 0:
        x = (1.*x - (nb)/2.)/nb
        shiftx = (nb)/2.
    else:
       x = (1.*x - (nb-1)/2.)/nb
       shiftx = (nb-1.)/2.+1

    if (na % 2) == 0:
        y = (1.*y-(na/2.))/na
        shifty = (na)/2.
    else:
        y= (1.*y - (na-1)/2.)/ na
        shifty= (na-1.)/2+1

#----------------------------------------------------------------
    Cphi = 0.114517

    a = ko/(tab_k)

    delta_a = np.exp(delta)

    N = int((2*np.pi)/delta)

    imagetot = np.complex_(np.zeros((na,nb)))
    
    phi = np.zeros((na,nb))
    arrange = np.arange(tab_k.shape[0])

    o = interval[0]
    l = interval[1]
    int1 = np.where((tab_k >= o) & (tab_k <= l))
    int1 = arrange[int1]
    count = int1.shape[0]

   #for h in range(int1[0], int1[count-1]): was originally so scale option could be used
    for h in range(tab_k.shape[0]):
        for i in range(N):
            uv=0
            t=float(delta*i)
            #for j in range(int1[0], int1[count-1]+1): for scale option
            uv=np.exp(-0.5*( (a[h]*x - ko*np.cos(t))**2. + (a[h]*y - ko*np.sin(t))**2.))
            uv = uv / a[h]**2.
        

            imageFT=np.roll((np.fft.fft2(wt[:,:,h])),int(shiftx), axis=1)
            imageFT=np.roll(imageFT, int(shifty), axis=0)
            imageFT=imageFT*uv

            imageFT=np.roll(imageFT ,int(shiftx), axis=1)
            imageFT=np.roll(imageFT, int(shifty), axis=0)

            ampli=abs(imageFT)
            phi=np.arctan2(imageFT.imag,imageFT.real)
            




            imageFT= ampli*np.cos(phi) + 1.j*ampli*np.sin(phi)
            j=0
            image=np.fft.ifft2(imageFT)

            da=a[0]*((delta_a-1.)/(delta_a**(h+1.)))

            imagetot=imagetot+image * delta * da /Cphi
    h_rec=imagetot.real*.95
    return h_rec



#########################################################################




def halo_transform(image):
    '''
    Performs halo wavelet transform on image.
    Returns wavelets wt as image cube
    '''

    na = image.shape[0]
    nb = image.shape[1]
    ko = 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko

    x = np.arange(nb)
    y = np.arange(na)
    x,y = np.meshgrid(x,y)
    if (nb % 2) == 0:
        x = (1.*x - (nb)/2.)/nb
        shiftx = (nb)/2.
    else:
       x = (1.*x - (nb-1)/2.)/nb 
       shiftx = (nb-1.)/2.+1

    if (na % 2) == 0:
        y=(1.*y-(na/2.))/na
        shifty=(na)/2.
    else:
        y = (1.*y - (na-1)/2.)/ na
        shifty=(na-1.)/2.+1

#--------------Spectral Logarithm--------------------#
    M = int(np.log(nb)/delta)
    a2 = np.zeros(M)
    a2[0] = np.log(nb)

    for i in range(M-1):
        a2[i+1] = a2[i]-delta

    a2 = np.exp(a2)
    tab_k = 1. / (a2)
    wt = np.complex_(np.zeros(((na,nb,M))))


    a = ko*a2

    imageFT = np.fft.fft2(image)
    imageFT= np.roll(imageFT,int( shiftx), axis=1)
    imageFT= np.roll(imageFT,int(shifty), axis=0)

    for j in range(M):
        uv = 0

        uv = np.exp(-0.5*((abs(a[j]*np.sqrt(x**2.+y**2.))**2.  - abs(ko))**2.))

        uv = uv * a[j]

        W1FT = imageFT*(uv)
        W1F2 = np.roll(W1FT,int(shiftx), axis =1)
        W1F2 = np.roll(W1F2,int(shifty),axis=0)
        W1=np.fft.ifft2(W1F2)
        wt[:,:,j]=wt[:,:,j]+ W1

    return wt, tab_k



######################################################################




def halo_inverse(wt, tab_k, multiscales=False):

    '''
    multiscales allows reconstruction when wavelets have features that
    are not necisacrliy at their original scales
    '''

#----------------definitions----------------#

    ko=5.336
    na=wt.shape[0]
    nb=wt.shape[1]
 
    a=ko/tab_k
    imagetot=np.complex_(np.zeros((na,nb)))


#--------------Coords---------------------------##

    x = np.arange( nb , dtype=float )
    y = np.arange( na , dtype=float )
    x , y = np.meshgrid( x, y )
    if (nb % 2) == 0:
        x = ( 1.*x - (nb)/2. )/ nb 
        shiftx = (nb)/2.
    else:
        x = (1.*x - (nb-1)/2.)/ nb
        shiftx = (nb-1.)/2.+1

    if (na % 2) == 0:
        y = (1.*y-(na/2.))/na
        shifty = (na)/2.
    else:
        y = (1.*y - (na-1)/2.)/ na
        shifty = (na-1.)/2+1

#---------------------transform----------------------#

    
        
    for h in range(tab_k.shape[0]):
        uv = 0
        if scale=True:
            for i in range(tab_k.shape[0]):
                uv = np.exp ( -0.5 * ( abs ( a[i] * np.sqrt ( x**2. + y**2. ) )- ko)**2.)
                uv = uv / a[i]

                imageFT = np.roll((np.fft.fft2(wt[:,:,h])),int(shiftx), axis=1)
                imageFT = np.roll(imageFT, int(shifty), axis=0)
                imageFT = imageFT*uv
        
                imageFT = np.roll(imageFT ,int(shiftx), axis=1)
                imageFT = np.roll(imageFT, int(shifty), axis=0)
                image = np.fft.ifft2(imageFT)            

                imagetot = imagetot+image 


        else:
            uv = np.exp ( -0.5 * ( abs ( a[h] * np.sqrt ( x**2. + y**2. ) )- ko)**2.)
            uv = uv / a[h]


            imageFT = np.roll((np.fft.fft2(wt[:,:,h])),int(shiftx), axis=1)
            imageFT = np.roll(imageFT, int(shifty), axis=0)
            imageFT = imageFT*uv
            
            imageFT = np.roll(imageFT ,int(shiftx), axis=1)
            imageFT = np.roll(imageFT, int(shifty), axis=0)
            
            image = np.fft.ifft2(imageFT)
            
            imagetot = imagetot+image 

    #imagetot=imagetot*0.95
    return imagetot



##############################################################################



def coherent_extraction(image, scale='scale'):

    '''
    Fan inverse Transform Followed by separation of coherent
    and non coherent parts of the wavelets according to Nguyen paper
    '''




#------------------Definitions---------------------------#
    ko= 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko
    na=float(image.shape[0])
    nb=float(image.shape[1])


#--------------Spectral Logarithm--------------------~
    M=int(np.log(nb)/delta)
    a2= np.zeros(M)
    a2[0]=np.log(nb)

    for i in range(M-1):
        a2[i+1]=a2[i]-delta

    a2=np.exp(a2)
    tab_k = 1. / (a2)
    if scale != 'scale':
        a2=np.log(a2)
        a=np.linspace(a2[0],a2[M-1],int(scale))
        a2=np.exp(a)
        tab_k=1./(a2)
        M=scale
    
#-----------------UVPlane--------------#
    x=np.arange(nb)
    y=np.arange(na)
    x,y=np.meshgrid(x,y)
    if (nb % 2) == 0:
        x=(1.*x - (nb)/2.)/nb
        shiftx = (nb)/2.
    else:
       x= (1.*x - (nb-1)/2.)/nb 
       shiftx=(nb-1.)/2.+1

    if (na % 2) == 0:
        y=(1.*y-(na/2.))/na
        shifty=(na)/2.
    else:
        y= (1.*y - (na-1)/2.)/ na
        shifty=(na-1.)/2.+1

    S1 = np.zeros((na,nb,M))
    wt= np.zeros((na,nb,M), dtype=complex)

    S1a=np.zeros(M)

    a= ko*a2
    N=int(np.pi/delta)



#--------extra definitions for extraction------------#
    phaseo=np.zeros((na,nb,N))
    amplio=np.zeros((na,nb,N))
    Sdn=np.zeros((N,M))
    Sdc=np.zeros((N,M))
    temoin=np.zeros((na,nb))
    module=np.zeros((na,nb,M))
    S1c=np.zeros((na,nb,M))
    S1n=np.zeros((na,nb,M))
    Wnp=np.zeros((na,nb), dtype=complex)
    W1a=np.zeros((na,nb, M), dtype=complex)
    W1n=np.zeros((na,nb,M), dtype=complex)
    Wcp=np.zeros((na,nb), dtype=complex)
    W1c=np.zeros((na,nb,M,N), dtype=complex)
    S1m=np.zeros(M)
    S1ac=np.zeros(M)
    S1ac=np.zeros(M)
    S1an=np.zeros(M)

#----------------Fourier Domain------------------------#
    
    imageFT= np.fft.fft2(image)
    imageFT= np.roll(imageFT,int( shiftx), axis=1)
    imageFT= np.roll(imageFT,int(shifty), axis=0)

    for j in range(M):
        for i in range(N):
            uv=0.
            t=float(delta*i)

            uv=np.exp( -.5*((a[j]*x - ko*np.cos(t))**2. + (a[j]*y - ko*np.sin(t))**2.))

            uv=uv* a[j]


            W1FT=imageFT*uv
            W1F2=np.roll(W1FT,int(shiftx), axis =1)
            W1F2=np.roll(W1F2,int(shifty),axis=0)
            W1=np.fft.ifft2(W1F2)
            wt[:,:,j]=wt[:,:,j]+ W1
            S1[:,:,j]= S1[:,:,j] + abs(W1)**2.

            
#------------------------Extraction--------------------------------------#

            q=2.8
            module=abs(W1)
            tresh=module.max()
            treshp=module.max()*2.
        
            while ((treshp-tresh) != 0):
                tresh=treshp
                temoin = temoin*0
                
                indx=np.where(module <= tresh)
                temoin=(module[indx])**2.
                Sigtresh=np.sum(temoin)/(temoin.shape[0])
                treshp = q *np.sqrt(Sigtresh)
                
            tresh=treshp
            cohe= np.where(module > tresh)
        
            if (module[cohe].shape[0] > 0):

                Wcp[cohe]=W1[cohe]
                W1c[:,:,j,i] = W1c[:,:,j,i]+Wcp
                S1c[:,:,j] = S1c[:,:,j] + (np.conj(Wcp)*Wcp)
                Sdc[i,j]=np.mean((np.conj(Wcp)*Wcp))

                Wcp=Wcp*0
            noncohe =np.where(module <= tresh)
            

            if (module[noncohe].shape[0] >  0):
                Wnp[noncohe]=W1[noncohe]
                W1n[:,:,j] = W1n[:,:,j]+ Wnp
                S1n[:,:,j]= S1n[:,:,j] +(np.conj(Wnp)*Wnp)
                Sdn[i,j]=np.mean((np.conj(Wcp)*Wcp))
                Wnp=Wnp*0
        

        S1a[j]=np.mean(S1[:,:,j])*delta/ float(N)
        S1ac[j]=np.sum(S1c[:,:,j])*delta/(N*na*nb)
    #row_sums = wt.sum(axis=1)
    #wt = wt / row_sums[:, np.newaxis]
    return wt, W1n, W1c, tab_k, S1ac, S1a
