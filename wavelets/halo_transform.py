import numpy as np

def halo_transform(image):

    na=image.shape[0]
    nb=image.shape[1]
    ko= 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko

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

    #--------------Spectral Logarithm--------------------
    M=int(np.log(nb)/delta)
    a2= np.zeros(M)
    a2[0]=np.log(nb)

    for i in range(M-1):
        a2[i+1]=a2[i]-delta

    a2=np.exp(a2)
    tab_k = 1. / (a2)
    wt= np.complex_(np.zeros(((na,nb,M))))


    a= ko*a2

    imageFT= np.fft.fft2(image)
    #imageFT=np.fft.fftshift(image)
    imageFT= np.roll(imageFT,int( shiftx), axis=1)
    imageFT= np.roll(imageFT,int(shifty), axis=0)

    for j in range(M):
        uv = 0

        uv=np.exp(-0.5*((abs(a[j]*np.sqrt(x**2.+y**2.))**2.  - abs(ko))**2.))

        uv = uv * a[j]

        W1FT=imageFT*(uv)
        W1F2=np.roll(W1FT,int(shiftx), axis =1)
        W1F2=np.roll(W1F2,int(shifty),axis=0)
       #W1F2=np.fft.ifftshift(W1FT)
        W1=np.fft.ifft2(W1F2)
        wt[:,:,j]=wt[:,:,j]+ W1

    return wt, tab_k
