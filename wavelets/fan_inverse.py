import numpy as np

def fan_inverse(wt, tab_k):
    ko=5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko
    na=float(wt.shape[0])
    nb=float(wt.shape[1])
    
   # surf = na*nb
   # real=0
   # imaginary=0

    M=tab_k.shape[0]
    interval=[tab_k[0],tab_k[M-1]]  #Default if no interval input#

    ########################################

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
        shifty= (na-1.)/2+1

########################################
    Cphi=0.114517

    a=ko/(tab_k)

    delta_a= np.exp(delta)

    N=int((2*np.pi)/delta)

    imagetot=np.complex_(np.zeros((na,nb)))
    #uv_sum=np.zeros((na,nb))
    phi=np.zeros((na,nb))
    arrange=np.arange(tab_k.shape[0])

    o=interval[0]
    l=interval[1]
    int1= np.where((tab_k >= o) & (tab_k <= l))
    int1=arrange[int1]
    count=int1.shape[0]

    for h in range(int1[0], int1[count-1]):
        for i in range(N):
            uv=0
            t=float(delta*i)
            #for j in range(int1[0], int1[count-1]+1):
            uv=np.exp(-0.5*( (a[h]*x - ko*np.cos(t))**2. + (a[h]*y - ko*np.sin(t))**2.))
            uv = uv / a[h]**2.
        

            imageFT=np.roll((np.fft.fft2(wt[:,:,h])),int(shiftx), axis=1)
            imageFT=np.roll(imageFT, int(shifty), axis=0)
            imageFT=imageFT*uv
            #imageFT=np.fft.fftshift(imageFT)
            imageFT=np.roll(imageFT ,int(shiftx), axis=1)
            imageFT=np.roll(imageFT, int(shifty), axis=0)

            ampli=abs(imageFT)
           # arrange=np.arange((imageFT.shape[0]*imageFT.shape[1])).reshape(imageFT.shape[0],imageFT.shape[1])
            phi=np.arctan2(imageFT.imag,imageFT.real)
            




            imageFT= ampli*np.cos(phi) + 1.j*ampli*np.sin(phi)

            image=np.fft.ifft2(imageFT)

            da=a[0]*((delta_a-1.)/(delta_a**(j+1.)))

            imagetot=imagetot+image * delta * da /Cphi
    h_rec=imagetot.real*.95
    return h_rec
