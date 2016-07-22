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
