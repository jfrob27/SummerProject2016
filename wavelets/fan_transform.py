import numpy as np
def fan_transform(image, scale='scale'):



####################Definitions#############################
    ko= 5.336
    delta = (2.*np.sqrt(-2.*np.log(.75)))/ko
    na=float(image.shape[0])
    nb=float(image.shape[1])


#--------------Spectral Logarithm--------------------
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
            #W1F2=np.fft.ifftshift(W1FT)
            W1=np.fft.ifft2(W1F2)
            wt[:,:,j]=wt[:,:,j]+ W1
            S1[:,:,j]= S1[:,:,j] + abs(W1)**2.

        S1a[j]=np.mean(S1[:,:,j])*delta/ float(N)
    #row_sums = wt.sum(axis=1)
   # wt = wt / row_sums[:, np.newaxis]
    return wt, tab_k, S1a


#def phasetest(wt, tab_k):
#    #wt=wt[:,:,:16]
 #   n=wt.shape[0]
#    wta=np.zeros((n,n,tab_k.shape[0]))
#    phasemat=np.zeros((n,n,tab_k.shape[0]))
#    phase=np.arctan(wt.imag/wt.real)
 #   wt=wt.real
 ##   wtplus=(-1/(np.exp(wt/.5)+1)+1)*wt
#  #  wnorm=np.linalg.norm(wtplus)
 #   wtplus=wtplus/wnorm
#

 #   wtminus=((1/(np.exp(wt/.5)+1))*wt)
   # wnorm=np.linalg.norm(wtminus)
  #  wtminus=wtminus/wnorm
    #combine=np.zeros((n,n,tab_k.shape[0]))


    #for i in range(6, tab_k.shape[0]):
     #   phasei=5*np.exp(-abs(phase[:,:,i]))
      #  wti =wtplus[:,:,i]
        #wnorm=np.linalg.norm(wti[:,:])
        #wti=wti[:,:]/wnorm
#        pnorm=np.linalg.norm(phasei[:,:])
 #       phasei=phasei/pnorm
  #      wta[:,:,i]=(wti)
   #     phasemat[:,:,i]=phasei
#
 #       combine[:,:,i]=wti*phasei
#
 #       wnorm=np.linalg.norm(wtminus[:,:,i])
  #      wtminus[:,:,i]=wtminus[:,:,i]/wnorm
#
 #       cnorm=np.linalg.norm(combine[:,:,i])
  #      combine[:,:,i]=combine[:,:,i]/cnorm
   #     combine[:,:,i]=(combine[:,:,i]+wtminus[:,:,i])

    #wtminus=((1/(np.exp(wt/.5)+1))*wt)
    #wnorm=np.linalg.norm(wtminus)
   # wtminus=wtminus/wnorm
   # cnorm=np.linalg.norm(combine)
    #combine=combine/cnorm
   # combine=combine+wtminus[:,:,10:]
   # wtnorm=np.linalg.norm(wt)
    #wt=wt/wtnorm
#    for i in range(wt.shape[2]):
 #       wti=wt[:,:,i]
  #      norm=np.linalg.norm(wti)
   #     wti=wti/norm
    #    wt[:,:,i]=wti
#    for i in range(combine.shape[2]):
 #       combine[:,:,i]=combine[:,:,i]

  #  return wt,combine ,wta, phasemat

