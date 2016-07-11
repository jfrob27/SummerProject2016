

import numpy as np
import wavelet_transforms as wts
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



#extra definitions for extraction##
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

            

            #------Extraction-----------
            if j == 11:
                phaseo[:,:,i]=np.arctan2(W1.imag,W1.real)
                amplio[:,:,i]=abs(W1)
            q=2.65
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
                #if j == 4:
                #    print treshp, tresh
            tresh=treshp
            cohe= np.where(module > tresh)
        
            if (module[cohe].shape[0] > 0):

                Wcp[cohe]=W1[cohe]
                W1c[:,:,j,i] = W1c[:,:,j,i]+Wcp
                S1c[:,:,j] = S1c[:,:,j] + (np.conj(Wcp)*Wcp)
                Sdc[i,j]=np.mean((np.conj(Wcp)*Wcp))

                Wcp=Wcp*0
            noncohe =np.where(module <= tresh)
            #if j == 11:
             #   print Wnp.shape
            if (module[noncohe].shape[0] >  0):
                Wnp[noncohe]=W1[noncohe]
                W1n[:,:,j] = W1n[:,:,j]+ Wnp
                S1n[:,:,j]= S1n[:,:,j] +(np.conj(Wnp)*Wnp)
                Sdn[i,j]=np.mean((np.conj(Wcp)*Wcp))
                Wnp=Wnp*0
        

        S1a[j]=np.mean(S1[:,:,j])*delta/ float(N)
        S1ac[j]=np.sum(S1c[:,:,j])*delta/(N*na*nb)
    #row_sums = wt.sum(axis=1)
   # wt = wt / row_sums[:, np.newaxis]
    return wt, W1n, W1c, tab_k, S1ac, S1a


def powerlawmod(wt, wtC, tab_k,  wherestart, slope, S1ac, S1a):
    #S1a=np.mean(abs(wt)**2., axis=(0,1))
    Wc=np.sum(wtC[:], axis=3)
    print Wc.shape
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
        #wtfori=wtfori.astype(complex)

        difference = slope * ( x[i] - x[wherestart] ) - powernew[i] + power[wherestart]

        constant= ( (-2*np.mean(wtfori)) - np.sqrt(4*((np.mean(wtfori)**2.))-4*np.mean(wtfori**2.)*(1-np.exp(difference))))/(2)
        #print constant
        #wtmod[:,:,i]=wtfori+constant
        for j in range(awt.shape[3]):
            wtmod[:,:,i,j]=+constant*(np.sum(awt[:,:,i,j].flatten())/np.sum(wtfori.flatten()))
            a=np.sum(awt[:,:,i,j])/np.sum(wtfori)
            test=test+a
            ctest=constant*a+ctest
            wtmod[:,:,i,j]=awt[:,:,i,j]+constant*a
            #print a
        #print 'sum = ', ctest
    wtmod=np.sum(wtmod[:], axis=3)
    return wtmod

import fbm2d as fbm
import matplotlib.pyplot as plt
import perlin2d

#----Main Function Used to Plot and Test Modified Power Law of non-gaussian part


def function():
    plt.close('all')
    image=fbm.fbm2d(-3.2,256,256)

    #perlin,matrix=perlin2d.perlin2d(8, .7)
    #image=np.sum(abs(matrix), axis=2)
    #image=image-image.min()

    image=np.exp(image)
    image=image-image.min()
    #norm=np.linalg.norm(image)
    #image=image/norm
    plt.figure('original image')
    plt.imshow(image.real, interpolation= 'none', cmap='Greys_r')
    


    wt,Wn,Wc, tab_k, S1ac, S1a= fan_transform(image)
    wCmod=powerlawmod(abs(wt), Wc, tab_k, 14, -2.5, S1ac, S1a)
    Wc=np.sum(Wc[:], axis=3)
    print Wc.shape, Wc[7]
    phase=(np.arctan2((Wn.imag+Wc.imag),(Wn.real+Wc.real)))
    cphase=np.arctan2(Wc.imag,Wc.real)
    nphase=np.arctan2(Wn.imag,Wn.real)
    wCmod[np.isnan(wCmod)]=0
    #wCmod=abs(wCmod*np.cos(cphase)+wCmod*np.sin(cphase))
    modcopy=abs(wCmod.copy())
    modcopy=modcopy*np.cos(cphase)
    wCmod=abs(wCmod)
    wtnew=abs(Wn.copy())
    #wtnew=wtnew*np.cos(nphase)
    #wCmod=wCmod*np.cos(cphase)+wCmod*np.sin(cphase)
    

    wtnew[:]=wtnew[:]+wCmod[:]
    #wtnew[:]=np.sqrt(wtnew[:]**2.+wCmod[:]**2.)
    #wtnew=abs(wtnew)
    wtnew = wtnew*np.cos(phase)
    

    rec_image=wts.halo_inverse(wtnew,tab_k)
    plt.figure('rec_image')
    plt.imshow(rec_image.real, interpolation= 'none' , cmap= 'Greys_r')
    plt.colorbar()

    Wcmod_rec=wts.halo_inverse(modcopy, tab_k)
    plt.figure('wcmod')
    plt.imshow(Wcmod_rec.real, interpolation= 'none', cmap = 'Greys_r')
    plt.colorbar()
    plt.show()


    Wc_rec=wts.halo_inverse(Wc, tab_k)
    plt.figure('wc')
    plt.imshow(Wc_rec.real, interpolation= 'none', cmap = 'Greys_r')
    plt.colorbar()
    plt.show()

    Wn_rec=wts.halo_inverse(Wn,tab_k)
    plt.figure('wn')
    plt.imshow(Wn_rec.real, interpolation= 'none', cmap = 'Greys_r')
    plt.colorbar()
    plt.show()
   # for i in range(5):
    #    plt.figure(5+i)
     #   print 'i'
      #  plt.imshow(Wc.real[:,:,i+11], interpolation = 'none', cmap= 'Greys_r')
       # plt.show()


#powerlaws#
    plt.figure()
    plt.plot(np.log(tab_k), np.log(S1ac), 'ro', label= 'S1ac')
    plt.plot(np.log(tab_k), np.log(np.mean(((abs(modcopy))**2.), axis=(0,1))),             'go', label= 'Wcmod')
    plt.plot(np.log(tab_k), np.log(S1a), 'bo', label= 'S1a')
    plt.plot(np.log(tab_k), np.log(np.mean(((abs(wt))**2.), axis=(0,1))),                 'r-', label= 'wt')
    wt,tab_k,s1a=wts.fan_transform(rec_image)
    #plt.figure(1)
    plt.plot(np.log(tab_k), np.log(np.mean(((abs(Wc))**2.), axis=(0,1))), 'b--', label= 'Wc')
    plt.plot(np.log(tab_k), np.log(np.mean(((abs(wt))**2.), axis=(0,1))), 'k-', label= 'newimage')
    plt.legend()
    #print wCmod.min()
    dy= np.log(np.mean(((abs(modcopy[11]))**2.), axis=(0,1)))- np.log(np.mean(((abs(modcopy[12]))**2.),axis=(0,1)))
    dx=np.log(tab_k[10])-np.log(tab_k[11])
    print dy/dx

#analyze image
    plt.figure('hist')
    plt.hist(rec_image.real.flatten(), bins=100)
    plt.show()
    plt.figure('imagehist')
    plt.hist(image.real.flatten(), bins=100)
    plt.show()
    print 'mod max, min, mean = ' , Wcmod_rec.max(), Wcmod_rec.min(),Wcmod_rec.mean()

    #wt,Wn,Wc, tab_k, S1ac, S1a= fan_transform(rec_image)

    
