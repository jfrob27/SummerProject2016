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

    Contact GitHub API Training Shop Blog About 

