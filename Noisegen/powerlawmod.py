import numpy as np

def powerlawmod(wt, wtC, tab_k,  wherestart, slope,):   


    '''
    function used to modify power law of non-gaussian part of image. wt is 
    original image wavelet transform. wtC is coherent part of wavelet 
    transform. where start is the point that will be the interesection of 
    the two powerlaws. Modificiation is done by multiplication 
    by a constant
    Returns Modified wavelets of coherent part, wtCmod.
    '''
    
    
#-----------Definitions and Everything as Magnitudes--------#     
    Wc=abs(wtC)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)

    power=np.log(np.mean((abs(wt)**2.), axis=(0,1)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(0,1)))

    end=wtmod.shape[2]
   
   
   
   
#Power of wavelets corresponding to slope input calculated#
    for i in range(int(wherestart-6),end):
        test=0
        ctest=0
        wtfori=abs(Wc[:,:,i])
       

        difference = slope * ( x[i] - x[wherestart] ) - powernew[i] + power[wherestart]

        constant= np.sqrt(np.exp(difference))
        
        wtmod[:,:,i]=wtfori*constant
     
    return wtmod




#################################################################################




def interceptmod(wt, wtC, tab_k,  incr,):   
    '''
    
    '''
    
    Wc=abs(wtC)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)

    power=np.log(np.mean((abs(wt)**2.), axis=(0,1)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(0,1)))

    end=wtmod.shape[2]
   
    for i in range(end):
        wtfori=abs(Wc[:,:,i])
       

        difference = incr

        constant= np.sqrt(np.exp(difference))
        
        wtmod[:,:,i]=wtfori*constant
     
    return wtmod
