# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:35:20 2022

@author: shuaibin
"""

"Generating E field of sample arm of HoloOCT"
import numpy as np
import random as rand
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy import fftpack
import time

def GaussWaveFront(mu,thgma,d):
    return 1/thgma**2/np.pi*np.exp(-(d-mu)**2/thgma**2)

# sum=0
# thegma=100
# for x in np.arange(-2*thegma,2*thegma,thegma/200):
#     for y in np.arange(-2*thegma,2*thegma,thegma/200):
#         sum=sum+GaussWaveFront(0, thegma, np.sqrt(x**2+y**2))*(thegma/200)**2
# print('integral of the function is: ',sum)

    
def SpectrumGauss(mu,thgma,d):
    return 1/thgma/np.sqrt(np.pi)*np.exp(-(d-mu)**2/thgma**2)

image_step_size=2 # um
phantom_step_size=1 # um

def GenPhantom(phantomSize,max_distance,thresh_hold):
    ################# generating phantom
    phantom=np.zeros((phantomSize,phantomSize),dtype=int) # pixel step size = 1um
    dotii=[]
    dotjj=[]
    ndot=0
    # phantom[int(phantomSize/2),int(phantomSize/2)]=-2000
    # dotii=[int(phantomSize/2)]
    # dotjj=[int(phantomSize/2)]
    # ndot=1

    # for ii in range(32,96,10):
    #     phantom[ii,32]=(ii-32)*40
    #     phantom[ii,96]=(ii-32)*40
    #     dotii=dotii+[ii,ii]
    #     dotjj=dotjj+[32,96]
    #     ndot+=2
    
    for ii in np.arange(phantomSize):
        for jj in np.arange(phantomSize):
            if rand.randint(0,1/thresh_hold)==1/thresh_hold:
                phantom[ii,jj]=rand.randint(-max_distance,max_distance) # defocus distance, unit um
                dotii=dotii+[ii]
                dotjj=dotjj+[jj]
                ndot=ndot+1
    return [phantom,dotii,dotjj,ndot]


def GenSampleE(data=[],NA=0.01,wavelength=1.3,imageSize=256):
    ################### calculating E field
    # wavelength=1.3 # um
    # NA=0.13
    if len(data)!=4:
        print('phantom data incorrect')
    else:
        phantom=data[0]
        dotii=data[1]
        dotjj=data[2]
        ndot=data[3]
        SEfield=np.zeros((imageSize,imageSize),dtype=np.complex128) # pixel step size = 0.1um
        size_diff=round((imageSize*image_step_size-phantom.shape[0]*phantom_step_size)/2)
        for dot in range(ndot):
            distance=phantom[dotii[dot],dotjj[dot]] #um
            thgma=np.abs(round(distance*NA/np.sqrt((1-NA**2)))) #um
            hStart=max(0,round((dotii[dot]*phantom_step_size+size_diff-2*thgma)/image_step_size))
            hStop=min(imageSize,round((dotii[dot]*phantom_step_size+size_diff+2*thgma)/image_step_size))
            vStart=max(0,round((dotjj[dot]*phantom_step_size+size_diff-2*thgma)/image_step_size))
            vStop=min(imageSize,round((dotjj[dot]*phantom_step_size+size_diff+2*thgma)/image_step_size))
            
            for ii in range(hStart,hStop):
                for jj in range(vStart,vStop):
                    d=np.sqrt((ii*image_step_size-dotii[dot]*phantom_step_size-size_diff)**2+
                              (jj*image_step_size-dotjj[dot]*phantom_step_size-size_diff)**2)
                    if not d>2*thgma:
                        SEfield[ii,jj]=SEfield[ii,jj]+GaussWaveFront(0, thgma, d)*np.exp(
                                        -1j*np.sign(distance)*np.sqrt(d**2+distance**2)/wavelength*2*np.pi)*np.exp(
                                            -1j*2*np.pi/wavelength*distance)
        return SEfield

def InterfereSR(SEfield=[],RTiltH=0,RTiltV=0,wavelength=1.3): # RtiltH, RtiltV are reference beam tilt angle in 180 degree unit
    if SEfield.shape[0]==0:
        print('Sample E field wrong dimension')
    else:
        imageSize=SEfield.shape[0]
        InterfereField=np.zeros((imageSize,imageSize),dtype=np.float32)
        for ii in np.arange(imageSize):
            for jj in np.arange(imageSize):
                # if np.abs(SEfield[ii,jj])<0.00000001:
                #     InterfereField[ii,jj]=1
                # else:
                    InterfereField[ii,jj]=1+np.abs(SEfield[ii,jj])**2+2*np.abs(SEfield[ii,jj])*np.cos(np.angle(SEfield[ii,jj])-
                                          (ii*image_step_size*np.sin(RTiltH/180*np.pi)/wavelength*2*np.pi+jj*image_step_size*np.sin(RTiltV/180*np.pi)/wavelength*2*np.pi))
        
        #np.abs(SEfield[ii,jj])**2+
        return InterfereField
    

def RemoveTilt(I,wavelength,RTiltH,RTiltV):
    imageSize=I.shape[0]
    FI=fftpack.fft2(I)
    FI=np.fft.fftshift(FI)
    # plt.figure()
    # plt.imshow(np.abs(FI))
    # plt.clim(0,0.05)
    # plt.title('fft of interference pattern')
    # # spatial filtering complex conjugate 
    for ii in range(imageSize-12):
        FI[imageSize-ii-12:,ii]=0
    FI[:,imageSize-12:]=0
    
    iFFI=fftpack.ifft2(FI)
    # FIFFI=fftpack.fft2(IFFI)
    # plt.figure()
    # plt.imshow(np.abs(FIFFI))
    # plt.title('interference pattern filtered')
    
    # remove the tilting of fringes
    TiFFI=np.zeros(I.shape,dtype=np.complex64)
    for ii in np.arange(imageSize):
        for jj in np.arange(imageSize):
            TiFFI[ii,jj]=iFFI[ii,jj]*np.exp(1j*(ii*image_step_size*np.sin(RTiltH/180*np.pi)/wavelength*2*np.pi+
                                                jj*image_step_size*np.sin(RTiltV/180*np.pi)/wavelength*2*np.pi))
            
    return TiFFI

def Windowing(TiFFI,N):
    spectrum=np.zeros(N,dtype=np.float32)
    for ii in range(N):
        spectrum[ii]=SpectrumGauss(N/2,N/4,ii)
    spectrum=spectrum/np.max(spectrum)
    
    for ii in range(N):
        TiFFI[:,:,ii]=TiFFI[:,:,ii]*spectrum[ii]
    return TiFFI

def RemoveDC(I):
    imageSize=I.shape[0]
    FI=fftpack.fft2(I)
    FI=np.fft.fftshift(FI)
    # plt.figure()
    # plt.imshow(np.abs(FI))
    # plt.clim(0,0.05)
    # plt.title('fft of interference pattern')
    # # spatial filtering complex conjugate 
    # for ii in range(imageSize):
    #     FI[imageSize-ii-12:,ii]=0
    FI[int(imageSize/2)-2:int(imageSize/2)+2,int(imageSize/2)-2:int(imageSize/2)+2]=0
    iFFI=fftpack.ifft2(FI)
    # FIFFI=fftpack.fft2(IFFI)
    # plt.figure()
    # plt.imshow(np.abs(FIFFI))
    # plt.title('interference pattern filtered')
    
    # remove the tilting of fringes
    # TiFFI=np.zeros(I.shape,dtype=np.complex64)
    # for ii in np.arange(imageSize):
    #     for jj in np.arange(imageSize):
    #         TiFFI[ii,jj]=iFFI[ii,jj]*np.exp(1j*(ii*np.sin(RTiltH/180*np.pi)/wavelength*2*np.pi+
    #                                                  jj*np.sin(RTiltV/180*np.pi)/wavelength*2*np.pi))
    return iFFI

def PropagateHolo(TiFFI,wavelength,Z):
    global pixel_step_inverse
    imageSize=TiFFI.shape[0]
    FTIFFI=fftpack.fft2(TiFFI)
    # FTIFFI=np.fft.fftshift(FTIFFI)
    # plt.figure()
    # plt.imshow(np.angle(FTIFFI))
    # # plt.clim(0,0.05)
    # plt.title('angle fft of interference pattern after tilter after filter')
    
    PFI=np.zeros(TiFFI.shape,dtype=np.complex64)
    dP=1/image_step_size/2/(imageSize/2)
    # PPG=np.zeros(FI.shape,dtype=np.complex64)
    for ii in range(imageSize):
        for jj in range(imageSize):
            # PPG[ii,jj]=np.exp(-1j*2*np.pi*np.sqrt((1/1.3)**2-(dP*(ii-512))**2-(dP*(jj-512))**2)*4000)
            PFI[ii,jj]=FTIFFI[ii,jj]*np.exp(1j*2*np.pi*np.sqrt((1/wavelength)**2-(dP*(ii-imageSize/2))**2-
                                    (dP*(jj-imageSize/2))**2)*Z)*np.exp(-1j*2*np.pi/wavelength*Z)
     
    
    # FPFI=np.zeros(FI.shape,dtype=np.complex64)
    # for ii in range(1024):
    #     for jj in range(1024):
    #         if 90<np.sqrt((ii-512)**2+(jj-512)**2)<100:
    #             FPFI[ii,jj]=PFI[ii,jj]
    
    iFPFI=fftpack.ifft2(PFI)

    return iFPFI

def ReconMAT(imageSize=400,lamd1=1.275,lamdn=1.325,N=512):
    dP = 1/image_step_size/2/(imageSize/2)
    RECON=np.zeros((imageSize,imageSize,N),dtype=np.complex64)
    wavenumber=np.arange(1/lamdn,1/lamd1,(1/lamd1-1/lamdn)/N)
    for kk,wave in enumerate(wavenumber):
        # print('processing wavelength '+str(kk))
        for ii in range(imageSize):
            for jj in range(imageSize):
                RECON[ii,jj,kk]=2*np.pi*np.sqrt((wave)**2-(dP*(ii-imageSize/2))**2 -(dP*(jj-imageSize/2))**2)+2*np.pi*wave
    return RECON

def ReconHolo(TiFFI,RECON,zmax,dz):
    imageSize=TiFFI.shape[0]
    start=time.time()
    FTiFFI=np.zeros(TiFFI.shape,dtype=np.complex64)
    for kk in range(TiFFI.shape[2]):
        FTiFFI[:,:,kk]=fftpack.fft2(TiFFI[:,:,kk])
    print('time for FFT: '+str((time.time()-start)/60)+'min')

    vol=np.zeros((imageSize,imageSize,int(2*zmax/dz)),dtype=np.float32)
    z=0
    for Z in np.arange(-zmax,zmax,dz):
        # print('processing depth: '+str(Z))
        start=time.time()
        tmp3=np.multiply(FTiFFI,np.exp(1j*RECON*Z))
        tmp2=np.sum(tmp3,2)
        print('time elapsed for multiply: '+str((time.time()-start))+'s')
        start=time.time()
        vol[:,:,z]=np.abs(fftpack.ifft2(tmp2))
        z=z+1
        print('time elapsed for fft: '+str((time.time()-start))+'s')
    return vol

# def ReconHoloFFT(TiFFI,RECON,zmax,dz):
#     imageSize=TiFFI.shape[0]
#     start=time.time()
#     FTiFFI=np.zeros(TiFFI.shape,dtype=np.complex64)
#     for kk in range(TiFFI.shape[2]):
#         FTiFFI[:,:,kk]=fftpack.fft2(TiFFI[:,:,kk])
#     print('time for FFT: '+str((time.time()-start)/60)+'min')
    
#     dP = 1/image_step_size/2/(imageSize/2)
#     wavenumber=np.arange(1/1.325,1/1.275,(1/1.275-1/1.325)/512)
#     for kk,wave in enumerate(wavenumber):
#         # print('processing wavelength '+str(kk))
#         for ii in range(imageSize):
#             for jj in range(imageSize):
#                 FTiFFI[ii,jj,kk]=FTiFFI[ii,jj,kk]*(0.5-())


#     for Z in np.arange(-zmax,zmax,dz):
#         # print('processing depth: '+str(Z))
#         start=time.time()
#         tmp3=np.multiply(FTiFFI,np.exp(1j*RECON*Z))
#         tmp2=np.sum(tmp3,2)
#         print('time elapsed for multiply: '+str((time.time()-start))+'s')
#         start=time.time()
#         vol[:,:,z]=np.abs(fftpack.ifft2(tmp2))
#         z=z+1
#         print('time elapsed for fft: '+str((time.time()-start))+'s')
#     return vol

if __name__ == "__main__":
    # data=GenPhantom(512,3000,1/2000)
    # phantom=data[0]
    # dotii=data[1]
    # dotjj=data[2]
    # ndot=data[3]
    
    # plt.figure()        
    # plt.imshow(phantom)
    # plt.colorbar()
    # plt.clim(-1500,1500)
    # plt.title('phantom')

    # np.save("phantom",phantom,allow_pickle=True)
    # np.save("dotii",dotii,allow_pickle=True)
    # np.save("dotjj",dotjj,allow_pickle=True)

    # phantom=np.load("phantom.npy",allow_pickle=True)
    # dotii=np.load("dotii.npy",allow_pickle=True)
    # dotjj=np.load("dotjj.npy",allow_pickle=True)
    # ndot=len(dotjj)

    # data=[phantom,dotii,dotjj,ndot]
    # wavenumber=np.arange(1/1.325,1/1.275,(1/1.275-1/1.325)/512)
    # njobs=8
    # section=int(512/njobs)
    # id=1#int(id)
    # startwave=(id-1)*section
    # stopwave=id*section
    
    # TiFFI=np.zeros((400,400,section),dtype=np.complex64)
    # # for jobs in range(4):
    # #     TiFFI2=np.load("TiFFI"+str(jobs+1)+".npy",allow_pickle=True)
    # #     TiFFI[:,:,jobs*128:jobs*128+128]=TiFFI2
    
    # # TiFFI=np.load("TiFFI.npy",allow_pickle=True)
    # # iFPFI=np.zeros((400,400,section),dtype=np.complex64)
    # start=time.time()

    # for ii,kk in enumerate(wavenumber[startwave:stopwave]):
    #     print('generating wavenumber '+str(ii)+' time elapsed: '+str((time.time()-start)/60)+'min')
    #     SEfield=GenSampleE(data,0.07,1/kk,400)
    #     # plt.figure()
    #     # plt.imshow(np.angle(SEfield))
    #     # plt.colorbar()
    #     # plt.clim(-np.pi,np.pi)
    #     # plt.title('E field phase')
    #     # plt.figure()
    #     # plt.imshow(np.abs(SEfield))
    #     # plt.colorbar()
    #     # # plt.clim(0,0.0002)
    #     # plt.title('E field intensity')
        
    #     I=InterfereSR(SEfield,7,7,1/kk)
    #     TiFFI[:,:,ii]=RemoveTilt(I,1/kk,7,7)
    # np.save("TiFFI"+str(id),TiFFI,allow_pickle=True)
    
    TiFFI=np.zeros((400,400,512),dtype=np.complex64)
    for jobs in range(8):
        TiFFI2=np.load("TiFFI"+str(jobs+1)+".npy",allow_pickle=True)
        TiFFI[:,:,jobs*64:jobs*64+64]=TiFFI2
    start=time.time()
    TiFFI=Windowing(TiFFI,512)
    print('time for windowing: '+str(time.time()-start)+'s')
    start=time.time()
    RECON=ReconMAT(400,1.275,1.325,512)
    print('time for Recon matrix: '+str(time.time()-start)+'s')
    vol=ReconHolo(TiFFI,RECON,16,8)
    np.save("vol",vol,allow_pickle=True)
    
    image=np.max(vol,0)
    plt.figure()
    plt.imshow(np.abs(image.T),extent=[0,800,-3000,3000])
    image=np.max(vol,2)
    plt.figure()
    plt.imshow(np.abs(image),extent=[0,800,0,800],vmax=0.15)
      
    ims = []
    fig ,ax= plt.subplots()
    for ii in range(300):
        im = ax.imshow(np.abs(vol[ii+50,:,:]), animated=True,vmin=0,vmax=0.1,extent=[-3000,3000,0,1600])
        title=ax.text(0,1650,' Y '+str(ii))
        ims.append([im,title])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=500)