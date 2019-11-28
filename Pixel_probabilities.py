import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('tkagg')
from ligo.skymap import postprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import healpy as hp
from ligo.skymap.io import fits
from ligo.skymap import plot
from ligo.skymap import postprocess
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from matplotlib.pyplot import cm
import matplotlib

import os
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['GOTO_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['openmp'] = 'True'


nside=128
npix=hp.nside2npix(nside)
theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))
ra_pix = phi
de_pix = -theta + np.pi/2.0

deg2perpix = hp.nside2pixarea(nside, degrees=True)

#sector_data = np.loadtxt('Probabilities_1', skiprows=1)
#prob = np.loadtxt('Predictions_2048_GW150914_snr-20.txt')
#sector_id = sector_data[:,0]
#ra_min = sector_data[:,1]
#ra_max = sector_data[:,2]
#de_min = sector_data[:,3]
#e_max = sector_data[:,4]

ra_min = np.loadtxt('RA_min_2048.txt')
ra_max = np.loadtxt('RA_max_2048.txt')
de_min = np.loadtxt('Dec_min_2048.txt')
de_max = np.loadtxt('Dec_max_2048.txt')

for i in range(0,100):
    prob = np.loadtxt("Prediction_files/Preds_"+str(i)+".txt")
    p = np.zeros(npix)
    for j in range(npix):
        for k in range(len(prob)):
            if ra_pix[j]>ra_min[k] and ra_pix[j]<ra_max[k] and de_pix[j]>de_min[k] and de_pix[j]<de_max[k]:
                p[j] = prob[k]
    
    p = p/sum(p)
#    fits.writeto("Healpy_Predictions/Healpy_preds_"+str(i)+".fits", np.transpose(p))
    np.savetxt("Healpy_Predictions/Healpy_preds_"+str(i)+".txt", np.transpose(p))
    
    
