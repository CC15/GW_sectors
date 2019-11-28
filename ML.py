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
from matplotlib.pyplot import cm
import matplotlib


nside=128
npix=hp.nside2npix(nside)
theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))
ra_pix = phi
de_pix = -theta + np.pi/2.0

deg2perpix = hp.nside2pixarea(nside, degrees=True)

#sector_data = np.loadtxt('Probabilities_1', skiprows=1)
prob = np.loadtxt('Predictions_GW150914_pure_signal.txt')
#sector_id = sector_data[:,0]
#ra_min = sector_data[:,1]
#ra_max = sector_data[:,2]
#de_min = sector_data[:,3]
#e_max = sector_data[:,4]

ra_min = np.loadtxt('RA_min_2048.txt')
ra_max = np.loadtxt('RA_max_2048.txt')
de_min = np.loadtxt('Dec_min_2048.txt')
de_max = np.loadtxt('Dec_max_2048.txt')

p = np.zeros(npix)
for i in range(npix):
    for j in range(len(prob)):
        if ra_pix[i]>ra_min[j] and ra_pix[i]<ra_max[j] and de_pix[i]>de_min[j] and de_pix[i]<de_max[j]:
            p[i] = prob[j]

p = p/sum(p)

hp.mollview(p, title="Machine Learning Prob-Density",rot=(180,0))
#plt.savefig('PD.png')

cls = 100 * postprocess.find_greedy_credible_levels(p)

ax = plt.axes(projection='astro hours mollweide')
ax.grid()
event_ra = 1.7540559
event_de = -1.11701
projector  = hp.projector.MollweideProj()
x1,y1 = projector.ang2xy(np.array([-event_de+np.pi/2.0,event_ra+np.pi]))
img = ax.imshow_hpx(p, vmin=0., vmax=p.max(),cmap='cylon')
cs = ax.contour_hpx((cls, 'ICRS'),linewidths=0.5, levels=[50,90],colors=['green','blue'])
ax.plot_coord(SkyCoord(event_ra, event_de, unit='rad'), 'x',markeredgecolor='black', markersize=5)
text = []
pp = np.round([50,90]).astype(int)
ii = np.round(np.searchsorted(np.sort(cls),[50,90]) * deg2perpix).astype(int)
for i, p in zip(ii, pp):
            # FIXME: use Unicode symbol instead of TeX '$^2$'
            # because of broken fonts on Scientific Linux 7.
            text.append(u'{:d}% area: {:d} deg²'.format(p, i, grouping=True))
ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')
ax.figure.savefig('GW150914_pure_signal.png',dpi=150)
