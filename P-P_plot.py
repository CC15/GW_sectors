import numpy as np
import healpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

def my_contour1(array):
    arg = np.argsort(array)
    total = np.sum(array)
    #Ratio = np.array([0.9,1.0/2,1.0/4,1.0/8,1.0/16,1.0/32,1.0/64])
    Ratio = np.array([0.997,0.95,0.9,0.8,0.68])
    #Ratio = np.array([0.0625,0.125,0.25,0.5])
    cont = np.zeros([len(array)],float)
    value = np.zeros([len(Ratio)+2],float)
    value[-1]  = np.max(array)
    k=1
    
    for ratio in Ratio:
        part=0.0
        for i in arg:
            part = part + array[i]
            if part/total>(1-ratio):
                value[k] = array[i]
                print(value[k])
                break

        for i in range(len(array)):
            if array[i]>value[k]:
                cont[i]=k
        
        k=k+1
    
    return value, cont


def my_contour_area(array,sig_level):
    ratio = float(sig_level)/100
    arg = np.argsort(array)
    total = np.sum(array)
    part = 0.0
    k=0
    for i in arg:
        part = part + array[i]
        if part/total>(1-ratio):
            break
        k = k+1
    
    area = (1-float(k)/float(len(array)))*4*np.pi
    return area

def my_cl(array,value):
    ratio=1.0
    arg = np.argsort(array)
    total = np.sum(array)
    part=0.0
    for i in arg:
        part = part + array[i]
        if array[i]>value:
            ratio = part/total
            break
    
    return 1-ratio


#f1 = open('log_map','r')
#lines = f1.readlines()
#map_gps = np.asarray([str('ProbDen_inj')+line.split(' ')[0]+str('.fits') for line in lines])
#ra = np.asarray([float(line.split(' ')[3]) for line in lines])
#dec = np.asarray([float(line.split(' ')[4]) for line in lines])

ra_test = np.loadtxt('RA_test.txt')
dec_test = np.loadtxt('Dec_test.txt')

cl = []
for i in range(2600):
#    map_gps = np.asarray([str('Healpy_Predictions/Healpy_Preds_')+str(i)+str('.fits')])
    map_gps = np.loadtxt("Healpy_Predictions/Healpy_preds_"+str(i)+".txt")
#    n = []
    area90 = []
    area50 = []
#    n = healpy.fitsfunc.read_map(map_gps)
#    print(sum(n))
    declination = dec_test[i]
    right_ascension = ra_test[i]
    value = healpy.pixelfunc.get_interp_val(map_gps,-declination+np.pi/2,right_ascension)
    cl.append(my_cl(map_gps,value)*100)
    area90.append(my_contour_area(map_gps,90))
    area50.append(my_contour_area(map_gps,50))

print(cl)

x_arr = np.array([0,100])
y_arr = np.array([0,1])
plt.hist(cl,500,cumulative=True,histtype='step',normed=True)
plt.plot(x_arr,y_arr,'--')
plt.xlim(0,100)
plt.xlabel('confidence level')
plt.ylabel('cumulative ratio')
#plt.title('coherent snr from '+str(snr_min)+' to '+str(snr_max))
plt.savefig('CLvCR_SNR-20-35_2048_sectors_2000_samples_2_det')


#plt.hist(np.log10(area90),50,cumulative=True,histtype='step',density=True,label='90%, median='+str(np.median(area90)))
#plt.hist(np.log10(area50),50,cumulative=True,histtype='step',density=True,label='50%, median='+str(np.median(area50)))
#plt.legend(loc=4)
#plt.ylabel('cumulative ratio')
#plt.xlabel('log(deg^2)')
#plt.savefig('area')

