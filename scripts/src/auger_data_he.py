import numpy as np

def ratio_and_error(x, y, dx, dy):
    rx2 = (dx/x)**2
    ry2 = (dy/y)**2
    
    ratio = x/y
    ratioError = ratio*np.sqrt(rx2 + ry2)
    
    return ratio, ratioError

#Energy bins for the histograms Auger
ebins_ = np.array([18.702, 18.801, 18.8996, 18.9986, 
                   19.0996, 19.2006, 19.3002, 19.3997, 
                   19.5002, 19.6011, 19.7001, 19.7983, 
                   19.8976, 19.9979, 20.0993, 20.1998, 
                   20.2986, 20.3973])

ecens_ = np.array([18.7515, 18.8503, 18.9491, 19.0491, 
                   19.1501, 19.2504, 19.34995, 19.44995, 
                   19.55065, 19.6506, 19.7492, 19.84795, 
                   19.94775, 20.0486, 20.14955, 20.2492, 20.34795])

dE_ = 10**ebins_[1:] - 10**ebins_[:-1]  # bin energy widths

#Data collected from Auger
auger1 = np.array([5338, 2039, 666, 181, 38, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])			#A=1
auger2 = np.array([10550, 8036, 5566, 3442, 1759, 630, 135, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0])		#2<A<4
auger3 = np.array([1919, 2307, 2463, 2373, 2132, 1773, 1337, 796, 455, 232, 70, 21, 3, 0, 0, 0, 0])	#5<A<22
auger4 = np.array([53, 69, 89, 114, 146, 174, 188, 157, 134, 114, 67, 44, 12, 3, 1, 0, 0])				#23<A<38
auger5 = np.array([0, 0, 0, 0, 8, 11, 16, 19, 22, 24, 18, 16, 7, 4, 4, 4, 2])						#39<A<56
#auger = auger1+auger2+auger3+auger4+auger5
auger_ = np.array([16970,12109,8515,5939,4048,2567,1664,979,619,373,152,80,23,9,6,0,0])


#Normalization of auger data
n_auger_ = auger_.sum()
n_auger_ = float(n_auger_)
nauger_ = auger_/n_auger_
nauger1 = auger1/n_auger_
nauger2 = auger2/n_auger_
nauger3 = auger3/n_auger_
nauger4 = auger4/n_auger_
nauger5 = auger5/n_auger_

sauger_ = nauger_/n_auger_
sigma_auger_ = np.sqrt(auger_)
sauger1 = nauger1/n_auger_
sauger2 = nauger2/n_auger_
sauger3 = nauger3/n_auger_
sauger4 = nauger4/n_auger_
sauger5 = nauger5/n_auger_

Jauger_ = auger_ / dE_ # Energy flux
Jauger_err_ = np.sqrt(auger_) / dE_ # Flux error

# The first bin is always 1
Jauger_scaled_, Jauger_err_scaled_ = ratio_and_error(x=Jauger_,
                                                  dx=Jauger_err_,
                                                   y=Jauger_[0],
                                                  dy=Jauger_err_[0])

# Added by Christian
# ==================
binSize = 0.1
minBin = 18.45
maxBin = 20.35
ecens = np.arange(minBin, maxBin, binSize)
ebins = np.arange(minBin-binSize/2, maxBin+binSize/2, binSize)
dE = 10**ebins[1:] - 10**ebins[:-1]  # bin energy widths

auger = np.array(
        [76176, 44904, 26843, 16970, 12109,
         8515, 5939, 4048, 2567, 1664,
         979, 619, 373, 152, 80,
         23, 9, 6, 0, 0],
        dtype=np.float64)

Jauger = auger / dE # Energy flux
Jauger_err = np.sqrt(auger) / dE # Flux error

# The first bin is always 1
Jauger_scaled, Jauger_err_scaled = ratio_and_error(x=Jauger,
                                                  dx=Jauger_err,
                                                   y=Jauger[0],
                                                  dy=Jauger_err[0])
# ==================
