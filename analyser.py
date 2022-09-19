import wave
import numpy as np
import scipy.fft
import scipy.signal
from scipy.ndimage import gaussian_filter1d

from matplotlib import pyplot as plt

with open("DRAW.NP",'rb') as i_f:
    wav = np.load(i_f).astype(float)
    wav = wav[:,0]+1j*wav[:,1]
    wav-= wav.mean()

#plt.plot(np.linspace(0,1,len(wav)),wav)
#plt.show()

f = scipy.fft.fftshift(scipy.fft.fft(wav)/len(wav))
ff = (f.real**2+f.imag**2)**.5
x = scipy.fft.fftshift(scipy.fft.fftfreq(len(wav),1/len(wav)))
peaks,_ = scipy.signal.find_peaks(ff)

peaks = [(x[p[1]],p[1]) for p in sorted([(ff[p],p) for p in peaks],reverse=True)[:20]]

ps = [p[1] for p in peaks]

print(f[ps],ff[ps],x[ps],sep='\n')
plt.title("REAL")
plt.plot(x,f.real)
plt.plot(x[ps],f.real[ps],'x')
plt.figure()
plt.title("IMAG")
plt.plot(x,f.imag)
plt.plot(x[ps],f.imag[ps],'x')
plt.figure()
plt.title("MAG")
plt.plot(x,ff)
plt.plot(x[ps],ff[ps],'x')
plt.show()