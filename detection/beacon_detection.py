import matplotlib.pyplot as plt
import numpy as n
import scipy.signal as ss

sr=768e3
center_freq=500e3
f0=457e3

# this is where the center frequency will appear in the complex
# baseband signal
baseband_center_freq = f0-center_freq

# sample-rate 768e3
z = n.fromfile("nonamp.bin",dtype=n.complex64)

# maximum frequency deviation in Hz
# As per ETSI EM 300 718-1 specification
# 
max_freq_dev=1000

# fft length
nfft=77000

# how many samples to step when doing an fft
step=770

def spectrogram(z,nfft=77000,step=770,sr=768e3,wf=ss.hann(77000)):
    n_steps=int((len(z)-nfft)//step)
    Sxx=n.zeros([nfft,n_steps])
    tvec=n.arange(n_steps)*step/sr
    fvec=n.fft.fftshift(n.fft.fftfreq(nfft,d=1/sr))
    for i in range(n_steps):
        Sxx[:,i]=n.fft.fftshift(n.abs(n.fft.fft(z[(i*step):(i*step+nfft)]*wf))**2.0)
    return(tvec,fvec,Sxx)


#freq_vec, time_vec, Sxx=ss.spectrogram(z,fs=sr,nperseg=nfft,noverlap=int(0.9*nfft))

time_vec,freq_vec,Sxx=spectrogram(z,sr=sr,nfft=nfft,step=step,wf=ss.hann(77000))

# frequency indices of interest
fidx=n.where( n.abs(freq_vec - baseband_center_freq) < max_freq_dev)[0]

freq_vec2=freq_vec[fidx]

print(Sxx.shape)
pwr=n.abs(Sxx[fidx,:])**2.0
#noise_floor = n.median(pwr)

#dB=dB-noise_floor

# remove median power to remove constant tones
for fi in range(pwr.shape[0]):
    # signal to noise ratio for each frequency bin
    pwr[fi,:]=(pwr[fi,:]-n.median(pwr[fi,:]))/n.median(pwr[fi,:])

# detect center frequency
# we know that the blips are within 70 ms
duration=n.max(time_vec)-n.min(time_vec)
# this is how many seconds there are in one time sample of the spectrogram
dt=(step/sr)

# this is how many time samples contain a blip
n_blip_samples=int((duration*70e-3)/dt)

# go through each frequency bin
# and count how much power is in blips
total_blip_pwr=n.zeros(pwr.shape[0])
for fi in range(pwr.shape[0]):
    # todo: could be improved by integrating power in one second segments
    # sum the peak power of each one second segment, because each time
    # sample is approximately the length of a blip (770 samples)
    idx=n.argsort(pwr[fi,:])
    total_blip_pwr[fi]=n.mean(pwr[fi,idx[(pwr.shape[1]-n_blip_samples):(pwr.shape[1])]])

max_idx=n.argmax(total_blip_pwr)
# estimate standard deviation with median
std_estimate = n.median(n.abs(total_blip_pwr - n.median(total_blip_pwr)))
    
# plot signal-to-noise ratio of blip pwr
plt.plot(freq_vec2,total_blip_pwr/std_estimate)
plt.axvline(freq_vec2[max_idx],color="red")
plt.axhline(total_blip_pwr[max_idx]/std_estimate,color="red")
plt.title("Amount of power in blips")
plt.xlabel("Frequency (Hz)")
plt.ylabel("SNR/standard deviation")
plt.show()

# amount of power from beacon divided by one standard deviation
peak_beacon_power = total_blip_pwr[max_idx]/std_estimate


# make sure no negative value
pwr[pwr<0]=1e-3

dB=10.0*n.log10(pwr)    
plt.pcolormesh(time_vec,freq_vec2,dB,vmin=-3,vmax=30)
plt.axhline(freq_vec2[max_idx],color="red",alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
#plt.pcolormesh(dB-noise_floor)
plt.colorbar()
plt.show()



