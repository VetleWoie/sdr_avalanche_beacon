#!/usr/bin/env python3

import matplotlib.pyplot as plt
import digital_rf as drf
import numpy as n

data_dir="data/mammuth_element_470k_100k"
channel="loop"
d=drf.DigitalRFReader(data_dir)
b=d.get_bounds(channel)
print(b)

z=d.read_vector_c81d(b[0]+100000,1000000,channel)
plt.plot(z.real)
plt.plot(z.imag)
plt.show()
fvec=n.fft.fftshift(n.fft.fftfreq(1000000,d=1.0/100e3))+470e3
plt.plot(fvec/1e3,10.0*n.log10(n.abs(n.fft.fftshift(n.fft.fft(z)))**2.0))
plt.xlabel("Frequency (kHz)")
plt.ylabel("Power spectral density")
plt.show()
