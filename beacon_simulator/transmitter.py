import numpy as np
from matplotlib import pyplot as plt

def generate2DGausian(std, shape):
    '''
    Generates a 2D gausian of size shape
    std: Standard deviation of gausian
    shape: 2D tuple object 
    '''
    s = np.arange(-shape[0]//2, shape[0]//2)
    t = np.arange(-shape[1]//2, shape[1]//2)
    s,t = np.meshgrid(s,t)
    gausian = np.exp(-((s**2+t**2)/(2*std**2)))
    return gausian

def generateGausian(std, shape):
    s = np.arange(shape)
    return(np.exp(-0.5*s**2/std**2))

sampling = 1000
radius = 1
angle = np.pi
M = 1

zeroborder = sampling//radius//20


#Calculate magnitude of magnetic field in polar space
r = np.linspace(0, radius, sampling)
#Make sure power dont scale to infinty when r approches zero
r += generateGausian(10, sampling)

#Measure angle from 0 to 2 * pi
theta = np.linspace(0, 2*angle, sampling)

theta, r = np.meshgrid(theta, r)

H = M / (4*np.pi*r**3)*np.sqrt(1+3*np.cos(theta)**2)
plt.imshow(H, cmap='hot')
plt.show()

#Calculate magnitude of magnetic field in cartesian space
x = np.linspace(-radius, radius, sampling)
y = np.linspace(-radius, radius, sampling)

x,y = np.meshgrid(x,y)

#Again make sure power does not scale to infinity near zero
damper = generate2DGausian(1000, (sampling, sampling))
r = np.sqrt(x**2+y**2) + damper

theta = np.arctan(y/x)

H = M / (4*np.pi*r**3)*np.sqrt(1+3*np.cos(theta)**2)
plt.imshow(H, cmap='hot')
plt.show()

