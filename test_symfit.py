# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 20:12:35 2018

@author: Main
"""
import symfit
import numpy as np
import matplotlib.pyplot as plt

def gausian(x, a, b, c, d):
    return a*np.exp(-((b - x)**2)/(2*c**2)) + d

def gausian_cross_correlation(x, a0, b0, c0, a1, b1, c1, d):
    return (a0**2)*(a1**2)*np.exp(-((-b0 + b1 + x)**2)/(2*(c0**2 + c1**2)))*np.sqrt(2*np.pi/(c0**2 + c1**2)) + d

#make fake data
p0 = [2.5, .5, 2.2]
p1 = [1.3, -0.7, 0.7]
p2 = [3.3, .1, 1.2]
d = 0

xdata0 = np.linspace(-10, 10, 200) # From -10 to 10 in 200 steps
ynoise = np.random.normal(0.0, scale=.2, size=(200,))
ydata0 = gausian_cross_correlation(xdata0, *p0, *p1, 0) + ynoise
xdata1 = np.linspace(-10.5, 10, 205)
ynoise = np.random.normal(0.0, scale=.2, size=(205,))
ydata1 = gausian_cross_correlation(xdata1, *p0, *p2, 0) + ynoise
xdata2 = np.linspace(-10.5, 10.5, 210)
ynoise = np.random.normal(0.0, scale=.2, size=(210,))
ydata2 = gausian_cross_correlation(xdata2, *p2, *p1, 0) + ynoise

plt.plot(xdata0, ydata0)
plt.plot(xdata1, ydata1)
plt.plot(xdata2, ydata2)
plt.show()

#make the model and fit
x0, x1, x2, y0, y1, y2 = symfit.variables("x0, x1, x2, y0, y1, y2")
a0, b0, c0, d0 = symfit.parameters("a0, b0, c0, d0")
a1, b1, c1, d1 = symfit.parameters("a1, b1, c1, d1") 
a2, b2, c2, d2 = symfit.parameters("a2, b2, c2, d2")
model = {y0: (a0**2)*(a1**2)*symfit.exp(-((-b0 + b1 + x0)**2)/(2*(c0**2 + c1**2)))*symfit.sqrt(2*symfit.pi/(c0**2 + c1**2)) + d0,
         y1: (a0**2)*(a2**2)*symfit.exp(-((-b0 + b2 + x1)**2)/(2*(c0**2 + c2**2)))*symfit.sqrt(2*symfit.pi/(c0**2 + c2**2)) + d1,
         y2: (a2**2)*(a1**2)*symfit.exp(-((-b2 + b1 + x2)**2)/(2*(c2**2 + c1**2)))*symfit.sqrt(2*symfit.pi/(c2**2 + c1**2)) + d2}
fit = symfit.Fit(model, x0=xdata0, x1=xdata1, x2=xdata2, y0=ydata0, y1=ydata1, y2=ydata2,
                 constraints=[symfit.Ge(c0, 0), symfit.Ge(c1, 0), symfit.Ge(c2, 0), symfit.Ge(a0, 0), symfit.Ge(a1, 0), symfit.Ge(a2, 0)])
fit_result = fit.execute()

#plot and print results
y_result0, y_result1, y_result2 = fit.model(x0=xdata0, x1=xdata1, x2=xdata2, **fit_result.params)

plt.plot(xdata0, ydata0)
plt.plot(xdata0, y_result0)
plt.plot(xdata1, ydata1)
plt.plot(xdata1, y_result1)
plt.plot(xdata2, ydata2)
plt.plot(xdata2, y_result2)
print("a0 = {}, b0 = {}, c0 = {}, d0 = {}".format(*p0,0))
print("a1 = {}, b1 = {}, c1 = {}, d1 = {}".format(*p1,0))
print("a2 = {}, b2 = {}, c2 = {}, d2 = {}".format(*p2,0))
print(fit_result)
