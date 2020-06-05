# -*- coding: utf-8 -*-
"""
Created on Fri May  23rd 22:37:21 2020

@author: Christian


******************************************************************************

First, run this code in your consol. This will make the animations display
in a seperate window. The animations will not work otherwise.

%matplotlib qt

******************************************************************************
"""

import numpy as np
import openseespytools_working.response as opr
import scipy

# =============================================================================
# Animate a simple response
# =============================================================================

npoints = 1000
x = np.linspace(0,2*np.pi,npoints)
y = np.sin(x)

# A plot of every second data point at 24 fps
opr.AnimateXY(x,y, fps = 24, rFactor = 2)

# animiate a curve, but skipping some start and end frames
opr.AnimateXY(x,y, fps = 24, skipStart = 300,  skipEnd = 500)

# animate only the  a curve, but skipping the start frames
opr.AnimateXY(x,y, fps = 24, rFactor = 2, outputFrames = 48)


# =============================================================================
# Animate a pair of curves
# =============================================================================

# Create a vector with some noise.
permutate = np.random.normal(0,1,npoints)/4
Ynoise = y + permutate
Ynoise = scipy.signal.savgol_filter(Ynoise,53,2)

# Define the first and second curve
Curve1 = np.zeros([3*npoints,2])
Curve2 = np.zeros([3*npoints,2])

# Define the first and second curve
Curve1[:,0] = np.concatenate([x,x[::-1],x])
Curve1[:,1] = np.concatenate([y,-y[::-1],y])
Curve2[:,0] = np.concatenate([x,x[::-1],x])
Curve2[:,1] = np.concatenate([Ynoise,-Ynoise[::-1],Ynoise])


opr.AnimateCyclicXYCurves(Curve1, Curve2,NFrames=300,fps = 60)

