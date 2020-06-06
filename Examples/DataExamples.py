# -*- coding: utf-8 -*-
"""
Created on Fri May  23rd 22:37:21 2020

@author: Christian


******************************************************************************

These functions allow users to compare xy curves.


******************************************************************************
"""

import numpy as np
import openseespytools.data as opd
import matplotlib.pyplot as plt
import scipy


# =============================================================================
# Find Index for cycle reversal points some input data
# =============================================================================

"""
This is useful for finding data reversal points in our data. If your data is 
noisey, you may want to skip some sub-peaks and only find big reversals.
"""

# We define a vector that has three cycle changes directions three times
x1 = np.linspace(0,1,21)
x2 = np.linspace(1,-1,21)
x3 = np.linspace(-1,3,21)
x = np.concatenate([x1,x2,x3])

revesalIndexes = opd.GetCycleIndicies(x)


# We can also plot the to make sure it worked. We'll define a y vector for
# illustrative purposes.
y1 = np.sin(x1)
y2 = np.e**x2*np.sin(x2)
y3 = np.e**-x3*np.sin(x3)
y = np.concatenate([y1,y2,y3])

revesalIndexes = opd.GetCycleIndicies(x, VectorY = y, CreatePlot = True)


# we can also find the reversal point for more tricky curves
# We can also skip small revsals. y is used only for illustrative purposes
x = np.linspace(0, 10, 1000)

# a triangle with small reversals
triangleBig = scipy.signal.sawtooth(x*2,0.5)
triangleSmall = scipy.signal.sawtooth(x*20,0.5)/7
triangle = triangleBig + triangleSmall

# a noisey triangle signla
permutate = np.random.normal(0,1,1000)/2
Ynoise = triangleBig + permutate
Ynoise = scipy.signal.savgol_filter(Ynoise,53,2)



plt.plot(x,triangle)

# Here We find the peaks of some tricky curves using the optional arguements
noiseIndexes = opd.GetCycleIndicies(Ynoise, VectorY = Ynoise, CreatePlot = True, peakDist = 20, peakProminence = 0.2)
triangleIndexes = opd.GetCycleIndicies(triangle, VectorY = triangle, CreatePlot = True, peakDist = 200, peakProminence = 0.1)


# =============================================================================
# Get a sub-Vector
# =============================================================================

"""
We can also get the sub-vector of a curve between our two indexes
"""
# get the sub vectors
[subvectorx1, subvectory1] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[0],noiseIndexes[1], 100)
[subvectorx2, subvectory2] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[1],noiseIndexes[2], 100)
[subvectorx3, subvectory3] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[2],noiseIndexes[3], 100)


# show the subvectors
plt.plot(subvectorx1, subvectory1)
plt.plot(subvectorx2, subvectory2)
plt.plot(subvectorx3, subvectory3)
plt.show()

# GetCycleSubVector(x,y,revesalIndexes[1])

# Subvector1 = revesalIndexes[]


# =============================================================================
# Shift one sampel
# =============================================================================


"""
This is used if you want to shift a piecewise x-y curve into another x domain.
Linear interpolation is used. The function only works for a monotonic domain.

This can be usful when you want to compare two curves in a common domain.
"""

# Define our curves
npoints = 1000
x = np.linspace(0, 6, npoints)
y = np.sin(x)

# Define the sample domain
xTarget = np.linspace(0, 6, 10)

# Define the sample range
yTarget = opd.ShiftDataFrame(x, y, xTarget)

# Plot the curves
fig, ax = plt.subplots()
plt.plot(x,y)
plt.plot(xTarget,yTarget)


# we can shift two curves intoa a common space.
# We define two curves
x1 = np.linspace(0, 6, 350)
x2 = np.linspace(0, 6, 756)
y1 = np.ones(npoints)
y2 = np.sin(x2)

# Shift both curves
y1Target = opd.ShiftDataFrame(x1, y1, xTarget)
y2Target = opd.ShiftDataFrame(x2, y2, xTarget) 

# Now we can easily do operations between two curves!
dy = y2Target - y1Target



# =============================================================================
# Compare the similarity of two curves
# =============================================================================

"""
We can also sample two curves.
We can use this to make decisions about our data!
"""

# Define some initial vector
npoints = 1000
x = np.linspace(0, 6, npoints)
y = np.sin(x)


# Create two vectors with some noise.
permutate = np.random.normal(0, 1, npoints)
YnoiseBig = y + permutate / 2
YnoiseSmall = y + permutate / 8
YnoiseBig = scipy.signal.savgol_filter(YnoiseBig,53,2)
YnoiseSmall= scipy.signal.savgol_filter(YnoiseSmall,53,2)

# Here were sample two different curves to return the diffete
R1 = opd.SampleData(x,y,x,YnoiseSmall,Nsample=20)

# Here were
R2 = opd.SampleData(x,y,x,YnoiseBig,Nsample=20)

if R1 < R2:
    print( 'Curve 1 is a better fit, and the objective function is:', R1)
else:
    print( 'Curve 1 is a better fit, and the objective function is:', R2)


fig, ax = plt.subplots()

plt.plot(x,YnoiseBig,x,YnoiseSmall,x,y)

