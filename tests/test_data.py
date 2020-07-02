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

def test_CyclicIndexes():
    # We define a vector that has three cycle changes directions three times
    x1 = np.linspace(0,1,21)
    x2 = np.linspace(1,-1,21)
    x3 = np.linspace(-1,3,21)
    x = np.concatenate([x1,x2,x3])
    
    revesalIndexes = opd.GetCycleIndicies(x)
    
    assert np.all(revesalIndexes == np.array([ 0, 20, 41, 62]))


def CyclicIndexes2():
    x1 = np.linspace(0,1,21)
    x2 = np.linspace(1,-1,21)
    x3 = np.linspace(-1,3,21)
    x = np.concatenate([x1,x2,x3])

    y1 = np.sin(x1)
    y2 = np.e**x2*np.sin(x2)
    y3 = np.e**-x3*np.sin(x3)
    y = np.concatenate([y1,y2,y3])
    
    revesalIndexes = opd.GetCycleIndicies(x, VectorY = y, CreatePlot = True)

    plt.close('all')

    return revesalIndexes
    # assert np.all(revesalIndexes == np.array([ 0, 20, 41, 62]))
    
    
    
def test_plot_fn(monkeypatch):
    # repress the show plot attribute
    monkeypatch.setattr(plt, 'show', lambda: None)
    revesalIndexes = CyclicIndexes2()    
    assert np.all(revesalIndexes == np.array([ 0, 20, 41, 62]))


def triangles():
    np.random.seed(19991231)

    x = np.linspace(0, 10, 1000)
    # a triangle with small reversals
    triangleBig = scipy.signal.sawtooth(x*2,0.5)
    triangleSmall = scipy.signal.sawtooth(x*20,0.5)/7
    triangle = triangleBig + triangleSmall
    triangleIndexes = opd.GetCycleIndicies(triangle, VectorY = triangle, CreatePlot = True, peakDist = 200, peakProminence = 0.1)
    return triangleIndexes
    
def test_triangles(monkeypatch):
    
    solution = np.array([  0, 172, 314, 455, 628, 769, 942, 999])

    monkeypatch.setattr(plt, 'show', lambda: None)
    revesalIndexes = triangles()    
    assert np.all(revesalIndexes == solution)
    
    
def noise():
    np.random.seed(19991231)

    x = np.linspace(0, 10, 1000)
    triangleBig = scipy.signal.sawtooth(x*2,0.5)    # a noisey triangle signla
    permutate = np.random.normal(0,1,1000)/2
    Ynoise = triangleBig + permutate
    Ynoise = scipy.signal.savgol_filter(Ynoise,53,2)
    
    noiseIndexes = opd.GetCycleIndicies(Ynoise, VectorY = Ynoise, CreatePlot = True, peakDist = 20, peakProminence = 0.2)
    
    return noiseIndexes

def test_noise(monkeypatch):

    solution = np.array([  0, 162, 329, 465, 614, 778, 934, 999])
    monkeypatch.setattr(plt, 'show', lambda: None)
    noiseIndexes = noise()
    
    assert np.all(noiseIndexes == solution)

# test_noise()


# =============================================================================
# Get a sub-Vector
# =============================================================================


def sub_vector():
    np.random.seed(19991231)
    
    # get the sub vectors
    x = np.linspace(0, 10, 1000)
    triangleBig = scipy.signal.sawtooth(x*2,0.5)    # a noisey triangle signla
    permutate = np.random.normal(0,1,1000)/2
    Ynoise = triangleBig + permutate
    Ynoise = scipy.signal.savgol_filter(Ynoise,53,2)
    noiseIndexes = opd.GetCycleIndicies(Ynoise, VectorY = Ynoise, CreatePlot = True, peakDist = 20, peakProminence = 0.2)
    
    [subvectorx1, subvectory1] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[0],noiseIndexes[1], 100)
    # [subvectorx2, subvectory2] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[1],noiseIndexes[2], 100)
    # [subvectorx3, subvectory3] = opd.GetCycleSubVector(x, Ynoise, noiseIndexes[2],noiseIndexes[3], 100)
    
    return subvectorx1[55], subvectory1[23]
    
    
    
def test_sub_vector(monkeypatch):
    
    
    monkeypatch.setattr(plt, 'show', lambda: None)
    check = sub_vector()
    
    solutionx = 0.9009009009009009
    solutiony = -0.29884739267546534
    solution  = np.array([solutionx, solutiony])
    
    assert np.max(np.abs((check - solution))) < 10**-8

# =============================================================================
# Shift one sampel
# =============================================================================


"""
This is used if you want to shift a piecewise x-y curve into another x domain.
Linear interpolation is used. The function only works for a monotonic domain.

This can be usful when you want to compare two curves in a common domain.
"""
def test_shiftDataFrame():

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
    solution = [0.0, 0.618369803069737, 0.9719379013633127, 0.9092974268256817]
    solution = np.array(solution)  
    
    assert np.max(np.abs((yTarget[:4] - solution))) < 10**-8


def test_shiftDataFrame2():
    npoints = 1000
    xTarget = np.linspace(0, 6, 10)
    
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
    solution = np.array([-1.0, -0.3816321204274141, -0.028067401425340144])
    assert np.max(np.abs((dy[:3] - solution))) < 10**-8


# # =============================================================================
# Compare the similarity of two curves
# =============================================================================

"""
We can also sample two curves.
We can use this to make decisions about our data!
"""
def test_sampleData():
    # Define some initial vector
    npoints = 1000
    x = np.linspace(0, 6, npoints)
    y = np.sin(x)
    np.random.seed(19991231)
    
    
    # Create two vectors with some noise.
    permutate = np.random.normal(0, 1, npoints)
    YnoiseBig = y + permutate / 2
    YnoiseBig = scipy.signal.savgol_filter(YnoiseBig,53,2)
        
    # Here were
    R2 = opd.SampleData(x,y,x,YnoiseBig,Nsample=20)
    
    solution =  2.0660527205276886
    assert np.max(np.abs((R2 - solution))) < 10**-8


def test_sampleData_norm():
    # Define some initial vector
    npoints = 1000
    x = np.linspace(0, 6, npoints)
    y = np.sin(x)
    np.random.seed(19991231)
    
    def norm(c1x, c1y, c2x, c2y):
        R = np.max(np.abs(c1y-c2y))
        return R
    
    # Create two vectors with some noise.
    permutate = np.random.normal(0, 1, npoints)
    YnoiseBig = y + permutate / 2
    YnoiseBig = scipy.signal.savgol_filter(YnoiseBig,53,2)
        
    # Here were
    R2 = opd.SampleData(x,y,x,YnoiseBig,Nsample=20, Norm = norm)
    
    solution =  6.279415498198926
    assert np.max(np.abs((R2 - solution))) < 10**-8
# test_sampleData_norm()