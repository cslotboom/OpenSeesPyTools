# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:13:13 2019
@author: Christian
"""
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks




def SampleCyclicData(ExperimentX, ExperimentY, AnalysisX, AnalysisY,
                     Nsample = 10, peakDist = 2, peakwidth = None):
    """
    
    TODO: Consider renaming to "SampleData"
    
    This functions samples two data sets that undergo a cyclic. The function 
    automatically detects reversals in the data. Adjustments may be necessary
    to the tolerance of 
    
    The Experiment and Analysis must have the same number of cycles

    Parameters
    ----------
    ExperimentX : Array
        The X values from the experiment dataset.
    ExperimentY : Array
        The Y values from the experiment dataset.
    AnalysisX : Array
        The X values from the analysis dataset.
    AnalysisY : Array
        The Y values from the analysis dataset.

    Raises
    ------
    Exception
        If the datasets don't have.

    Returns
    -------
    Rnet : float
        Returns a the sampled value 

    """
       
    # We get the indicies where the reversal happens.
    
    ExperimentIndicies = GetCycleIndicies(ExperimentX, ExperimentY, Nsample, peakDist, peakwidth)
    AnalysisIndicies = GetCycleIndicies(AnalysisX, AnalysisY, Nsample, peakDist, peakwidth)
    
    # We check that both curves have the same number of indicies
    NIndex = len(ExperimentIndicies) - 1
    NIndex_2 = len(AnalysisIndicies) - 1
    R= np.zeros(NIndex)
    
    # If they don't we create a error
    if NIndex!=NIndex_2:
        raise Exception('The experiment and Analysis have a different number of cycles')
        
    # For each index, we loop through 
    for ii in range(NIndex):
        # We get define the arrays for the x and y coordinants of the sub-vector
        Ex = np.zeros(NIndex)
        Ey = np.zeros(NIndex)
        Ax = np.zeros(NIndex)
        Ay = np.zeros(NIndex)
        
        # We get the subvector values between the indicies
        [Ex,Ey] = GetCycleSubVector(ExperimentX , ExperimentY, ExperimentIndicies[ii], ExperimentIndicies[ii+1], Nsample)
        [Ax,Ay] = GetCycleSubVector(AnalysisX , AnalysisY, AnalysisIndicies[ii], AnalysisIndicies[ii+1], Nsample)
        
        # We sample each point on the curve using the difference between the
        # two points
        R[ii] = np.sum(((Ey-Ay)**2 + (Ex-Ax)**2)**0.5)
        
    Rnet = np.sum(R)
    
    return Rnet


def SampleMonotonicData(ExperimentX, ExperimentY, AnalysisX, AnalysisY,
                        Nsample = 10):
    """
    This functions works the same way as the cyclic data function
    
    This functions samples two data sets of data and returns a 
    
    The Experiment and Analysis must have the same number of cycles

    Parameters
    ----------
    ExperimentX : Array
        The X values from the experiment dataset.
    ExperimentY : Array
        The Y values from the experiment dataset.
    AnalysisX : Array
        The X values from the analysis dataset.
    AnalysisY : Array
        The Y values from the analysis dataset.

    Returns
    -------
    Rnet : float
        Returns a the sampled value 

    """
    
    Nindex = len(ExperimentX)
    R= np.zeros(Nsample)
    
    for ii in range(Nsample):
        Ex = np.zeros(Nsample)
        Ey = np.zeros(Nsample)
        Ax = np.zeros(Nsample)
        Ay = np.zeros(Nsample)
        
        
        [Ex,Ey] = GetCycleSubVector(ExperimentX , ExperimentY, 0, Nindex, Nsample)
        [Ax,Ay] = GetCycleSubVector(AnalysisX , AnalysisY, 0, Nindex, Nsample)
        
        R[ii] = np.sum(((Ey-Ay)**2 + (Ex-Ax)**2)**0.5)
        
    Rnet = np.sum(R)
    
    return Rnet


def GetCycleSubVector(VectorX, VectorY, Index1, Index2, Nsample):
    """
    
    This function takes a input x y curve, then returns a linearlized curve
    between two indicies
    
    

    Parameters
    ----------
    VectorX : Array
        The input X vector.
    VectorY : Array
        The input Y vector.
    Index1 : int
        The first index we want to calculate values between.
    Index2 : int
        The second index we want to calculate values between.
    Nsample : int
        The desired number of data points between the two vectors.

    Returns
    -------
    xSample : TYPE
        The x points of the sample curve.
    ySample : TYPE
        The y points of the sample curve.

    """
    
    x1 = VectorX[Index1]
    x2 = VectorX[Index2]
    
    
    TempDataX = VectorX[Index1:(Index2+1)]
    TempDataY = VectorY[Index1:(Index2+1)]
    xSample = np.linspace(x1,x2,Nsample)
    ySample = ShiftDataFrame(TempDataX,TempDataY,xSample)
        
    return xSample,ySample
        
    
def GetCycleIndicies(VectorX, VectorY = [], CreatePlot = False, peakDist = 2, 
                     peakWidth = None, peakProminence = None):
    """
    This function finds the index where there is areversal in the XY data. 
    You may need to adjust the find peaks factor to get satisfactory results.
    

    Parameters
    ----------
    VectorX : 1D array
        Input X Vector.
    VectorY : 1D array
        Input Y Vector. This is only used if we want to plot the values.
    CreatePlot : Boolean
        This switch specifies whether or not to display the output.plot
    peakDist : TYPE, optional
        The sampling distance used to find peaks. The default is 2.

    Returns
    -------
    Indexes : Arrays
        Returns the arrays at which reversals occur.

    """
    
    
    
    # We find the intermediate peak values
    # We use height = 0 to select only positive or negative peaks.
    MaxIndex,_ = find_peaks(VectorX, height = 0, distance = peakDist, 
                            width = peakWidth, prominence = peakProminence)
    
    MinIndex,_ = find_peaks(-VectorX, height = 0, distance = peakDist, 
                            width = peakWidth, prominence = peakProminence)    
        
    # We store define an array that will later be used to store the
    # the max and min values in an array of indexes
    Nindex = len(MaxIndex) + len(MinIndex) + 2
    Indexes = np.zeros(Nindex, dtype = int)
    
    
    # # If no value was found, we assume monotonic
    # if not MaxIndex: 
    #     print('No local peaks were found, assuming x data is monotonic.')
    #     Nindex = 2
    #     MinIndex = [0]
    #     MaxIndex = [len(VectorX) - 1]
    #     Indexes = np.zeros(Nindex, dtype = int)


    # Define first and last point
    Indexes[0] = 0
    Indexes[-1] = len(VectorX) - 1
    
    # if only one point exists
    
    L1 = len(MinIndex)
    L2 = len(MaxIndex)
    
    # We check the order, starting with the minimum number of possibilities
    # if there are no minimus, order doesn't matter
    if L1 == 0 and L2 == 0:
        order = 1
    elif L1 == 0:
        order = 2
    elif L2 == 0:
        order = 1
    elif MinIndex[0] < MaxIndex[0]:
        order = 1
    else:
        order = 2
    
    # We assign the order. We also check if there are any entries to assign.
    if order == 1:
        Indexes[1:-1:2] = MinIndex
        if L2 !=0:
            Indexes[2:-1:2] = MaxIndex
    else:
        Indexes[1:-1:2] = MaxIndex
        if L1 !=0:
            Indexes[2:-1:2] = MinIndex
    
    # make a picture if it is true
    if CreatePlot == True:
        # MaxValueXpos = VectorX[MaxIndex]
        # MaxValueYpos = VectorY[MaxIndex]

        # MaxValueXneg = VectorX[MinIndex]
        # MaxValueYneg = VectorY[MinIndex]
        if VectorY == []:
            VectorY = VectorX
        
        MaxValueXValue = VectorX[Indexes]
        MaxValueYValue = VectorY[Indexes]        

        fig = plt.subplots()
        line  = plt.plot(Indexes,MaxValueYValue,'x')
        # line  = plt.plot(MaxIndex,MaxValueYpos,'x')
        # line  = plt.plot(MinIndex,MaxValueYneg,'x')
        
        line2  = plt.plot(VectorY[:])
        plt.title('Peak Indexes')
        plt.show()
        
        
        fig = plt.subplots()        
        line2  = plt.plot(VectorX,VectorY)
        line  = plt.plot(MaxValueXValue,MaxValueYValue,'x')
        plt.title('Peak Index x values')
        plt.show()
        
    
    
    return Indexes


def LinearInterpolation(x1, x2, y1, y2, x):
    dx = (x2 - x1)
    if dx == 0:
        print('x1 and x2 are the same, using y1')
        y = y1
    else:
        y = ((y2 - y1) / (dx))*(x - x1) + y1
    
    return y


def ShiftDataFrame(Samplex, Sampley, Targetx):
    """
    
    SciPy's ID interpolate baseically does what this does, it probably makes
    more sense to use that instead.
    
    This functions shifts x/y of a sample vector to a target x vector.
    Intermediate values are found using linearlizaton
    Both the input and putput data must be a function.
    
    The sample point MUST be within the bounds
    
              sample x/y data
              x1___x2____u1___x2_________x5_________x6
              y1___x2____x1___x2_________x5_________x6
    
              Targert x data
              x1_____x2___x3____x4___x5______x6__x7

    Returns:
              y1_____y2___y3____y4___y4______y5__y7
              
    # We will need to break if 
    Parameters
    ----------
    Samplex : Array 1d.
        The x values for the sample curves we want to shift into.
    Sampley : Array 1d.
        The y values for the sample curves we want to shift into.
    Targetx : Array 1d.
        The X vector we want to shift the target into.

    Returns
    -------
    Targety : Array 1d.
        The Y vector we shifted  the target into.

    """
    
        
    # 
    SampleRate = 1    
    
    # Get number of samples needed
    Nsamples = len(Targetx)
    Targety = np.zeros(Nsamples)
    
    MinData = np.min(Samplex)
    MaxData = np.max(Samplex)
    
    # Define indexes
    ii = 0
    jj = 1
    x1 = Samplex[0]
    y1 = Sampley[0]
    x2 = Samplex[jj]
    y2 = Sampley[jj]
    Currentmax = max(x1,x2)
    Currentmin = min(x1,x2)
    
    while (ii < Nsamples):
        
        #Get the target Point
        x = Targetx[ii]
        
        if x < MinData:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Target is less than minimum bounds")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            #Targety = None
            break
        if x > MaxData:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Target is greater than maximum bounds")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            #Targety = None
            break
        
        # We look to linearly interpolate between the two points
        
        # We check if x is within the range of the current sample points
        # if it isn't, we check the next data points
        while ((x < Currentmin) or (Currentmax < x)):
            jj += 1
            #print(jj)
            x1 = x2
            y1 = y2
            x2 = Samplex[jj]
            y2 = Sampley[jj]

            Currentmax = max(x1,x2)
            Currentmin = min(x1,x2)
               
        
        # Interpolate
        Targety[ii] = LinearInterpolation(x1,x2,y1,y2,x)
        
        ii+=1
        
    return Targety


## Combine two sets of data
#Samplex = np.array(range(50))/2.5
#Sampley = Samplex**2
#
#Targetx = np.array([-1., 2., 4., 10., 11., 16.,21])
#
#Targety = ShiftDataFrame(Sampley,Samplex, Targetx)
#
#fig, ax = plt.subplots()
#line1, = ax.plot(Samplex,Sampley, label='sampleData')
#line2, = ax.plot(Targetx,Targety, label='targetData')


        
#def ShiftDT(samplex,samply,dt1,dt2):
## This function shifts data from one constant x frame to a different constant x frame using linear interpolation.
#    
#    
#    