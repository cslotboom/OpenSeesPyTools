# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

import openseespy.opensees as op

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib.animation as animation

import openseespytools.data as D


def AnimateXYFile(fileName_x, fileName_y='', column_x=1, column_y=1, 
               skipStart = 0, skipEnd = 0, rFactor=1, outputFrames=0, fps = 24, 
               Xbound = [],Ybound = []):
    """
    
    Animation function


    Parameters
    ----------
    fileName_x : string
        The name of the file to read x data from. 
    fileName_y : string, optional
        The name of the file to read y data from. If there is no path 
        specified, the x file name will be used as well. The default is ''.
    column_x : TYPE, optional
        The column in the x data file data is read from. The default is 1.
    column_y : TYPE, optional
        The column in the y data file data is read from. The default is 1.
    skipStart : int, optional
        If specified, this many datapoints will be skipped from the data start.
        The default is 0.
    skipEnd : int, optional
        If specified, this many frames will be skipped at the end of 
        the analysis. The default is 0.
    rFactor : int, optional
        If specified, only every "x" frames will be reduced by this factor. 
        The default is 1.
    outputFrames : int, optional
        The number of frames to be included after all other reductions. If the
        reduced number of frames is less than this value, no change is made.
        The default is 0.
    fps : int, optional
        Number of animation frames to be displayed per second. The default is 24.
    Xbound : TYPE, optional
        DESCRIPTION. The default is [].
    Ybound : TYPE, optional
        DESCRIPTION. The default is [].

    
    """

    # Get the x data
    InputData_x = np.loadtxt(fileName_x, delimiter=',')    
    xinput = InputData_x[:, column_x]
    
    # Get the y data
    if fileName_y == '':
        fileName_y = fileName_x
    InputData_y = np.loadtxt(fileName_y, delimiter=',')    
    yinput = InputData_y[:, column_y]      
    
    # Animate the output.
    AnimateXY(xinput, yinput, skipStart, skipEnd, rFactor, 
           outputFrames, fps, Xbound, Ybound)


def AnimateXY(xinput, yinput, skipStart = 0, skipEnd = 0, rFactor=1, 
           outputFrames=0, fps = 24, Xbound = [],Ybound = []):
    """
    Parameters
    ----------
    xinput : 1d array
        The input x coordinants. 
    yinput : 1d array
        The input y coordinants. 
    skipStart : int, optional
        If specified, this many datapoints will be skipped from the data start.
        The default is 0.
    skipEnd : int, optional
        If specified, this many frames will be skipped at the end of 
        the analysis. The default is 0.
    rFactor : int, optional
        If specified, only every "x" frames will be reduced by this factor. 
        The default is 1.
    outputFrames : int, optional
        The number of frames to be included after all other reductions. If the
        reduced number of frames is less than this value, no change is made.
        The default is 0.
    fps : int, optional
        Number of animation frames to be displayed per second. The default is 24.
    Xbound : [xmin, xmax], optional
        The domain of the chart. The default is 1.1 the max and min values.
    Ybound : [ymin, ymax], optional
        The range of the chart. The default is 1.1 the max and min values.

    
    """

    # Check if the x and y data are of the same length, if not raise a error.
    if len(xinput) != len(yinput):
        raise Exception('Lengths of input vectors unequal')
    
    # If end data is not being skipped, use the full vector length.
    if skipEnd ==0:
        skipEnd = len(xinput)
    
    
    # Set up bounds based on data from 
    if Xbound == []:
        xmin = 1.1*np.min(xinput)
        xmax = 1.1*np.max(xinput)
    else:
        xmin = Xbound[0]       
        xmax = Xbound[1]
    
    if Ybound == []:
        ymin = 1.1*np.min(yinput)  
        ymax = 1.1*np.max(yinput)        
    else:
        ymin = Ybound[0]       
        ymax = Ybound[1]          
    
    
    # Remove unecessary data
    xinput = xinput[skipStart:skipEnd]
    yinput = yinput[skipStart:skipEnd]

    # Reduce the data if the user specifies
    if rFactor != 1:
        xinput = xinput[::rFactor]
        yinput = yinput[::rFactor]
    
    # If the Frames isn't specified, use the length of the reduced vector.
    if outputFrames == 0:
        outputFrames = len(xinput)
    else:
        outputFrames = min(outputFrames,len(xinput))
    
    xinput = xinput[:outputFrames]
    yinput = yinput[:outputFrames]    
    
    
    # Initialize the plot
    fig1 = plt.figure()
    data = np.array([xinput,yinput])
    line, = plt.plot([], [], )
    line2, = plt.plot([], [],'ro')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    # Define the update function
    def update_line(time, data, line, line2):
        
        # Get the current data        
        currentData = data[...,:(time+1)]
        
        # Update the background line
        line.set_data(currentData)

        # Get the Current xy point
        [x,y] = currentData[:,-1]
        # update te second line
        line2.set_data(x,y)
        
        return line, line2
    
    interval = 1000/fps
    # interval = 50
    
    line_ani = animation.FuncAnimation(fig1, update_line, outputFrames, 
                                       fargs=(data, line, line2), 
                                       interval=interval, blit=True)
    return line_ani
    

def AnimateCyclicXYCurves(Curve1, Curve2, NFrames = 120, fps = 24, 
                          peakDist = 10, Xbound = [], Ybound = []):
    """
    This functions plots two curves, allowing us to compare experimental and 
    non-experiemntal.


    Parameters
    ----------
    Curve1 : TYPE
        the first curve to animate.
    Curve2 : TYPE
        the second curve to animate.
    NFrames : TYPE, optional
        Number of frames between cycles in the animation. The default is 48.
    fps : TYPE, optional
        The number of frames per second. The default is 24.
    peakDist : TYPE, optional
        This is used to find X direction reversals. We look for peaks within 
        this distance. The default is 10.
    Xbound : [xmin, xmax], optional
        A custome bound on the graphs xlimits. The default is [].
    Ybound : [xmin, xmax], optional
        A custome bound on the graphs ylimits. The default is [].


    """
    
    # Detect reversal points
    curve1Indicies = D.GetCycleIndicies(Curve1[:,0], Curve1[:,1], peakDist = 10)
    curve2Indicies = D.GetCycleIndicies(Curve2[:,0], Curve2[:,1], peakDist = 20)
    
    # Define the number of cycles, i.e. the number of indexes in the range
    Ncycles = len(curve1Indicies) - 1
    Ncycles_2 = len(curve2Indicies) - 1
    
    if Ncycles!=Ncycles_2:
        raise Exception('The experiment and Analysis have a different number of cycles')
    
    Nsteps = Ncycles*NFrames
    
    # Initialize animation curve
    animationCurve1 = np.zeros([Nsteps,2])
    animationCurve2 = np.zeros([Nsteps,2])
    
    # Create the animation curve for the
    for ii in range (Ncycles):
    
        [Ex,Ey] = D.GetCycleSubVector(Curve1[:,0] , Curve1[:,1], curve1Indicies[ii], curve1Indicies[ii+1], NFrames)
        [Ax,Ay] = D.GetCycleSubVector(Curve2[:,0] , Curve2[:,1], curve2Indicies[ii], curve2Indicies[ii+1], NFrames)
        
        animationCurve1[ii*NFrames:(ii+1)*NFrames,0] = Ex
        animationCurve1[ii*NFrames:(ii+1)*NFrames,1] = Ey

        animationCurve2[ii*NFrames:(ii+1)*NFrames,0] = Ax
        animationCurve2[ii*NFrames:(ii+1)*NFrames,1] = Ay
    
    
    if Xbound == []:
        xmin = 1.1*np.min([animationCurve1[:,0],animationCurve2[:,0]])
        xmax = 1.1*np.max([animationCurve1[:,0],animationCurve2[:,0]])
    else:
        xmin = Xbound[0]       
        xmax = Xbound[1]
    if Ybound == []:
        ymin = 1.1*np.min([animationCurve1[:,1],animationCurve2[:,1]])  
        ymax = 1.1*np.max([animationCurve1[:,1],animationCurve2[:,1]])        
    else:
        ymin = Ybound[0]       
        ymax = Ybound[1] 
    
    animationCurve1 =animationCurve1.T
    animationCurve2 =animationCurve2.T
    
    outputFrames = Nsteps
    
    # Initialize the plot
    fig1 = plt.figure()
    line, = plt.plot([], [], )
    line2, = plt.plot([], [],'bo')
    line3, = plt.plot([], [], )
    line4, = plt.plot([], [],'ro')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    # Define the update function
    def update_line(time, animationCurve1, animationCurve2, line, line2, line3, line4):
        
        
        currentData1 = animationCurve1[...,:(time+1)]
        currentData2 = animationCurve2[...,:(time+1)]

        line.set_data(currentData1)
        [x1,y1] = currentData1[:,-1]
        line2.set_data(x1,y1)
        
        line3.set_data(currentData2)
        [x2,y2] = currentData2[:,-1]
        line4.set_data(x2,y2) 
        
        return line, line2, line3, line4
    
    ani = animation.FuncAnimation(fig1, update_line, outputFrames, 
                                  fargs=(animationCurve1, animationCurve2, line, 
                                         line2, line3, line4), 
                                  interval=1000/fps, blit=True)
    
    return ani


def PlotFiberResponse(FiberName):
    
    FiberData  = np.loadtxt(FiberName,delimiter=' ')
    ypos = FiberData[:,1::6]
    Stress = FiberData[:,4::6]
    
    
    fig, = plt.plot(ypos[0,:],Stress[0,:])
    
    pass
