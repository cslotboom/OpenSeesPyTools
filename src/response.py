# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

import openseespy.opensees as op

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

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


def PlotFiberResponse(FiberName, LoadStep):
    
    fiberData  = np.loadtxt(FiberName,delimiter=' ')
    FiberYPosition = fiberData[:,1::5]
    FiberStress = fiberData[:,4::5]
    
    fig, ax = plt.subplots()
    line = plt.plot(FiberYPosition[LoadStep,:], FiberStress[LoadStep,:])
    
    return fig, ax



def Animatefiber2D(fiberYPosition, fiberResponse, skipStart = 0, skipEnd = 0, rFactor=1, 
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
    if len(fiberYPosition) != len(fiberResponse):
        raise Exception('Lengths of input vectors unequal')
    
    # If end data is not being skipped, use the full vector length.
    if skipEnd ==0:
        skipEnd = len(fiberYPosition)
    
    
    # Set up bounds based on data from 
    if Xbound == []:
        xmin = 1.1*np.min(fiberYPosition)
        xmax = 1.1*np.max(fiberYPosition)
    else:
        xmin = Xbound[0]       
        xmax = Xbound[1]
    
    if Ybound == []:
        ymin = 1.1*np.min(fiberResponse)  
        ymax = 1.1*np.max(fiberResponse)        
    else:
        ymin = Ybound[0]       
        ymax = Ybound[1]          
    
    
    # Remove unecessary data
    xinputs = fiberYPosition[skipStart:skipEnd, :]
    yinputs = fiberResponse[skipStart:skipEnd, :]

    # Reduce the data if the user specifies
    if rFactor != 1:
        xinputs = xinputs[::rFactor, :]
        yinputs = yinputs[::rFactor, :]
    
    # If the Frames isn't specified, use the length of the reduced vector.
    if outputFrames == 0:
        outputFrames = len(xinputs[:, 0])
    else:
        outputFrames = min(outputFrames,len(xinputs[:, 0]))
    
    # Get the final output frames. X doesn't change
    xinputs = xinputs[:outputFrames, :]
    yinputs = yinputs[:outputFrames, :]    
    xinput = xinputs[0,:]
    
    # Initialize the plot
    fig, ax = plt.subplots()
    line, = ax.plot(xinput, yinputs[0,:])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    print(xmin)
    
    Frames = np.arange(0, outputFrames)
    FrameStart = int(Frames[0])
    FrameEnd = int(Frames[-1])
    
    # Slider Location and size relative to plot
    # [x, y, xsize, ysize]
    axSlider = plt.axes([0.25, .03, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Frame', FrameStart, FrameEnd, valinit=FrameStart, valfmt = '%d')
    
    # Animation controls
    global is_manual
    is_manual = False # True if user has taken control of the animation   
    
    def on_click(event):
        # Check where the click happened
        (xm,ym),(xM,yM) = plotSlider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            global is_manual
            is_manual=False        
    
    # Define the update function
    # def update_line(time, xinput, yinputs, line):
    def update_line_slider(time):
        global is_manual
        is_manual=True

        time = int(time)
        # Get the current data        
        y = yinputs[time,:]
        
        # Update the background line
        line.set_data(xinput, y)
        
        fig.canvas.draw_idle()    
        
        return line,
    
    
    def update_plot(ii):
    
        # If the control is manual, we don't change the plot    
        global is_manual
        if is_manual:
            return line,
       
        # Find the close timeStep and plot that
        CurrentFrame = int(np.floor(plotSlider.val))
        CurrentFrame += 1
        if CurrentFrame >= FrameEnd:
            CurrentFrame = 0
        
        # Update the slider
        plotSlider.set_val(CurrentFrame)
        is_manual = False # the above line called update_slider, so we need to reset this
        return line,  
    
    
    plotSlider.on_changed(update_line_slider)
    
    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)    
    
    interval = 1000/fps
    
    line_ani = animation.FuncAnimation(fig, update_plot, outputFrames, 
                                       # fargs=(xinput, yinputs, line), 
                                       interval=interval)
    return line_ani




def AnimateFiber2DFile(fiberFileName, skipStart = 0, skipEnd = 0, rFactor=1, 
           outputFrames=0, fps = 24, Xbound = [],Ybound = []):
    
    fiberData  = np.loadtxt(fiberFileName,delimiter=' ')
    fiberYPosition = fiberData[:,1::5]
    fiberStress = fiberData[:,4::5]    
    
    ani = AnimateFiber2D(fiberYPosition, fiberStress, skipStart, skipEnd, rFactor, outputFrames, fps, Xbound, Ybound)
    
    return ani
    
    
