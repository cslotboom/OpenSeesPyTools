# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:34:02 2020

@author: Christian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def PlotXYFile(fileName_x, fileName_y='', column_x=1, column_y=1, 
               skipStart = 0, skipEnd = 0, rFactor=1, outputFrames=0, fps = 24, 
               Xbound = [],Ybound = []):

    """Parameters
    fileName_x: str
        The name of the file to read x data from. 
    fileName_y: str
        The name of the file to read y data from. If there is no path 
        specified, the x file name will be used as well.
    column_x: int
        The column in the x data file data is read from.
    column_y: int
        The column in the y data file data is read from.
    skipStart: int
        If specified, this many datapoints will be skipped from the data start.
    skipStart: int
        If specified, this many frames will be skipped at the analysis start.
    rFactor: int
        If specified, only every "x" frames will be reduced by this factor
    outputFrames: int
        The number of frames to be included after all other reductions. If the
        reduced number of frames is less than this value, no change is made.
    fps: int
        Number of animation frames to be displayed per second
    
    """

    # Get the x data
    InputData_x = np.loadtxt(fileName_x,delimiter=',')    
    xinput = InputData_x[:,column_x]
    
    # Get the y data
    if fileName_y == '':
        fileName_y = fileName_x
    InputData_y = np.loadtxt(fileName_y,delimiter=',')    
    yinput = InputData_y[:,column_y]      
    
    PlotXY(xinput, yinput, skipStart, skipEnd, rFactor, 
           outputFrames, fps, Xbound,Ybound)



def PlotXY(xinput, yinput, skipStart = 0, skipEnd = 0, rFactor=1, 
           outputFrames=0, fps = 24, Xbound = [],Ybound = []):

    """Parameters
    xinput: numpy array
        The x data to animate.
    yinput: numpy array
        The y data to animate.
    skipStart: int
        If specified, this many datapoints will be skipped from the data start.
    skipStart: int
        If specified, this many frames will be skipped at the analysis start.
    rFactor: int
        If specified, only every "x" frames will be reduced by this factor
    outputFrames: int
        The number of frames to be included after all other reductions. If the
        reduced number of frames is less than this value, no change is made.
    fps: int
        Number of animation frames to be displayed per second
    
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
        
    if Xbound == []:
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
        
        
        # lastPoint = np.array([CurrentData_x[-1],CCurrentData_y[-1])
        currentData = data[...,:(time+1)*2]
        line.set_data(currentData)
        # print(currentData)    
        # [x,y] = print(currentData[:,-1])
        [x,y] = currentData[:,-1]
        # line2.set_data(currentData[:,-1],currentData[:,-1])
        line2.set_data(x,y)
        
        # plt.plot(currentData[:,-2], currentData[:,-1])
        # line.set_data(data[...,:time])    
        
        return line, line2
    
    interval = 1000/fps
    # interval = 50
    
    line_ani = animation.FuncAnimation(fig1, update_line, outputFrames, 
                                       fargs=(data, line, line2), 
                                       interval=interval, blit=True)



FileName = "TS6_Experiment_PT_Force_1_2.5.csv"
PlotFileXY(FileName,FileName, 0, 1, skipStart = 5600, outputFrames = 1000, rFactor = 2, fps = 24)

# line_ani.save('lines.mp4', writer=writer)
# lastPoint = TS2_Experiment_PT_Force.csv







