 # -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019

@author: Christian
"""

# TODO Change import to import at op
import sys
# C:\Users\Christian\.01_Thesis\Scripts\Python\ImageProcessing
sys.path.append('C:\\Users\\Christian\\Anaconda3\\Lib\\site-packages\\openseespy')
import openseespy.opensees as op


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib.animation as animation

import DataFunctions as df
import os




def getNodesCoords():
    """

    Returns
    -------
    nodeCoords : Array
        Returns the an array with the node ID, followed by node coordinate for 
        each node in the model

    """
    nodeTags = op.getNodeTags()
    sizeN = len(nodeTags)
    
    nodeCoords = np.zeros([sizeN,3])
    
    for ii in range(sizeN):
        # Get Coordinants
        tag = nodeTags[ii]
        Ncord = op.nodeCoord(tag)
        nodeCoords[ii,0] = tag
        nodeCoords[ii,1:] = Ncord
        
    return nodeCoords
    

def printNodes(FileName,delim=',', FileFormat='%.4e'):
    """

    Parameters
    ----------
    FileName : str
        The name of the file to be output.
    delim : TYPE, optional
        The desired delimiter for the output file. The default is ','.
    FileFormat : TYPE, optional
        The desired number format for the output file. The default is '%.4e'.

    Returns
    -------
    None.

    """
    outputCoords = getNodesCoords()
    np.savetxt(FileName, outputCoords, delimiter=delim,fmt=FileFormat)


def getElementConnectivity1D():
    """
    

    Returns
    -------
    elementConnectivity1d : TYPE
        Returns an array with the elmenet tag as well as the nodes used to 
        define the element.

    """
    ElementTags = np.array(op.getEleTags())
    sizeE = len(ElementTags)

    elementConnectivity1d = np.zeros([sizeE,3],dtype = int)
    elementConnectivity2d = np.zeros([sizeE,5],dtype = int)

    # Get the Nodes associated with each element
    index1d = 0
    index2d = 0
    
    # Plot the elements in the model
    for ii in range(sizeE):
                
        # Get the nodes
        element = int(ElementTags[ii])
        nodes = op.eleNodes(element)
        
        # If the element has only two nodes.
        if len(nodes) == 2:            
            # Store the element
            elementConnectivity1d[index1d, 0] = element
            elementConnectivity1d[index1d, 1:] = nodes
            
            # Incriment Index
            index1d += 1
        elif len(nodes) == 4:
            pass
        
    return elementConnectivity1d
        
        

def printElements(FileName,fmt ='%.5i' ):
    outputCoords = getElementConnectivity1D()
    np.savetxt(FileName, outputCoords, delimiter=",",fmt='%.5i')


def TestDisplay():
    
    def pickEvent(event):
        chosenObject = event.artist
        ind = event.ind
        tag = ElementTags[ind]
        print(tag)
    
    # Set viewport visual style
    bg_colour = 'lightgrey' # background colour
    pl.rc('font', family='Monospace', size=10) # set font for labels
    node_style = {'color':'black', 'marker':'.', 'markersize':10} # nodes
    ele_style = {'color':'black', 'linewidth':1, 'linestyle':'-'} # elements
    # axis_style = {'color':'grey', 'linewidth':1, 'linestyle':'--'} # x=0, y=0 lines
    offset = 0.05 #offset for text
    # # 2D
    # bc_style = {'color':'black', 'markeredgewidth':1, 'markersize':9,
    #             'fillstyle':'none'} # node translation fixity (boundary conditions)
    # bcrot_style = {'color':'black', 'markeredgewidth':1, 'markersize':10,
    #                'fillstyle':'none'} # node rotation fixity (boundary conditions)
    # # 3D
    # azimuth = -50 #degrees
    # elevation = 20 #degrees
    # bc_style3d = {'length':0.3, 'arrow_length_ratio':0.5, 'colors':'black'}
    # bcrot_style3d = {}
    
    ChartName="Sample Chart"
    
    ## Create Figure
    
    
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect=1, frameon=False)
    fig.set_facecolor(bg_colour)
    # fig.text(0.01, 0.01, ', '.join(tclfiles), va='bottom', ha='left', color='grey', fontweight='bold') # display file
    fig.text(0.01, 0.01, ChartName, va='bottom', ha='left', color='grey', fontweight='bold') # display file
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92)
    
    # Plot Element Tags
    NodeTags = op.getNodeTags()
    SizeN = len(NodeTags)
    for ii in range(SizeN):
        # Get Coordinants
        Ncord = op.nodeCoord(NodeTags[ii])
        
        # Plot the point
        ax.plot(Ncord[0], Ncord[1], linewidth=0, **node_style)
        ax.text(Ncord[0]+offset, Ncord[1]+offset,'N'+str(NodeTags[ii]), fontweight='bold') #label node
    
    # Get Element
    ElementTags = np.array(op.getEleTags())
    SizeE = len(ElementTags)
    
    # Plot the elements in the model
    for ii in range(SizeE):
        # Get the Nodes associated with each element
        tag = int(ElementTags[ii])
        TempNodes = op.eleNodes(tag)
        
        # Get the coordinants associated with each node
        Ncord1 = op.nodeCoord(TempNodes[0])
        Ncord2 = op.nodeCoord(TempNodes[1])
        
        # Draw Line Between nodes
        if Ncord1 and Ncord2: # make sure both nodes exist before using them
            ax.plot((Ncord1[0], Ncord2[0]), (Ncord1[1], Ncord2[1]), marker='', **ele_style)
            ax.text(offset+(Ncord1[0]+Ncord2[0])/2, offset+(Ncord1[1]+Ncord2[1])/2, 'E'+str(ElementTags[ii])) #label element
        
    fig.canvas.mpl_connect('pick_event',pickEvent)



def AnimateXYFile(fileName_x, fileName_y='', column_x=1, column_y=1, 
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
    
    AnimateXY(xinput, yinput, skipStart, skipEnd, rFactor, 
           outputFrames, fps, Xbound,Ybound)



def AnimateXY(xinput, yinput, skipStart = 0, skipEnd = 0, rFactor=1, 
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






def AnimateTime(timeInput, yInput, skipStartTime = 0, skipEndTime = 0, 
                rFactor=1, outputFrames=0, fps = 24, Xbound = [],Ybound = []):

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
    if len(timeInput) != len(yInput):
        raise Exception('Lengths of input vectors unequal')
    



def AnimateCyclicXYCurves(Curve1, Curve2, Npoint = 48, fps = 24, Xbound = [],
                          Ybound = []):
    """
    Most appropraite for two force based analyses
    

    Parameters
    ----------
    Curve1 : The first curve to animate
        DESCRIPTION.
    Curve2 : TYPE
        DESCRIPTION.
    NPoint : TYPE, optional
        DESCRIPTION. The default is 48.
    fps : TYPE, optional
        DESCRIPTION. The default is 24.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """

    
    # Detect reversal points
    curve1Indicies = df.GetCycleIndicies(Curve1[:,0], Curve1[:,1], peakDist = 10,CreatePlot = 'y')
    curve2Indicies = df.GetCycleIndicies(Curve2[:,0], Curve2[:,1], peakDist = 20)
    
    
    NIndex = len(curve1Indicies) - 1
    NIndex_2 = len(curve2Indicies) - 1
    
    if NIndex!=NIndex_2:
        raise Exception('The experiment and Analysis have a different number of cycles')
    
    Nsteps = NIndex*Npoint
    
    animationCurve1 = np.zeros([Nsteps,2])
    animationCurve2 = np.zeros([Nsteps,2])
    
    for ii in range (NIndex):
    
        [Ex,Ey] = df.GetCycleSubVector(Curve1[:,0] , Curve1[:,1], curve1Indicies[ii], curve1Indicies[ii+1], Npoint)
        [Ax,Ay] = df.GetCycleSubVector(Curve2[:,0] , Curve2[:,1], curve2Indicies[ii], curve2Indicies[ii+1], Npoint)
        
        animationCurve1[ii*Npoint:(ii+1)*Npoint,0] = Ex
        animationCurve1[ii*Npoint:(ii+1)*Npoint,1] = Ey

        animationCurve2[ii*Npoint:(ii+1)*Npoint,0] = Ax
        animationCurve2[ii*Npoint:(ii+1)*Npoint,1] = Ay
    
    
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
        
        
        currentData1 = animationCurve1[...,:(time+1)*2]
        currentData2 = animationCurve2[...,:(time+1)*2]

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




    
def ChonkyEQAnimation(DisplacementFileName,NodeFileName,ElementFileName, 
              Scale = 2, fps = 24, FrameInterval = 0, skipFrame =1):
    import DataFunctions as D
    # =============================================================================
    # Get the data
    # =============================================================================
    
    # Scale on displacement

    
    # Create the directory names and import the necessary file(s)
    BaseDirectory = os.getcwd()
    
    # Get Displacement file Names
    DisplacementDirectory = "%s\%s" % (BaseDirectory, DisplacementFileName)
    NodeDirectory = "%s\%s" % (BaseDirectory, NodeFileName)
    ElementDirectory = "%s\%s" % (BaseDirectory, ElementFileName)
    
    # read the files using numpy
    DisplacementData = np.loadtxt(DisplacementDirectory,dtype ='float32', delimiter=' ',)
    NodeCoords = np.loadtxt(NodeDirectory, delimiter=',')
    Elements = np.loadtxt(ElementDirectory, delimiter=',')
    
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    timeEarthquake = DisplacementData[:, 0]
    
    # names of each node
    Node_Labels = NodeCoords[:, 0]
    
    # Starting position of each node
    x0 = NodeCoords[:, 1]
    y0 = NodeCoords[:, 2]
    
    # Amount of each node
    N_nodes = len(x0)
    
    # This will need to be changed based on the amount of elements
    N_ele = len(Elements)
    N_Time = len(timeEarthquake)
    
    # Create mapping of x and y node values to their labels
    xy_labels = {}
    for ii in range(N_nodes):
        xy_labels[Node_Labels[ii]] = [x0[ii], y0[ii]]
    
    # Organize displacement data
    dxEarthquake = DisplacementData[:, 1::3]
    dyEarthquake = DisplacementData[:, 2::3]
    
    # Plot structure
    ## this could possibly be a function
    
    # Create variables to store
    dtFrames = 1 / fps
    Tmax = np.max(timeEarthquake)
    Tmin = np.min(timeEarthquake)
    TimeAnimation = np.arange(Tmin,Tmax,dtFrames)
    Ntime = len(TimeAnimation)
    
    dx = np.zeros([Ntime,N_nodes])
    dy = np.zeros([Ntime,N_nodes])
    
    # Shift the data into a common frame.
    for ii in range(N_nodes):
        dx[:,ii] = D.ShiftDataFrame(timeEarthquake,dxEarthquake[:,ii],TimeAnimation)
        dy[:,ii] = D.ShiftDataFrame(timeEarthquake,dyEarthquake[:,ii],TimeAnimation)
    

    # =============================================================================
    # Initialize Plots
    # =============================================================================
    
    #Create figure and axis object
    [fig,ax] = plt.subplots(facecolor='black')
    ax = plt.subplot(111,frameon=False)
    
    # Initialize plots - this might not be necessary
    PlotNodes, = plt.plot([0], [np.sin(0)], 'b')
    PlotElement, = plt.plot([0], [np.sin(0)], 'b')
    
    # Get bounds
    xmax = max(abs(x0))
    ymax = max(abs(y0))
    MaxValue = max(xmax,ymax)
    
    #Normalized y
    ybound = ymax*(ymax/MaxValue)
    xbound = xmax*(xmax/MaxValue)
    
    #ax.x_lim()
    ax.set_ylim(0,MaxValue)
    ax.set_xlim(-MaxValue,MaxValue)
    
    PlotElements = []
    # plot elements
    for ii in range(N_ele):
        TempNodes = [Elements[ii, 1], Elements[ii, 2]]
        Ncord1 = xy_labels[TempNodes[0]]
        Ncord2 = xy_labels[TempNodes[1]]
        x_cords = [Ncord1[0], Ncord2[0]]
        y_cords = [Ncord1[1], Ncord2[1]]
        PlotElement, = plt.plot(x_cords, y_cords, 'w')
        PlotElements.append(PlotElement)
    
    # plot nodes
    NodePlot, = ax.plot(x0, y0, 'r.', linewidth=0, label=Node_Labels)
    
    # If the interval is zero
    if FrameInterval == 0:
        FrameInterval = dtFrames*1000
    else: 
        pass

    
    # =============================================================================
    # Animation
    # =============================================================================
    
    def animate(ii):
        
        # Update Plots
        x = x0 + Scale*dx[ii,:]
        y = y0 + Scale*dy[ii,:]
        
        NodePlot.set_xdata(x) 
        NodePlot.set_ydata(y) 
        
        # factor database
        xy_labels = {}
        for jj in range(N_nodes):
            xy_labels[Node_Labels[jj]] = [x[jj], y[jj]]
        
        for jj in range(N_ele):
            # Get the node number for the first and second node connected by the element
            TempNodes = np.array([Elements[jj,1] , Elements[jj,2]])
            # Get the coordinants associated with each node
            Ncord1 = xy_labels[TempNodes[0]]
            Ncord2 = xy_labels[TempNodes[1]]
            
            tempx = [Ncord1[0], Ncord2[0]]
            tempy = [Ncord1[1], Ncord2[1]]
            # Draw Line Between nodes    
            PlotElements[jj].set_xdata(tempx)
            PlotElements[jj].set_ydata(tempy)
        return NodePlot, PlotElements
    
    steps = np.arange(0,Ntime)
    # dtFrames
    #ani = animation.FuncAnimation(fig, animate, steps, interval=50,blit=True, repeat=True)
    ani = animation.FuncAnimation(fig, animate, steps, interval = FrameInterval)
    
    return ani







def PlotFiberResponse(FiberName):
    
    FiberData  = np.loadtxt(FiberName,delimiter=' ')
    ypos = FiberData[:,1::6]
    Stress = FiberData[:,4::6]
    
    
    fig, = plt.plot(ypos[0,:],Stress[0,:])
    
    
    
    pass
    # Get Fiber Locations
    # Get stresses
    # Get Stiffness






