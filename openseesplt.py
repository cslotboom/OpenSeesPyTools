# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

# TODO Change import to import at op
# import sys
# C:\Users\Christian\.01_Thesis\Scripts\Python\ImageProcessing
# sys.path.append('C:\\Users\\Christian\\Anaconda3\\Lib\\site-packages\\openseespy')

import openseespy.opensees as op
# import openseespy.postprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib.animation as animation

import openseesplt.DataFunctions as df
# import openseesplt.StyleSheet
import os





# TODO: Create standarized "Draw" Function

# TODO: Seperate out earthquake enabling functions.









def getNodesCoords():
    """
    Returns
    -------
    nodeCoords : nxm array
        Returns the an array with the node ID, followed by node coordinate for 
        each node in the model
        n is the number of nodes
        m is the number of dimensions + 1        
        
    """
    # Get the Node tags
    nodeTags = op.getNodeTags()
    sizeN = len(nodeTags)
    
    # Get the number of Dimensions
    sizeM = op.nodeCoord(nodeTags[0])
    
    nodeCoords = np.zeros([sizeN, sizeM + 1])
    
    for ii in range(sizeN):
        # Get Coordinants
        tag = nodeTags[ii]
        Ncord = op.nodeCoord(tag)
        nodeCoords[ii,0] = tag
        nodeCoords[ii,1:] = Ncord
        
    return nodeCoords
    

def printNodes(FileName,delim=',', FileFormat='%.4e'):
    """
    This function saves the Node Coordinates of an active model
    
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
    This function gets the element connectivity for an active model
    
    Returns
    -------
    elementConnectivity1d : Array
        Returns an array with the elmenet tag as well as the nodes used to 
        define the element.
    """
    ElementTags = np.array(op.getEleTags())
    sizeE = len(ElementTags)

    elementConnectivity1d = np.zeros([sizeE,3],dtype = int)

    # Get the Nodes associated with each element
    index1d = 0
    
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

    
def getElementConnectivity2D():
    """
    This function gets the element connectivity for an active model
    
    Returns
    -------
    elementConnectivity1d : TYPE
        Returns an array with the elmenet tag as well as the nodes used to 
        define the element.
    """
    # Get Element tags
    ElementTags = np.array(op.getEleTags())
    sizeE = len(ElementTags)

    # Define Connectivity Matrix
    elementConnectivity2d = np.zeros([sizeE,5],dtype = int)

    # Get the Nodes associated with each element
    index2d = 0
    
    # Plot the elements in the model
    for ii in range(sizeE):
                
        # Get the nodes
        element = int(ElementTags[ii])
        nodes = op.eleNodes(element)
        
        # If the element has only two nodes.
        if len(nodes) == 4:            
            # Store the element
            elementConnectivity2d[index2d, 0] = element
            elementConnectivity2d[index2d, 1:] = nodes
            
            # Incriment Index
            index2d += 1
        elif len(nodes) == 3:
            pass
        
    return elementConnectivity2d
        
def printElements(FileName, delim = ',', fmt ='%.5i' ):
    """
    This saves gets the element connectivity for an active model to a file


    Parameters
    ----------
    FileName : string
        It's the file name.
    delim : TYPE, optional
        The format numbers will be stored in. The default is '%.5i'.
    fmt : TYPE, optional
        The format numbers will be stored in. The default is '%.5i'.

    Returns
    -------
    None.

    """
    
    outputCoords = getElementConnectivity1D()
    np.savetxt(FileName, outputCoords, delimiter = delim,fmt = '%.5i')


def DisplayWithEvents():
    """
    Plots the model background, and allows for the user to specific events.

    """
    
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
    curve1Indicies = df.GetCycleIndicies(Curve1[:,0], Curve1[:,1], peakDist = 10)
    curve2Indicies = df.GetCycleIndicies(Curve2[:,0], Curve2[:,1], peakDist = 20)
    
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
    
        [Ex,Ey] = df.GetCycleSubVector(Curve1[:,0] , Curve1[:,1], curve1Indicies[ii], curve1Indicies[ii+1], NFrames)
        [Ax,Ay] = df.GetCycleSubVector(Curve2[:,0] , Curve2[:,1], curve2Indicies[ii], curve2Indicies[ii+1], NFrames)
        
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


def GetEQDis2D(DispFileDir, dtFrames, saveFile = True, outputNameT = 'TimeOut',
               outputNameX = 'DispOutX', outputNameY = 'DispOutY'):
    """
    This function processes information from a 2D earthquake displacement file
    It's assumed that the input file records the x,y, theta, of all nodes.
    
    If 

    Parameters
    ----------
    DispFileDir : str
        The input file .
    dtFrames : float
        The desired time interval between animation frames.
    saveFile : str, optional
        If this is toggled on, a file will be saved. The default is True.
    outputNameT : str, optional
        Name for the output time file. The default is 'TimeOut'.
    outputNameX : str, optional
        Name for the output X displacement file. The default is 'DispOutX'.
    outputNameY : str, optional
        Name for the output Y displacement file. The default is 'DispOutY'.

    Returns
    -------
    TimeAnimation : array
        The output time array.
    dx : array
        The output x displacement of all nodes.
    dy : array
        The output y displacement of all nodes.

    """
    
    import openseesplt.DataFunctions as D    
    # read the files using numpy
    DisplacementData = np.loadtxt(DispFileDir,dtype ='float32', delimiter=' ',)
    
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    timeEarthquake = DisplacementData[:, 0]
    
    # This will need to be changed based on the amount of elements
    N_Time = len(timeEarthquake)
    
    # Organize displacement data
    dxEarthquake = DisplacementData[:, 1::3]
    dyEarthquake = DisplacementData[:, 2::3]
    
    # Plot structure
    ## this could possibly be a function
    
    # Create variables to store
    Tmax = np.max(timeEarthquake)
    Tmin = np.min(timeEarthquake)
    TimeAnimation = np.arange(Tmin,Tmax,dtFrames)
    Ntime = len(TimeAnimation)
    N_nodes = len(DisplacementData[0, 1::3])
    
    dx = np.zeros([Ntime,N_nodes])
    dy = np.zeros([Ntime,N_nodes])
    
    # Shift the data into a common frame.
    counter = 0
    for ii in range(N_nodes):
        if np.floor(10*ii/N_nodes) > counter:
            print('The processing is ', (counter + 1)*10, ' percent complete.')
            counter +=1
            
        dx[:,ii] = D.ShiftDataFrame(timeEarthquake,dxEarthquake[:,ii],TimeAnimation)
        dy[:,ii] = D.ShiftDataFrame(timeEarthquake,dyEarthquake[:,ii],TimeAnimation)
    
    if saveFile == True:
        np.savetxt(outputNameT + '.csv',TimeAnimation, delimiter = ',')
        np.savetxt(outputNameX + '.csv',dx, delimiter = ',')
        np.savetxt(outputNameY + '.csv',dy, delimiter = ',')
   
    return TimeAnimation, dx, dy    


def EQAnimation2D(dt, dx, dy, NodeFileName, ElementFileName, Scale = 1, fps = 24, 
                FrameInterval = 0, skipFrame =1, timeScale = 1):
    """
    This function animates a 2D earthquake. Only supports 1D elements, i.e. 
    beams.
    
    It's unlikely that the animation will actually run at the desired fps.
    Having a real time fps is dubious at best.

    Parameters
    ----------
    dt : 1D array
        The input time steps.
    dx : 1D array
        The input x displacement values for every node at each timestep.
    dy : 1D array
        The input y displacement values for every node at each timestep.
    NodeFileName : Str
        Name of the input node information file.
    ElementFileName : Str
        Name of the input element connectivity file.
    Scale : positive float, optional
        The scale on the xy displacements. The default is 1.
    fps : TYPE, optional
        The frames per second to be displayed. These values are dubious at best
        The default is 24.
    FrameInterval : The interval between frames to be used, optional
        DESCRIPTION. The default is 0.
    skipFrame : TYPE, optional
        DESCRIPTION. The default is 1.
    timeScale : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    
    """
    This function animates an earthquake, given a set of input files.
    

    
    """
       
    
    # =============================================================================
    # Get the data
    # =============================================================================
    
    # Scale on displacement
    dtFrames  = dt[1]
    Ntime = len(dt)
    
    # Create the directory names and import the necessary file(s)
    BaseDirectory = os.getcwd()
    
    # Get Displacement file Names
    NodeDirectory = "%s\%s" % (BaseDirectory, NodeFileName)
    ElementDirectory = "%s\%s" % (BaseDirectory, ElementFileName)
    
    # read the files using numpy
    NodeCoords = np.loadtxt(NodeDirectory, delimiter=',')
    Elements = np.loadtxt(ElementDirectory, delimiter=',')
    
    
    # names of each node
    Node_Labels = NodeCoords[:, 0]
    
    # Starting position of each node
    x0 = NodeCoords[:, 1]
    y0 = NodeCoords[:, 2]
    
    # Amount of each node
    N_nodes = len(x0)
    
    # This will need to be changed based on the amount of elements
    N_ele = len(Elements)
    
    # Create mapping of x and y node values to their labels
    xy_labels = {}
    for ii in range(N_nodes):
        xy_labels[Node_Labels[ii]] = [x0[ii], y0[ii]]
    
        
    # Get bounds
    dxMax = np.max(np.abs(dx))*Scale
    dyMax = np.max(np.abs(dy))*Scale
    
    xmax = 1.05*np.max(x0 + dxMax)
    xmin = 1.05*np.min(x0 - dxMax)
    ymax = 1.05*np.max(y0 + dyMax)
    ymin = 1.05*np.min(y0 - dyMax)
    MaxValue = np.max(np.abs([xmax,xmin,ymax,ymin]))
    
    ybound = ymax*(ymax/MaxValue)
    xbound = xmax*(xmax/MaxValue)        
    
    
    # ========================================================================
    # Initialize Plots
    # ========================================================================
    
    #Create figure and axis object
    [fig,ax] = plt.subplots(facecolor='black', figsize = (xbound,ybound))
    # fig.figsize((xbound,ybound))
    ax = plt.subplot(111,frameon=False)
    # ax = plt.subplot(frameon=False)
    
    # Initialize plots - this might not be necessary
    PlotNodes, = plt.plot([0], [np.sin(0)], 'b')
    PlotElement, = plt.plot([0], [np.sin(0)], 'b')
    

    
    #ax.x_lim()
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    
    # Add Text
    time_text = ax.text(0.95, 0.01, '', verticalalignment='bottom', 
                        horizontalalignment='right', transform=ax.transAxes,color='white')
    
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
        FrameInterval = dtFrames*1000/timeScale
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
        
        # update Text
        time_text.set_text(round(dtFrames*ii,1))
            
        return NodePlot, PlotElements
    
    steps = np.arange(0,Ntime)
    # dtFrames
    #ani = animation.FuncAnimation(fig, animate, steps, interval=50,blit=True, repeat=True)
    ani = animation.FuncAnimation(fig, animate, steps, interval = FrameInterval)
    
    return ani



def intializePlot(x0, y0, Node_Labels, xy_labels, Elements):
    """
    This functions creates a plot background with a specific style. It's mainly
    used by the animation functions, and not users.

    Parameters
    ----------
    x0 : TYPE
        DESCRIPTION.
    y0 : TYPE
        DESCRIPTION.
    Node_Labels : TYPE
        DESCRIPTION.
    xy_labels : TYPE
        DESCRIPTION.
    Elements : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : Figure object
        The output figure object.
    ax : Axis object
        The output axis object for the plot.
    PlotElements : array of element objects
        The array of structural elements repsresented as lines.
    NodePlot : TYPE
        The nodes of the plot.

    """
    
    
    # Get bounds
    xmax = max(abs(x0))
    ymax = max(abs(y0))
    MaxValue = 1.1*max(xmax,ymax)
    
    # #Normalized y
    # ybound = ymax*(ymax/MaxValue)
    # xbound = xmax*(xmax/MaxValue)
    
    #Create figure and axis object
    [fig,ax] = plt.subplots(facecolor='lightgrey')
    ax = plt.subplot(111,frameon=False)
    
    # Initialize plots - this might not be necessary
    PlotNodes, = plt.plot([0], [np.sin(0)], 'b')
    PlotElement, = plt.plot([0], [np.sin(0)], 'b')

    #ax.x_lim()
    ax.set_ylim(0,MaxValue)
    ax.set_xlim(-MaxValue,MaxValue)
    plt.xticks([])
    plt.yticks([])
    
    PlotElements = []
    N_ele = len(Elements)
    # plot elements
    for ii in range(N_ele):
        TempNodes = [Elements[ii, 1], Elements[ii, 2]]
        Ncord1 = xy_labels[TempNodes[0]]
        Ncord2 = xy_labels[TempNodes[1]]
        x_cords = [Ncord1[0], Ncord2[0]]
        y_cords = [Ncord1[1], Ncord2[1]]
        PlotElement, = plt.plot(x_cords, y_cords, 'w--', linewidth=1)
        PlotElements.append(PlotElement)
    
    # plot nodes
    NodePlot, = ax.plot(x0, y0, 'b.', linewidth=0, label=Node_Labels, markersize=1)

    return fig, ax, PlotElements, NodePlot


# =============================================================================
# 
# =============================================================================

def GetNodesAndElements(NodeFileName, ElementFileName):
    
    # read the files using numpy
    NodeCoords = np.loadtxt(NodeFileName, delimiter=',')
    # Elements = np.loadtxt(ElementFileName, delimiter=',')
    
    # names of each node
    Node_Labels = NodeCoords[:, 0]
    
    # Starting position of each node
    x0 = NodeCoords[:, 1]
    y0 = NodeCoords[:, 2]
    
    # Amount of each node
    N_nodes = len(x0)
    
    # Create mapping of x and y node values to their labels
    xy_labels = {}
    for ii in range(N_nodes):
        xy_labels[Node_Labels[ii]] = [x0[ii], y0[ii]]


def GetAllDisp(DispFileName):

    # read the files using numpy
    DisplacementData = np.loadtxt(DispFileName,dtype ='float32', delimiter=' ',)
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    timeEarthquake = DisplacementData[:, 0]
    
    # Organize displacement data
    dxEarthquake = DisplacementData[:, 1::3]
    dyEarthquake = DisplacementData[:, 2::3]
    
    return timeEarthquake, dxEarthquake, dyEarthquake
    
# =============================================================================
# 
# =============================================================================




def PlotDisp2D(DispFileName,NodeFileName,ElementFileName, Scale = 1, loadStep = -1):
    """
    This function plots reads data from input files then displays the displacement 
    displacement state of an object. Rotations are not included.
    
    Displacement files, Node files, and Element files, and  must be in the 
    standard format.

    Parameters
    ----------
    DispFileName : string
        Name of the displacement file to plot.
    NodeFileName : string
        A standard list of .
    ElementFileName : string
        The file name of the .
    Scale : float, optional
        A scale factor on displacements. The default is 1.
    loadStep : int, optional
        The load step to plot displacements. The default is the final step.

    Returns
    -------
    None.

    """
    
    # TODO: add more flexible inputs
    # Some users don't use files to store data.
    # inputs would could be a file name, or a series of vectors.
    
    
    # read the files using numpy
    DisplacementData = np.loadtxt(DispFileName,dtype ='float32', delimiter=' ',)
    NodeCoords = np.loadtxt(NodeFileName, delimiter=',')
    Elements = np.loadtxt(ElementFileName, delimiter=',')
    
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    # Orgnize the displacement data, seperating the x and components
    dx = DisplacementData[loadStep, 1::3]
    dy = DisplacementData[loadStep, 2::3]    
    
    
    # names of each node
    Node_Labels = NodeCoords[:, 0]
    
    # Starting position of each node
    x0 = NodeCoords[:, 1]
    y0 = NodeCoords[:, 2]
    
    # Amount of nodes and elements
    N_nodes = len(x0)
    N_ele = len(Elements)
    
    # Create mapping of x and y node values to their labels
    xy0_labels = {}
    for ii in range(N_nodes):
        xy0_labels[Node_Labels[ii]] = [x0[ii], y0[ii]]
    
    # Initialize Plot
    fig, ax, PlotElements, NodePlot =  intializePlot(x0, y0, Node_Labels, xy0_labels, Elements)    


    # Update Plots with displacement data
    x = x0 + Scale*dx
    y = y0 + Scale*dy
    
    # NodePlot.set_xdata(x) 
    # NodePlot.set_ydata(y) 
    
    
    # xy label database
    xy_labels = {}
    for jj in range(N_nodes):
        xy_labels[Node_Labels[jj]] = [x[jj], y[jj]]
    
    
    DispPlotElements = []
    N_ele = len(Elements)
    # plot elements
    for ii in range(N_ele):
        TempNodes = [Elements[ii, 1], Elements[ii, 2]]
        Ncord1 = xy_labels[TempNodes[0]]
        Ncord2 = xy_labels[TempNodes[1]]
        x_cords = [Ncord1[0], Ncord2[0]]
        y_cords = [Ncord1[1], Ncord2[1]]
        DispPlotElement, = plt.plot(x_cords, y_cords, 'w')
        DispPlotElements.append(DispPlotElement)
    
    # plot nodes
    DispNodePlot, = ax.plot(x, y, 'r.', linewidth=0, label=Node_Labels)    
    
    # DispPlotElements = []
    # for jj in range(N_ele):
    #     # Get the node number for the first and second node connected by the element
    #     TempNodes = np.array([Elements[jj,1] , Elements[jj,2]])
    #     # Get the coordinants associated with each node
    #     Ncord1 = xy_labels[TempNodes[0]]
    #     Ncord2 = xy_labels[TempNodes[1]]
        
    #     tempx = [Ncord1[0], Ncord2[0]]
    #     tempy = [Ncord1[1], Ncord2[1]]
    #     # Draw Line Between nodes    
    #     PlotElements[jj].set_xdata(tempx)
    #     PlotElements[jj].set_ydata(tempy)
        
    plt.show()
    
    
    

def PlotFiberResponse(FiberName):
    
    FiberData  = np.loadtxt(FiberName,delimiter=' ')
    ypos = FiberData[:,1::6]
    Stress = FiberData[:,4::6]
    
    
    fig, = plt.plot(ypos[0,:],Stress[0,:])
    
    
    
    pass
    # Get Fiber Locations
    # Get stresses
    # Get Stiffness


# =============================================================================
# Depriciated stuff
# =============================================================================








def ChonkyEQAnimation(DispFileName, NodeFileName, ElementFileName, 
              Scale = 2, fps = 24, FrameInterval = 0, skipFrame =1):
    
    """
    This function animates an earthquake, given a set of input files.
    
    It's chonky because it has to re-process the data every time. It would 
    be better to have one function that process the data, and another that
    plots the data.
    
    """
    
    import openseesplt.DataFunctions as D
    
    # It's chonky because it has to re-process the data every time. It would 
    # be better to have one function that process the data, and another that
    # plots the data
    
    
    # =============================================================================
    # Get the data
    # =============================================================================
    
    # Scale on displacement

    
    # Create the directory names and import the necessary file(s)
    BaseDirectory = os.getcwd()
    
    # Get Displacement file Names
    DisplacementDirectory = "%s\%s" % (BaseDirectory, DispFileName)
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
    # N_Time = len(timeEarthquake)
    
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
    
    # #Normalized y
    # ybound = ymax*(ymax/MaxValue)
    # xbound = xmax*(xmax/MaxValue)
    
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
        
        print(ii)
        
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
    print(steps)
    
    # dtFrames
    #ani = animation.FuncAnimation(fig, animate, steps, interval=50,blit=True, repeat=True)
    ani = animation.FuncAnimation(fig, animate, steps, interval = FrameInterval)
    # ani = animation.FuncAnimation(fig, animate, steps, interval = 1, repeat=True)
    
    return ani







