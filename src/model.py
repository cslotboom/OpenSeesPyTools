# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:01:49 2020

@author: Christian
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D # Needed for 3D do not remove
from matplotlib.widgets import Slider
import numpy as np
import openseespy.opensees as op
import os
from math import asin


import openseespytools.data as D 
import openseespytools.stylesheets as Style


""" The standard displacement file is assumed to have ALL displacements no
rotation.
    nodes: 
        The node list in standard format
	[Node1tag, node1X, node1Y]
	[Node2tag, node2X, node2Y]
	
	[NodeNtag, nodeNX, nodeNY]
	
	
    elements: 1D list
        The elements list in standard format.
	[element1, element2, ... , elementN]
	where
	elementi = [elementiTag, elementiNode1, ... , elementiNodeN]
    

"""


def checkNodeConnectivity():
    """
    This functions checks if each node in a active model is connected to an 
    element. 
    
    Note this does not mean that the model is okay! The model still might have
    an internal to element connectivity problem, i.e. not all element DOF have
    a stiffness.

    Returns
    -------
    check : boolean
        Returns true or false depending on if all nodes are connected to 
        something.

    """
    # Get node and element tags, define sets for the node and element
    nodeTags = set(op.getNodeTags())
    eleTags = op.getEleTags()
    eleNodes = set()
    
    # Check the node connectivity
    for ele in eleTags:
        tempElementNodes = op.eleNodes(ele)
        
        # Check if the elements nodes are part of the element node set.
        for node in tempElementNodes:
            # If they aren't, add them to the set
            if node not in eleNodes:
                eleNodes.add(node)
        
    # Check if both sets are equal to eachother
    if nodeTags == eleNodes:
        check = True
    else:
        check = False
           
    return check

def getNodesandElements():
    """
    This function returns the nodes and elments for an active model, in a 
    standardized format. The OpenSees model must be active in order for the 
    function to work.
    
    Returns
    -------
    nodes : 2dArray
        An array of all nodes in the model.
        Returns nodes in the shape:
        [Nodes, 3] in 2d and [Nodes, 4]
        For each node the information is tored as follows:
        [NodeID, x, y] or [NodeID, x, y, z]
    elements : Array 
        An list of all elements in. Each entry in the list is it's own'
        [element1, element2,...],   element1 = [element#, node1, node2,...]
    """
   
    # Get nodes and elements
    nodeList = op.getNodeTags()
    eleList = op.getEleTags()   
    
    # Check Number of dimensions and intialize variables
    ndm = len(op.nodeCoord(nodeList[0]))
    Nnodes = len(nodeList)
    nodes = np.zeros([Nnodes, ndm + 1])
    
    # Get Node list
    for ii, node in enumerate(nodeList):
        nodes[ii,0] = node
        nodes[ii,1:] = op.nodeCoord(nodeList[ii])           
    
    Nele = len(eleList)
    elements = [None]*Nele
    
    # Generate the element list by looping through all emenemts
    for ii, ele in enumerate(eleList):
        tempNodes = op.eleNodes(ele)
        
        tempNnodes = len(tempNodes)
        tempEle = np.zeros(tempNnodes + 1)
        
        tempEle[0] = int(ele)
        tempEle[1:] = tempNodes
        
        elements[ii] = tempEle       
    
    return nodes, elements

def getModeShapeData(modeNumber):
     
    # Get nodes and elements
    nodeList = op.getNodeTags()
     
    # Check Number of dimensions and intialize variables
    ndm = len(op.nodeCoord(nodeList[0]))
    Nnodes = len(nodeList)
    modeshape = np.zeros([Nnodes, ndm + 1])
    
    op.wipeAnalysis()
    eigenVal = op.eigen(modeNumber)
    Tn = 4*asin(1.0)/(eigenVal[modeNumber-1])**0.5
         
    for ii, node in enumerate(nodeList):
         modeshape[ii,0] = node
         tempData = op.nodeEigenvector(nodeList[ii], modeNumber)
         modeshape[ii,1:] = tempData[0:ndm]
         

        
    return modeshape, Tn

# =============================================================================
# Enabling functions
# =============================================================================

    
def getAnimationDisp(OutputDatabase, dtFrames, ndm, InputFileName = 'NodeDisp_All.out',   outputName = 'NodeDisp_Ani', saveFile = True,):
    """
    This function prepars an input displacement  for animation.
    Often the input file for displacement is very large, and takes a long time
    to read. Using this fuction will reduce the input animation time, and
    allow for data to be accessed more quickly.
    
    It's assumed that the input file records the x,y (or x,y,z) of all nodes.
    The time in the dispalcement file is shifted into the domain 
    of the animiation.
        

    Parameters
    ----------
    DispFileDir : str
        The input displacement file name .
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
    timeAni : array
        The output time array.
    deltaAni : 
        The output xy or xyz displacement of all nodes in animation time.


    """    

    # read the files using numpy
    timeSteps, nodeDisp = OutputDatabase.readNodeDispData(InputFileName)
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    timeEarthquake = timeSteps
           
    deltaEQ = nodeDisp
    # Create variables to store animation time data
    Tmax = np.max(timeEarthquake)
    Tmin = np.min(timeEarthquake)
    timeAni = np.arange(Tmin,Tmax,dtFrames)
    
    Ntime = len(timeAni)
    N_nodes = len(nodeDisp[0,:,0])
    
    # Define an array that has the xy or xyz information for animation over all time
    deltaAni = np.zeros([Ntime, N_nodes, ndm])

    # Define a counter, this will be used to keep track of how long. It can take
    # a long time for big models!
    counter = 0
    
    for ii in range(ndm):               # For each coordinat (x,y) or (xyz):
        for jj in range(N_nodes):       # for each node:
            
            # Keep track of how many iterations have been complete
            if np.floor(10*jj*(ii+1)/(N_nodes*(ndm))) > counter:
                print('The processing is ', (counter + 1)*10, ' percent complete.')
                counter +=1
            
            # Get the node earthquake displacement for the dimension 'n'
            NodeEQDisp = deltaEQ[:, jj, ii]
            
            # Shift the displacement into the animation displacement frame
            deltaAni[:,jj,ii] = D.ShiftDataFrame(timeEarthquake, NodeEQDisp, timeAni)
    
   
    # Save the file information if it is requested.
    if saveFile == True:
        outputDisp = np.zeros([Ntime, N_nodes*ndm + 1])
        outputDisp[:,0] = timeAni
        for ii in range(ndm):
            outputDisp[:,(ii+1)::ndm] = deltaAni[:,:,ii]
            
        outputDir = os.path.join(OutputDatabase.LoadCaseDir, outputName)
        # OutputDatabase
        fmt = OutputDatabase.fmt
        delim = OutputDatabase.delim
        ftype = OutputDatabase.ftype
        
        np.savetxt(outputDir + ftype, outputDisp, delimiter = delim, fmt = fmt)
    
    return timeAni, deltaAni    

def _getSubSurface(NodeList,  ax, Style):
    """   
    Creates and returns the sub-surface objects for a node list.
    The input list of nodes corespond the the vertices of a quadralateral 

    Parameters
    ----------
    NodeList : The input list of nodes for the sub surface
    
    fig : the input figure object
        
    ax : the input axis object

    Returns
    -------
    tempLine : matplotlib object
        The matplotlib line objects.
    tempSurface : matplotlib object
        The matplotlib surface objects.

    """
    aNode = NodeList[0]
    bNode = NodeList[1]
    cNode = NodeList[2]
    dNode = NodeList[3]
    
    tempx = np.array([aNode[0], bNode[0], cNode[0], dNode[0]])
    tempy = np.array([aNode[1], bNode[1], cNode[1], dNode[1]])
    tempz = np.array([aNode[2], bNode[2], cNode[2], dNode[2]])
    
    # Store line information   
    tempSurfacex = np.array([[aNode[0],dNode[0]], [bNode[0],cNode[0]]])
    tempSurfacey = np.array([[aNode[1],dNode[1]], [bNode[1],cNode[1]]])
    tempSurfacez = np.array([[aNode[2],dNode[2]], [bNode[2],cNode[2]]])
    
    # Create objects
    tempLine, = ax.plot(tempx, tempy, tempz, 'w', **Style.ele_solid_line)
    tempSurface = ax.plot_surface(tempSurfacex, tempSurfacey, tempSurfacez, **Style.ele_solid)

    return tempLine, tempSurface

def _getCubeSurf(Nodes, xyz_labels, ax, Style):
    """
    This functions plots the nodes and surfaces for a 8 node element, and 
    returns the objects

    Parameters
    ----------
    Nodes : TYPE
        The input nodes of the 8 node  element.
    xyz_labels : TYPE
        The xyz labels for all nodes in the .
    fig : TYPE
        The input figure.
    ax : TYPE
        The input axis.

    Returns
    -------
    tempLines : The lines objects defining the 8 node element

    tempSurfaces : The suface objects defining the 8 node element


    """
    tempLines = 6*[None]
    tempSurfaces = 6*[None]

    # 2D Planer four-node shell elements
    iNode = xyz_labels[Nodes[0]]
    jNode = xyz_labels[Nodes[1]]
    kNode = xyz_labels[Nodes[2]]
    lNode = xyz_labels[Nodes[3]]
    iiNode = xyz_labels[Nodes[4]]
    jjNode = xyz_labels[Nodes[5]]
    kkNode = xyz_labels[Nodes[6]]
    llNode = xyz_labels[Nodes[7]]
    
    #TODO test this code
    # [iNode, jNode, kNode, lNode, iiNode, jjNode, kkNode, llNode,] = [xyz_labels[*Nodes]]
    
    
    # For the 8D we need to draw 6 surfaces. The outline will be 
    # included in each surface plot - every line is drawn twice
    # as a result. There probably is a better way.
    
    # get the lines and surfaces for our element
    [tempLines[0], tempSurfaces[0]] = _getSubSurface([iNode, jNode, kNode, lNode],  ax, Style)
    [tempLines[1], tempSurfaces[1]] = _getSubSurface([iNode, jNode, jjNode, iiNode],  ax, Style)
    [tempLines[2], tempSurfaces[2]] = _getSubSurface([iiNode, jjNode, kkNode, llNode],  ax, Style)
    [tempLines[3], tempSurfaces[3]] = _getSubSurface([lNode, kNode, kkNode, llNode],  ax, Style)
    [tempLines[4], tempSurfaces[4]] = _getSubSurface([jNode, kNode, kkNode, jjNode],  ax, Style)
    [tempLines[5], tempSurfaces[5]] = _getSubSurface([iNode, lNode, llNode, iiNode],  ax, Style)

    return tempLines, tempSurfaces

# =============================================================================
# Plotting function enablers
# =============================================================================


def _update_Plot_Disp(nodes, elements, fig, ax, Style, DisplacementData = np.array([])):
    """
    This functions plots an image of the model in it's current diplacement
    state. If no displacement data is passed to the funtion, it plots the base
    model.
    It returns the plotted matplotlib objects, the object types will depend on
    if the domain is in 2D or 3D.

    Parameters
    ----------
    nodes : Array
        Node array in standard format
    elements : Ele list
        Element list in standard format
    fig : matplotlib fig object
        Maptlotlib figure object to update.
    ax : Axis object
        Matplotlib axis object to update.
    DisplacementData : [Ntimestep, Nnode * Ndm], optional
        The input displacement data to plot for each node. The format for input
        displacement is:
            [timestep1, node1x, node1y, ... nodeNx, nodeNy]
            [timestep2, node1x, node1y, ... nodeNx, nodeNy]
        The default is [], which plots the model without displacement
    scale : float, optional
        A scale factor on displacements. The default is 1.
    node_tags : Boolean, optional
        A flag that turns on or off node tags. The default is False.
    ele_tags : Boolean, optional
        A flag that turns on or off Element tags. The default is False.

    Returns
    -------
    figNodes : matplotlib object
        The node element plot object.
    figElements : list of matplotlib objects
        A list of the line elements in the model.
    figSurfaces : list of matplotlib objects
        A list of the surface elements in the model.
    figText : list of matplotlib objects
        The object for all lables in the array.

    """  
    
    
    # Amount of nodes and elements
    Nnode = len(nodes[:,0])
    Nele = len(elements)
    nodeLabels = nodes[:, 0]
    
    # Initialize an array if node tags are asked for
    if Style.node_tags == True:
        nodeTags = [None]*Nnode
    
    # Get intial node coordinates
    tempCoords = nodes[:,1:]
    ndm = len(tempCoords[0,:])
            
    # If displacemetns are not asked for, return an appriately sized array.
    if DisplacementData.size == 0:    
        DisplacementData = np.zeros([Nnode, ndm])
    
    # Get Plot nodes/elements
    Nele = len(elements)
    
    # Create an appriprately sized array if element tags are asked for.
    figTags = [None]*Nele
        
    # Find the number of surfaces
    Nsurf = len([ele for ele in elements if len(ele) >= 4])
    Nsurf8Node = len([ele for ele in elements if len(ele) == 9])
    
    # For each 8 node surface there are a total of 6 nodes, so we need
    # 5 additional surfaces for each 8Node element.
    figSurfaces = [None]*(Nsurf + Nsurf8Node*5)
    figLines = [None]*(Nele + Nsurf8Node*5)
    currentSurf = 0
    currentEle = 0
    
    # Plot the 2D
    if ndm == 2:
               
        x = nodes[:, 1] + DisplacementData[:,0]
        y = nodes[:, 2] + DisplacementData[:,1]          
        
        # xy label database for the current displacement
        xy_labels = {}
        for jj in range(Nnode):
            xy_labels[nodeLabels[jj]] = [x[jj], y[jj]]

        # plot elements with the current displacement
        for ii in range(Nele):
            
            tempEle = elements[currentEle]
            
            # Get element Nodes
            TempNodesLables = tempEle[0]
            TempNodes = tempEle[1:]
            
            # This is the xy coordinates of each node in the group
            TempNodeCoords = [xy_labels[node] for node in TempNodes] 
            coords_x = [xy[0] for xy in TempNodeCoords]
            coords_y = [xy[1] for xy in TempNodeCoords]
        
            # Plot element and store object
            if len(TempNodes) == 2:
                figLines[currentEle], = ax.plot(coords_x, coords_y, 'w', **Style.ele)
            
            if len(TempNodes) == 3 or len(TempNodes) == 4:
                figLines[currentEle], = ax.plot(coords_x, coords_y, 'w', **Style.ele_surf_line)
                [figSurfaces[currentSurf]] = ax.fill(coords_x, coords_y, **Style.ele_surf)

                currentSurf += 1
            
            currentEle += 1
            
            if Style.ele_tags == True:
                tagPosition = [np.average(coords_x), np.average(coords_y)]
                tag = str(int(TempNodesLables))
                tempTag = [*tagPosition, tag]
                figTags[ii] = ax.text(*tempTag, **Style.ele_tags_style) #label elements   
                
        # plot nodes
        # figNodes = ax.scatter(x, y, **node_style)    
        figNodes, = ax.plot(x, y, **Style.node)    
        
        if Style.node_tags == True:
            for ii in range(Nnode):
                # Store the node tag then create the tag object
                tag = str(int(nodes[ii,0]))
                nodeTags[ii] = ax.text(1.02*x[ii], 1.02*y[ii],tag, **Style.node_tags_style) #label elements           
        
        
    elif ndm == 3:
        
        # In 3D we need to unpack the element array
        elements = [*elements]
                       
        x = nodes[:, 1] + DisplacementData[:,0]
        y = nodes[:, 2] + DisplacementData[:,1]      
        z = nodes[:, 3] + DisplacementData[:,2]           
        
        # xyz label database for the current displacement
        xyz_labels = {}
        for jj in range(Nnode):
            xyz_labels[nodeLabels[jj]] = [x[jj], y[jj], z[jj]]
        
        # plot elements with the current displacement
        for ii in range(Nele):
            
            # Get elementNodes
            tempEle = elements[ii]
            tempEleTag = str(int(tempEle[0]))
            tempNodes = tempEle[1:]
            
            # This is the xy coordinates of each node in the group
            tempNodeCoords = [xyz_labels[node] for node in tempNodes] 
            tempx = [xyz[0] for xyz in tempNodeCoords]
            tempy = [xyz[1] for xyz in tempNodeCoords]
            tempz = [xyz[2] for xyz in tempNodeCoords]
            
            if len(tempNodes) == 2:
        
                # Plot element and store object
                figLines[currentEle], = ax.plot(tempx, tempy, tempz, 'w', **Style.ele)
            
                currentEle += 1
                
            if len(tempNodes) == 4:
                
                # Plot element and store object
                figLines[currentEle], = ax.plot(tempx, tempy, tempz, 'w', **Style.ele_surf_line)                
                
                tempVertsx = np.array([[tempx[0],tempx[3]], [tempx[1],tempx[2]]])
                tempVertsy = np.array([[tempy[0],tempy[3]], [tempy[1],tempy[2]]])
                tempVertsz = np.array([[tempz[0],tempz[3]], [tempz[1],tempz[2]]])                
                
                figSurfaces[currentSurf] = ax.plot_surface(tempVertsx, tempVertsy, tempVertsz, **Style.ele_surf)
                
                currentSurf += 1
                currentEle += 1
                
            if len(tempNodes) == 8:
                
                [tempeles, tempVertices] = _getCubeSurf(tempNodes, xyz_labels, ax, Style)
                
                # Store elements and surfaces
                for jj in range(6):
                    figLines[currentEle] = tempeles[jj]
                    figSurfaces[currentSurf] = tempVertices[jj]
                    currentSurf += 1
                    currentEle += 1
                    
            # currentEle += 1s
            # add text if it is asked for 
            if Style.ele_tags == True:
                tempCoords = [np.average(tempx), np.average(tempy), np.average(tempz)]
                figTags[ii] = ax.text(*tempCoords, tempEleTag, **Style.ele_tags_style) #label elements
                    
                    
        # GetNodes
        figNodes, = ax.plot(x, y,z, **Style.node)
        # figNodes = ax.scatter(x, y, z, **node_style)
        if Style.node_tags == True:
            for ii in range(Nnode):
                # Store the node tag then create the tag object
                tag = str(int(nodes[ii,0]))
                nodeTags[ii] = ax.text(1.02*x[ii], 1.02*y[ii], 1.02*z[ii], 
                                       tag, **Style.node_tags_style) #label elements  
    

    return figNodes, figLines, figSurfaces, figTags

def _setStandardViewport(fig, ax, Style, nodeCords, ndm, Disp = np.array([]) ):
    """
    This function sets the standard viewport size of a function, using the
    nodes as an input.
       
    Parameters
    ----------
    fig : matplotlib figure object
        The figure. This is passed just incase it's needed in the future.
    ax : matplotlib ax object
        The axis object to set the size of.
    nodes : array
        An array of the bounding node coordinants in the object. This can
        simply be the node coordinats, or it can be the node coordinats with 
        updated
    ndm : int
        The number of dimenions.

    Returns
    -------
    fig : TYPE
        The updated figure.
    ax : TYPE
        The updated axis.

    """
    
    # Adjust plot area, get the bounds on both x and y
    nodeMins = np.min(nodeCords, 0)
    nodeMaxs = np.max(nodeCords, 0)
    
    # Get the maximum displacement in each direction
    if len(Disp) != 0:
        
        # if it's the animation
        if len(Disp.shape) == 3:
            dispmax = np.max(np.abs(Disp), (0,1))

        # if it's regular displacement 
        else:
            dispmax = np.max(np.abs(Disp), 0)

        nodeMins = np.min(nodeCords, 0) - dispmax  
        nodeMaxs = np.max(nodeCords, 0) + dispmax  


    viewCenter = np.average([nodeMins, nodeMaxs], 0)
    viewDelta = 1.1*(abs(nodeMaxs - nodeMins))
    viewRange = max(nodeMaxs - nodeMins)
        
    if ndm == 2:
        ax.set_xlim(viewCenter[0] - viewDelta[0]/2, viewCenter[0] + viewDelta[0]/2)
        ax.set_ylim(viewCenter[1] - viewDelta[1]/2, viewCenter[1] + viewDelta[1]/2)       
        		
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    if ndm == 3:
        ax.set_xlim(viewCenter[0]-(viewRange/2), viewCenter[0]+(viewRange/2))
        ax.set_ylim(viewCenter[1]-(viewRange/2), viewCenter[1]+(viewRange/2))
        ax.set_zlim(viewCenter[2]-(viewRange/2), viewCenter[2]+(viewRange/2))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
	
    if Style.show_axis == False:
        plt.axis('off')
    
    return fig, ax

def _initializeFig(nodeCords, ndm, Disp = np.array([]) ):
    
    # set the maximum figure size
    maxFigSize = 8
    
    # Find the node 
    nodeMins = np.min(nodeCords, 0)
    nodeMaxs = np.max(nodeCords, 0)    
    
    # Get the maximum displacement in each direction
    if len(Disp) != 0:
        
        # if it's the animation
        if len(Disp.shape) == 3:
            dispmax = np.max(np.abs(Disp), (0,1))
            
        # if it's regular displacement 
        else:
            dispmax = np.max(np.abs(Disp), 0)

        nodeMaxs = np.min(nodeCords, 0) - dispmax   
        nodeMaxs = np.max(nodeCords, 0) + dispmax   
        
    # Find the difference between each node.
    nodeDelta = np.abs(nodeMaxs - nodeMins)
    dmax = np.max(nodeDelta[:2])
    
    # Set the figure size
    figsize = (maxFigSize*nodeDelta[0]/dmax , maxFigSize*nodeDelta[1]/dmax)
    
    # Initialize figure
    if ndm == 2:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1)
    elif ndm == 3:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')    
    
    return fig, ax

def _findModelData(OutputDatabase):
        
    try:
        nodes, elements = getNodesandElements()
        Model = True
    except:
        Model = False
        
    if Model == False:
        try:
            nodes, elements = OutputDatabase.readNodesandElements()
            Model = True
        except:
            Model = False
    
    if Model == False:
        raise Exception('No model or database found.')
    
    return nodes, elements

def _findModeData(OutputDatabase, modeNumber):
        # Get nodes and elements ( should this be a function?)
    try:
        # Here modeshape is a vector for every node, period is a single vector
        modeShape, modePeriods = getModeShapeData(modeNumber)
        Model = True
    except:
        Model = False
        
    if Model == False:
        try:
            modeShape, modePeriods = OutputDatabase.readModeShapeData(modeNumber)
            Model = True
        except:
            Model = False
    
    if Model == False:
        raise Exception('No model or database found.')
    
    return modeShape, modePeriods


def plot_model(OutputDatabase = None, CustomStyleFunction = None):
    """
    Plots an Model by first looking for an active model, the looking for a model database.

    Parameters
    ----------
    OutputDatabase : Database object, optional
        The database to be used to plot from. The default is None, where an
        active model is attempted to be plotted from instead.
        
    CustomStyleFunction : StyleSheet object, optional
        A custom style sheet to be used. The default is None.

    Returns
    -------
    fig : TYPE
        The output figure.
    ax : TYPE
        The axis of the ouput figure.

    """
    
    # Get Style Sheet
    if CustomStyleFunction == None:
        BasicStyle = Style.getStyle(Style.BasicStyleSheet)
    else:
        BasicStyle = Style.getStyle(CustomStyleFunction)
    
    # attempt to find data for the model
    nodes, elements = _findModelData(OutputDatabase)
    ndm = len(nodes[0,1:])        

    # Initialize model
    fig, ax = _initializeFig(nodes[:,1:], ndm)
    
    # Plot the first set of data
    _update_Plot_Disp(nodes, elements, fig, ax, BasicStyle)
    
    # Adjust viewport size
    _setStandardViewport(fig, ax, BasicStyle, nodes[:,1:], ndm)
    
    # Show the plot
    plt.show()
    
    return fig, ax


def plot_deformedshape(OutputDatabase, tstep = -1,   scale = 1,
                       CustomStyleFunction = None, CustomDispStyleFunction = None):
        
    """
    This function plots a the displacement of a model. It's assumed that node
    and element files are saved in the standard format.
    It's also assumed that a displlacement recorder file is in the directory
    and contains data in the standard format.


    Parameters
    ----------
    TimeStep : int
        The index of the load step to be plotted.
    scale : float, optional
        The scale on the displacements. The default is 1.
    node_tags : Boolean, optional
        A flag that is used to turn node tags on or off. The default is True.
    ele_tags : Boolean, optional
        A flag that is used to turn element tags on or off. The default is True.

    Returns
    -------
    None.

    """
    
    # attempt to find data for the model
    nodes, elements = _findModelData(OutputDatabase)
    
    # number of dimensions
    ndm = len(nodes[0,1:])
        
    # Get the displacements if there are any.
    timeSteps, nodeDisp = OutputDatabase.readNodeDispData()
    
    CurrentTimeStep = (np.abs(timeSteps - tstep)).argmin()
    if timeSteps[-1] < tstep:
        print("XX Warining: Time-step has exceeded maximum analysis time step XX")
        print("Using the last time-step.")
		
    # DeflectedNodeCoordArray = nodeArray[:,1:] + scale*Disp_nodeArray[int(jj),:,:]    
    
    currentDisp = nodeDisp[CurrentTimeStep,:,:]*scale
    printLine = "Deformation at time: " + str(round(timeSteps[CurrentTimeStep], 2))
    
    # Get Style Sheet for dispaced model
    if CustomDispStyleFunction == None:
        DispStyle = Style.getStyle(Style.BasicStyleSheet)
    else:
        DispStyle = Style.getStyle(CustomDispStyleFunction)

    # Get Style Sheet for static model
    if CustomStyleFunction == None:
        StaticStyle = Style.getStyle(Style.StaticStyleSheet)
    else:
        StaticStyle = Style.getStyle(CustomStyleFunction)
    
    # initialize figure
    fig, ax = _initializeFig(nodes[:,1:], ndm, currentDisp)
    
    # Plot basemodel if requested, then plot displacement
    if DispStyle.showUndeflected == True:
        _update_Plot_Disp(nodes, elements, fig, ax, StaticStyle)
    _update_Plot_Disp(nodes, elements, fig, ax, DispStyle,  currentDisp)

	# Adjust plot area.
    _setStandardViewport(fig, ax, DispStyle, nodes[:,1:], ndm, currentDisp)
    
    # add Text
    if DispStyle.summaryText == True:
        if ndm == 2:
            ax.text(0.10, 0.9, printLine, transform=ax.transAxes)
        elif ndm ==3:
            ax.text2D(0.10, 0.9, printLine, transform=ax.transAxes)
        plt.show()
    
    return fig,  ax


def plot_modeshape(OutputDatabase, modeNumber, scale = 1,
                   CustomStyleFunction = None, CustomDispStyleFunction = None):
    """
    This function plots a the displacement of a model. It's assumed that node
    and element files are saved in the standard format.
    It's also assumed that a displlacement recorder file is in the directory
    and contains data in the standard format.


    Parameters
    ----------
    TimeStep : int
        The index of the load step to be plotted.
    scale : float, optional
        The scale on the displacements. The default is 1.
    node_tags : Boolean, optional
        A flag that is used to turn node tags on or off. The default is True.
    ele_tags : Boolean, optional
        A flag that is used to turn element tags on or off. The default is True.

    Returns
    -------
    None.

    """
    
    # Error handling for nodes and elements
    nodes, elements = _findModelData(OutputDatabase)
    modeshape, modePeriod = _findModeData(OutputDatabase, modeNumber)
    
    # Scale Mode Shape
    modeshape[:,1:] = modeshape[:,1:]*scale
    
    # number of dimensions
    ndm = len(nodes[0,1:])
          
    # Get Style Sheet for dispaced model
    if CustomDispStyleFunction == None:
        DispStyle = Style.getStyle(Style.BasicStyleSheet)
    else:
        DispStyle = Style.getStyle(CustomDispStyleFunction)

    # Get Style Sheet for dispaced model
    if CustomStyleFunction == None:
        StaticStyle = Style.getStyle(Style.StaticStyleSheet)
    else:
        StaticStyle = Style.getStyle(CustomStyleFunction)
    
    # initialize figure
    fig, ax = _initializeFig(nodes[:,1:], ndm, modeshape[:,1:])
    
    # Plot basemodel if requested, then plot displacement
    if DispStyle.showUndeflected == True:
        _update_Plot_Disp(nodes, elements, fig, ax, StaticStyle)
    _update_Plot_Disp(nodes, elements, fig, ax, DispStyle,  modeshape[:,1:])

	# Adjust plot area.
    _setStandardViewport(fig, ax, DispStyle, nodes[:,1:], ndm)
	
    # add Text
    if DispStyle.summaryText == True:
        if ndm == 2:
            ax.text(0.10, 0.95, "Mode "+str(modeNumber), transform=ax.transAxes)
            ax.text(0.10, 0.90, "T = "+str("%.3f" % modePeriod)+" s", transform=ax.transAxes)    
        elif ndm ==3:
            ax.text2D(0.10, 0.95, "Mode "+str(modeNumber), transform=ax.transAxes)
            ax.text2D(0.10, 0.90, "T = "+str("%.3f" % modePeriod)+" s", transform=ax.transAxes)    
    
    plt.show()
    
    return fig, ax


def animate_deformedshape(OutputDatabase, dt, DispName = '', tStart = 0, tEnd = 0, Scale = 1, fps = 24, 
                          FrameInterval = 0, skipFrame =1, timeScale = 1,
                          CustomDispStyleFunction = None):
    """
    This defines the animation of an opensees model, given input data.
    
    For big models it's unlikely that the animation will actually run at the 
    desired fps in "real time". Matplotlib just isn't built for high fps 
    animation.

    Parameters
    ----------
    Model : string
        The name of the input model database.    
    LoadCase : string
        The name of the input loadcase.    
    dt : 1D array
        The time step between frames in the input file. The input file should
        have approximately the same number of time between each step or the
        animation will appear to speed up or slow down.
    tStart: float, optional
        The start time for animation. It can be approximate value and the program 
        will find the closest matching time step.
    tEnd: float, optional
        The end time for animation. It can be approximate value and the program 
        will find the closest matching time step.
    NodeFileName : Str
        Name of the input node information file.
    ElementFileName : Str
        Name of the input element connectivity file.
    Scale :  float, optional
        The scale on the xy/xyz displacements. The default is 1.
    fps : TYPE, optional
        The frames per second to be displayed. These values are dubious at best
        The default is 24.
    FrameInterval : float, optional
        The time interval between frames to be used. The default is 0.
    skipFrame : TYPE, optional
        DESCRIPTION. The default is 1.
    timeScale : TYPE, optional
        DESCRIPTION. The default is 1.
    Movie : str, optional 
        Name of the movie file if the user wants to save the animation as .mp4 file.

    Returns
    -------
    TYPE
        Earthquake animation.

    """
    
    # Get Style Sheet for dispaced model
    if CustomDispStyleFunction == None:
        DispStyle = Style.getStyle(Style.BasicStyleSheet)
        DispStyle.node_tags = False
        DispStyle.ele_tags = False
    else:
        DispStyle = Style.getStyle(CustomDispStyleFunction)
        
    # If no dispmacenet is given, read the default value. Otherwise look for a directory.
    nodes, elements = _findModelData(OutputDatabase)
    if DispName == '':
        time, Disp = OutputDatabase.readNodeDispData()
    else:
        time, Disp = OutputDatabase.readNodeDispData(DispName)

    Disp = Disp*Scale
    
    # Find number of nodes, elements, and time steps
    ndm = len(nodes[0,1:])
    Nnodes = len(nodes[:,0])
    Nele = len(elements)
    
    nodeLabels = nodes[:, 0]       

    # ========================================================================
    # Initialize Plots
    # ========================================================================
    
    # initialize figure
    fig, ax = _initializeFig(nodes[:,1:], ndm, Disp)    
    plt.subplots_adjust(bottom=.15) # Add extra space bellow graph
    
	# Adjust plot area.   
    _setStandardViewport(fig, ax, DispStyle, nodes[:,1:], ndm, Disp)
    
    EQObjects = _update_Plot_Disp(nodes, elements, fig, ax, DispStyle, Disp[0,:,:])
    [EqfigNodes, EqfigLines, EqfigSurfaces, EqfigText] = EQObjects

    # ========================================================================
    # Set animation frames
    # ========================================================================
   
    # Scale on displacement
    dtFrames  = 1/fps
    Ntime = len(Disp[:,0])
    Frames = np.arange(0,Ntime)
    framesTime = Frames*dt

    # If the interval is zero
    if FrameInterval == 0:
        FrameInterval = dtFrames*1000/timeScale
    else: 
        pass    
        
    FrameStart = Frames[0]
    FrameEnd = Frames[-1]
	
    if tStart != 0:
        index = (np.abs(time - tStart)).argmin()
        FrameStart = Frames[index]
	
    if tEnd != 0:
        if time[-1] < tEnd:
            print("XX Warining: tEnd has exceeded maximum analysis time step XX")
            print("XX tEnd has been set to final analysis time step XX")
        elif tEnd <= tStart:
            print("XX Input Warning: tEnd should be greater than tStart XX")
            print("XX tEnd has been set to final analysis time step XX")
        else:
            index = (np.abs(time - tEnd)).argmin()
            FrameEnd = Frames[index]

    aniFrames = FrameEnd-FrameStart  # Number of frames to be animated
	
    # ========================================================================
    # Begin animation
    # ========================================================================
       
    # Slider Location and size relative to plot
    # [x, y, xsize, ysize]
    axSlider = plt.axes([0.25, .03, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Time', framesTime[FrameStart], framesTime[FrameEnd], valinit=framesTime[FrameStart])
    
    # Animation controls
    global is_paused
    is_paused = False # True if user has taken control of the animation   
    
    def on_click(event):
        # Check where the click happened
        (xm,ym),(xM,yM) = plotSlider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it.
            return
        else:
            # Toggle on off based on clicking
            global is_paused
            if is_paused == True:
                is_paused=False
            elif is_paused == False:
                is_paused=True
                
    def animate2D_slider(Time):
        """
        The slider value is liked with the plot - we update the plot by updating
        the slider.
        """
        global is_paused
        is_paused=True
        # Convert time to frame
        TimeStep = (np.abs(framesTime - Time)).argmin()
               
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + Disp[TimeStep,:,:]
        
        # update node locations
        EqfigNodes.set_xdata(CurrentNodeCoords[:,0]) 
        EqfigNodes.set_ydata(CurrentNodeCoords[:,1])
           
        # Get new node mapping
        # I don't like doing this loop every time - there has to be a faster way
        xy_labels = {}
        for jj in range(Nnodes):
            xy_labels[nodeLabels[jj]] = CurrentNodeCoords[jj,:]
        
        # Define the surface
        SurfCounter = 0
        
        # update element locations
        for jj in range(Nele):
            # Get the node number for the first and second node connected by the element
            TempNodes = elements[jj][1:]
            # This is the xy coordinates of each node in the group
            TempNodeCoords = [xy_labels[node] for node in TempNodes] 
            coords_x = [xy[0] for xy in TempNodeCoords]
            coords_y = [xy[1] for xy in TempNodeCoords]
            
            # Update element lines    
            EqfigLines[jj].set_xdata(coords_x)
            EqfigLines[jj].set_ydata(coords_y)
            # Update the surface if necessary
            if 2 < len(TempNodes):
                tempxy = np.column_stack([coords_x, coords_y])
                EqfigSurfaces[SurfCounter].xy = tempxy
                SurfCounter += 1
               
        # redraw canvas while idle
        fig.canvas.draw_idle()    
            
        return EqfigNodes, EqfigLines, EqfigSurfaces, EqfigText

    def animate3D_slider(Time):
        
        global is_paused
        is_paused=True
        TimeStep = (np.abs(framesTime - Time)).argmin()
        
        # this is the most performance critical area of code
        
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + Disp[TimeStep,:,:]
        
        # update node locations
        EqfigNodes.set_data_3d(CurrentNodeCoords[:,0], CurrentNodeCoords[:,1], CurrentNodeCoords[:,2])
               
        # Get new node mapping
        # I don't like doing this loop every time - there has to be a faster way
        xyz_labels = {}
        for jj in range(Nnodes):
            xyz_labels[nodeLabels[jj]] = CurrentNodeCoords[jj,:]        
    
        SurfCounter = 0
            
        # update element locations
        for jj in range(Nele):
            # Get the node number for the first and second node connected by the element
            TempNodes = elements[jj][1:]
            # This is the xy coordinates of each node in the group
            TempNodeCoords = [xyz_labels[node] for node in TempNodes] 
            coords_x = [xyz[0] for xyz in TempNodeCoords]
            coords_y = [xyz[1] for xyz in TempNodeCoords]
            coords_z = [xyz[2] for xyz in TempNodeCoords]
            
            # Update element Plot    
            EqfigLines[jj].set_data_3d(coords_x, coords_y, coords_z)
            
            if len(TempNodes) > 2:
                # Update 3D surfaces
                tempVec = np.zeros([4,4])
                tempVec[0,:] = coords_x
                tempVec[1,:] = coords_y
                tempVec[2,:] = coords_z
                tempVec[3,:] = EqfigSurfaces[SurfCounter]._vec[3,:]
                EqfigSurfaces[SurfCounter]._vec = tempVec
                SurfCounter += 1
        
        # redraw canvas while idle
        fig.canvas.draw_idle()   

        return EqfigNodes, EqfigLines, EqfigSurfaces, EqfigText


    def update_plot(ii):
        # If the control is manual, we don't change the plot    
        global is_paused
        if is_paused:
            return EqfigNodes, EqfigLines, EqfigSurfaces, EqfigText
       
        # Find the close timeStep and plot that
        CurrentTime = plotSlider.val
        CurrentFrame = (np.abs(framesTime - CurrentTime)).argmin()

        CurrentFrame += 1
        if CurrentFrame >= FrameEnd:
            CurrentFrame = FrameStart
        
        # Update the slider
        plotSlider.set_val(framesTime[CurrentFrame])
        
        is_paused = False # the above line called update_slider, so we need to reset this
        return EqfigNodes, EqfigLines, EqfigSurfaces, EqfigText

    if ndm == 2:
        plotSlider.on_changed(animate2D_slider)
    elif ndm == 3:
        plotSlider.on_changed(animate3D_slider)
    
    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)

    ani = animation.FuncAnimation(fig, update_plot, aniFrames, interval = FrameInterval)

    plt.show()
    return ani


def _fiberInputHandling(InputType, LocalAxis):
    
    # Catch invalid input types
    if InputType not in ['stress', 'strain']:
        raise Exception('Invalid input type. Valid Entries are "stress" and "strain"')
    
    # Catch invalid Direction types
    if LocalAxis not in ['z', 'y']:
        raise Exception('Invalid LocalAxis type. Valid Entries are "z" and "y"')

    if InputType == 'stress':
        responseIndex = 3
        axisYlabel = "Fiber Stress"
    if InputType == 'strain':
        responseIndex = 4
        axisYlabel = "Fiber Strain"
    
    if LocalAxis == 'z':
        axisIndex = 1
        axisXlabel = "Local z value"
    if LocalAxis == 'y':
        axisIndex = 0
        axisXlabel = "Local y value"
    
    return responseIndex, axisIndex, axisXlabel, axisYlabel
    

def plot_fiberResponse2D(OutputDatabase, element, section, LocalAxis = 'y', 
                         InputType = 'stress', tstep = -1):
    """
    

    Parameters
    ----------
    Model : string
        The name of the input model database.    
    LoadCase : string
        The name of the input loadcase.    
    element : int
        The input element to be plotted
    section : TYPE
        The section in the input element to be plotted.
    LocalAxis : TYPE, optional
        The local axis to be plotted on the figures x axis. 
        The default is 'y', 'z' is also possible.
    InputType : TYPE, optional
        The quantity to be plotted. The default is 'stress', 'strain' is 
        also possible
    tstep : TYPE, optional
        The time step to be plotted. The program will find the closed time 
        step to the input value. The default is -1.

    """
    DispStyle = Style.getStyle(Style.BasicStyleSheet)
    
    # Catch errors
    outputs = _fiberInputHandling(InputType, LocalAxis)
    [responseIndex, axisIndex, axisXlabel, axisYlabel] = [*outputs]
            
    timeSteps, fiberData  = OutputDatabase.readFiberData2D(element, section)
    
    # find the appropriate time step
    if tstep == -1:
        LoadStep = -1
        printLine = "Final deformed shape"
    else:
        LoadStep = (np.abs(timeSteps - tstep)).argmin()			# index closest to the time step requested.
        if timeSteps[-1] < tstep:
            print("XX Warining: Time-Step has exceeded maximum analysis time step XX")
        printLine = 'Fibre '+  InputType + " at time: " + str(round(timeSteps[LoadStep], 2))
            

    fiberYPosition = fiberData[LoadStep,axisIndex::5]
    fiberResponse  = fiberData[LoadStep, responseIndex::5]
    
    # Sort indexes so they appear in an appropraiate location
    sortedIndexes = np.argsort(fiberYPosition)
    fibrePositionSorted = fiberYPosition[sortedIndexes]
    fibreResponseSorted = fiberResponse[sortedIndexes]
    
    
    fig, ax = plt.subplots()
    Xline = ax.plot([fibrePositionSorted[0],fibrePositionSorted[-1]],[0, 0], c ='black', linewidth = 0.5)
    line = ax.plot(fibrePositionSorted, fibreResponseSorted)
    
    xyinput = np.array([fibrePositionSorted,fibreResponseSorted]).T
    
    _setStandardViewport(fig, ax, DispStyle, xyinput, 2)
    
    ax.set_ylabel(axisYlabel)  
    ax.set_xlabel(axisXlabel)    
        
    plt.show()
    return fig, ax
    

def animate_fiberResponse2D(OutputDatabase, element, section,LocalAxis = 'y', InputType = 'stress', skipStart = 0, 
                            skipEnd = 0, rFactor=1, outputFrames=0, fps = 24, Xbound = [], Ybound = []):
    """
    Parameters
    ----------
    Model : string
        The name of the input model database.    
    LoadCase : string
        The name of the input loadcase.    
    element : int
        The input element to be plotted
    section : TYPE
        The section in the input element to be plotted.
    LocalAxis : string, optional
        The local axis to be plotted on the figures x axis. 
        The default is 'y', 'z' is also possible.
    InputType : string, optional
        The quantity 
    skipStart : int, optional
        If specified, this many datapoints will be skipped from the analysis
        data set, before reductions.
        The default is 0, or no reduction
    skipEnd : int, optional
        If specified, this many frames will be skipped at the end of the 
        analysis dataset, before reduction. The default is 0, or no reduction.
    rFactor : int, optional
        If specified, only every "x" frames will be plotted. e.g. x = 2, every 
        other frame is shown.
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
    
    # Catch errors
    outputs = _fiberInputHandling(InputType, LocalAxis)
    [responseIndex, axisIndex, axisXlabel, axisYlabel] = [*outputs]
    
    timeSteps, fiberData  = OutputDatabase.readFiberData2D(element, section)
                

    fiberYPosition = fiberData[:,axisIndex::5]
    fiberResponse  = fiberData[:, responseIndex::5]
    
    # Sort indexes so they appear in an appropraiate location
    sortedIndexes = np.argsort(fiberYPosition[0,:])
    fibrePositionSorted = fiberYPosition[:,sortedIndexes]
    fibreResponseSorted = fiberResponse[:,sortedIndexes]    
    
    # If end data is not being skipped, use the full vector length.
    if skipEnd ==0:
        skipEnd = len(fiberYPosition)    
    
    # Set up bounds based on data from 
    if Xbound == []:
        xmin = 1.1*np.min(fibrePositionSorted)
        xmax = 1.1*np.max(fibrePositionSorted)
    else:
        xmin = Xbound[0]       
        xmax = Xbound[1]
    
    if Ybound == []:
        ymin = 1.1*np.min(fibreResponseSorted)  
        ymax = 1.1*np.max(fibreResponseSorted)        
    else:
        ymin = Ybound[0]       
        ymax = Ybound[1]          
    
    # Remove unecessary data
    xinputs = fibrePositionSorted[skipStart:skipEnd, :]
    yinputs = fibreResponseSorted[skipStart:skipEnd, :]

    # Reduce the data if the user specifies
    if rFactor != 1:
        xinputs = xinputs[::rFactor, :]
        yinputs = yinputs[::rFactor, :]    
        timeSteps = timeSteps[::rFactor]
    
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
    plt.subplots_adjust(bottom=.15) # Add extra space bellow graph
    
    line, = ax.plot(xinput, yinputs[0,:])
    Xline = ax.plot([fibrePositionSorted[0,0],fibrePositionSorted[0,-1]], [0, 0], c ='black', linewidth = 0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
        
    ax.set_ylabel(axisYlabel)  
    ax.set_xlabel(axisXlabel)    

    Frames = np.arange(0, outputFrames)
    FrameStart = int(Frames[0])
    FrameEnd = int(Frames[-1])
    
    # Slider Location and size relative to plot
    # [x, y, xsize, ysize]
    axSlider = plt.axes([0.25, .03, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Time', timeSteps[FrameStart], timeSteps[FrameEnd], valinit=timeSteps[FrameStart])

    # Animation controls
    global is_paused
    is_paused = False # True if user has taken control of the animation   
    
    def on_click(event):
        # Check where the click happened
        (xm,ym),(xM,yM) = plotSlider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # Toggle on/off based on click
            global is_paused
            if is_paused == True:
                is_paused=False
            elif is_paused == False:
                is_paused=True       
    
    # Define the update function
    def update_line_slider(Time):
        global is_paused
        is_paused=True

        TimeStep = (np.abs(timeSteps - Time)).argmin()
        # Get the current data        
        y = yinputs[TimeStep,:]
        
        # Update the background line
        line.set_data(xinput, y)
        fig.canvas.draw_idle()    
        
        return line,
    
    
    def update_plot(ii):
    
        # If the control is manual, we don't change the plot    
        global is_paused
        if is_paused:
            return line,
       
        # Find the close timeStep and plot that
        CurrentTime = plotSlider.val
        CurrentFrame = (np.abs(timeSteps - CurrentTime)).argmin()

        CurrentFrame += 1
        if CurrentFrame >= FrameEnd:
            CurrentFrame = FrameStart
        
        # Update the slider
        plotSlider.set_val(timeSteps[CurrentFrame])        
        
        # Update the slider
        is_paused = False # the above line called update_slider, so we need to reset this
        return line,  
    
    
    plotSlider.on_changed(update_line_slider)
    
    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)    
    
    interval = 1000/fps
    
    line_ani = animation.FuncAnimation(fig, update_plot, outputFrames, 
                                       # fargs=(xinput, yinputs, line), 
                                       interval=interval)
									   
    plt.show()
    return line_ani





# =============================================================================
# Depiciated
# =============================================================================

def depricatedError():
    print('This functions is depricated, in the future no error will return')

def plot_active_model(CustomStyleFunction = None):
    depricatedError()
    raise Exception('This function is deprciated. Instead use plot_model')
    


def saveNodesandElements(nodeName = 'Nodes', eleName = 'Elements', 
                          delim = ',', fmt = '%.5e', ftype = '.out'):
    """   
        Depriciated

    """
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use database.saveNodesandElements()')
    
def readNodesandElements(nodeName = 'Nodes', eleName = 'Elements', delim = ',', 
                         dtype ='float32', ftype = '.out'):
    """   
        Depriciated

    """
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use database.readNodesandElements()')


def readDisp(DispName = 'All_Disp', outputDir = 'vis',  ftype = '.out', 
             delim = ' ', dtype ='float32'):
    """   
        Depriciated

    """
    depricatedError()
    # Load the relevant information
    raise Exception('This function is deprciated. Instead use make a database and use database.readNodeDispData()')

def plot_model_disp(LoadStep,    scale = 1, dispName = 'All_Disp', 
                    CustomStyleFunction = None, CustomDispStyleFunction = None,
                    nodeName = 'Nodes', eleName = 'Elements', delim = ',', 
                    dtype = 'float32', ftype = '.out'):
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use plot_deformedShape()')
    
    
    
    
def AnimateDisp(dt, deltaAni, nodes, elements, Scale = 1, 
                fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1, 
                CustomStyleFunction = None):
    """
    Depricated
    """
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use animate_deformedshape()')


def getDispAnimation(dt, deltaAni, nodes, elements, fig, ax, Style, Scale = 1, 
                     fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1):
    """
    Depricated
    """
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use animate_deformedshape()')
    
    
    
def AnimateDispSlider(dt, deltaAni, nodes, elements, Scale = 1, 
                fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1, 
                CustomStyleFunction = None):
    
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use animate_deformedshape()')    
    
    
def getDispAnimationSlider(dt, deltaAni, nodes, elements, fig, ax, Style, Scale = 1, 
                           fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1):
 
    depricatedError()
    raise Exception('This function is deprciated. Instead use make a database and use animate_deformedshape()')   