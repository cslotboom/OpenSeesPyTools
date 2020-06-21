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
#TODO
# Change the name elements to lines for figure defintion



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

# =============================================================================
# Enabling functions
# =============================================================================

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

def saveNodesandElements(nodeName = 'Nodes', eleName = 'Elements', 
                          delim = ',', fmt = '%.5e', ftype = '.out'):
    """   
    This file saves the node and element information for the structure. 
    For each node information is saved in the following format:
        Nodes:    [NodeID, xcord, ycord] or [NodeID, xcord, ycord, zcord]
    
    For elements, the element is saved with the element connectivity. 
    A different file is created for each type of element
    each possible element type.
        Elements: [EleID, eleNode1, eleNode2, ... , eleNodeN]

    Parameters
    ----------
    nodeName : str, optional
        The name of the file to be saved. The default is 'Nodes'.
    eleName : str, optional
        The name of the . The default is 'Elements'.
    delim : str, optional
        The delimeter for the output file. The default is ','.
    fmt : str, optional
        the format of the file to be saved in. The default is '%.5e'.

    """
    

    # Read noades and elements
    nodes, elements = getNodesandElements()

    # Sort through the element arrays
    ele2Node = np.array([ele for ele in elements if len(ele) == 3])
    ele3Node = np.array([ele for ele in elements if len(ele) == 4])
    ele4Node = np.array([ele for ele in elements if len(ele) == 5])
    ele8Node = np.array([ele for ele in elements if len(ele) == 9])

    # SaveNodes
    np.savetxt(nodeName + ftype, nodes, delimiter = delim, fmt = fmt)
    
    
    #TODO 
    # Don't save empty files!!!
    
    elements = [ele2Node, ele3Node, ele4Node, ele8Node]
    eleLabels = ['_2Node', '_3Node', '_4Node', '_8Node']
    
    # Save element arrays
    for ii in range(4):
        if len(elements[ii]) != 0:
            np.savetxt(eleName + eleLabels[ii] + ftype, ele2Node, delimiter = delim, fmt = fmt)
    
def readNodesandElements(nodeName = 'Nodes', eleName = 'Elements', delim = ',', 
                         dtype ='float32', ftype = '.out'):
    """   
    This function reads input node/element information, assuming it is in the 
    standard format. 

    If outputDir == False, the base directory will be used.    
    
    Parameters
    ----------
    nodeName : str, optional
        The base name for the node file. It will be appended to include
        the file type. The default is 'Nodes.out'.
    eleName : str, optional
        The base nae for the element files. The default is 'Elements.out'.
    delim : str, optional
        The delimiter for files to be read. The default is ','.
    dtype : TYPE, optional
        The data type to read in. The default is 'float32'.

    Returns
    -------
    nodes : Array
        An output vector in standard format
    elements : List
        An output Element vector in standard format.
        elements = [ele1, ele2,..., elen], 
        ele1 = [element, node 1, node 2, ... , node n]

    """
        
    # Load Node information
    nodes = np.loadtxt(nodeName + ftype, dtype, delimiter = delim)
       
    # Define Element tags
    eleFileNames = [eleName + '_2Node' + ftype, eleName + '_3Node' + ftype,
                    eleName + '_4Node' + ftype, eleName + '_8Node' + ftype]
    
    # Populate an array with the input element information
    TempEle = [[]]*4
    for ii, FileName in enumerate(eleFileNames):
        if os.path.isfile(FileName):
            TempEle[ii] = np.loadtxt(FileName, dtype,  delimiter = delim)

    # define the final element array
    elements = [*TempEle[0],*TempEle[1],*TempEle[2],*TempEle[3]]

    # Check if any files were read
    if elements is []:
        raise Exception('No files were found!')

    
    return nodes, elements

def readDisp(DispName = 'All_Disp', outputDir = 'vis',  ftype = '.out', 
             delim = ' ', dtype ='float32'):
    """
    Checked!!
    
    This file reads data from a input file, assuming it is in standard format. 
    Standard format for displacement means the xy coordinates, or xyz 
    coordinates for each node is organized as follows:
        [node1x, node1y, node2x, node2y,... nodeNx, nodeNy]
    Each column will contain the displacements over time.


    Parameters
    ----------
    DispName : str, optional
        Base name of the Disp file. The default is 'All_Disp'.
    ftype: str, file extension, optional
        The file extension name for the input file
    delim : str, optional
        delimieter for the name of the Disp file. The default is ' '.
    dtype : dtype, optional
        Data type to read in the delimiter. The default is 'float32'.

    Returns
    -------
    Disp : TYPE
        The displacement of each node in the file.

    """
        
    # Load the relevant information
    Disp = np.loadtxt(DispName + ftype, delimiter = delim, dtype = dtype)

    
    return Disp

def getAnimationDisp(DispFileDir, dtFrames, ndm,  saveFile = True, 
                     outputNameT = 'TimeOut', outputNameX = 'DispOutX', 
                     outputNameY = 'DispOutY', outputNameZ = 'DispOutZ'):
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
    DisplacementData = readDisp(DispFileDir)
    
    # Get vectors of time data and base x and y locations
    # x0 and y0 are vectors with n columns, where n is the number of nodes
    timeEarthquake = DisplacementData[:, 0]
       
    # Organize displacement data
    if ndm == 2:
        dxEarthquake = DisplacementData[:, 1::2]
        dyEarthquake = DisplacementData[:, 2::2]
        deltaEQ = [dxEarthquake,dyEarthquake]
    if ndm == 3:
        dxEarthquake = DisplacementData[:, 1::3]
        dyEarthquake = DisplacementData[:, 2::3]    
        dzEarthquake = DisplacementData[:, 3::3]    
        deltaEQ = [dxEarthquake,dyEarthquake,dzEarthquake]
    
           
    # Create variables to store animation time data
    Tmax = np.max(timeEarthquake)
    Tmin = np.min(timeEarthquake)
    timeAni = np.arange(Tmin,Tmax,dtFrames)
    
    Ntime = len(timeAni)
    

    # if ndm == 2:
    #     N_nodes = len(DisplacementData[0, 1::ndm])
    # elif ndm == 3:
    #     N_nodes = len(DisplacementData[0, 1::ndm])
        
    N_nodes = len(DisplacementData[0, 1::ndm])
    
    # Define an array that has the xy or xyz information for animation over all time
    deltaAni = np.zeros([Ntime,N_nodes,ndm])

    
    # Define a counter, this will be used to keep track of how long. It can take
    # a long time for big models!
    counter = 0
    
    # For each coordinat (x,y) or (xyz):
    for ii in range(ndm):
        
        # for each node:
        for jj in range(N_nodes):
            # Keep track of how many iterations have been complete
            if np.floor(10*jj*(ii+1)/(N_nodes*(ndm))) > counter:
                print('The processing is ', (counter + 1)*10, ' percent complete.')
                counter +=1
            
            # Get the node earthquake displacement for the dimension 'n'
            NodeEQDisp = deltaEQ[ii][:,jj]
            
            # Shift the displacement into the animation displacement frame
            deltaAni[:,jj,ii] = D.ShiftDataFrame(timeEarthquake, NodeEQDisp, timeAni)
    
    
    # Save the file information if it is requested.
    if saveFile == True:
        if ndm == 2:
            np.savetxt(outputNameT + '.csv',timeAni, delimiter = ',')
            np.savetxt(outputNameX + '.csv',deltaAni[:,:,0], delimiter = ',')
            np.savetxt(outputNameY + '.csv',deltaAni[:,:,1], delimiter = ',')
        if ndm == 3:
            np.savetxt(outputNameT + '.csv',timeAni, delimiter = ',')
            np.savetxt(outputNameX + '.csv',deltaAni[:,:,0], delimiter = ',')
            np.savetxt(outputNameY + '.csv',deltaAni[:,:,1], delimiter = ',')
            np.savetxt(outputNameZ + '.csv',deltaAni[:,:,2], delimiter = ',')   
    
    
    return timeAni, deltaAni    

def getSubSurface(NodeList,  ax, Style):
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

def getCubeSurf(Nodes, xyz_labels, ax, Style):
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
    
    # 
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
    [tempLines[0], tempSurfaces[0]] = getSubSurface([iNode, jNode, kNode, lNode],  ax, Style)
    [tempLines[1], tempSurfaces[1]] = getSubSurface([iNode, jNode, jjNode, iiNode],  ax, Style)
    [tempLines[2], tempSurfaces[2]] = getSubSurface([iiNode, jjNode, kkNode, llNode],  ax, Style)
    [tempLines[3], tempSurfaces[3]] = getSubSurface([lNode, kNode, kkNode, llNode],  ax, Style)
    [tempLines[4], tempSurfaces[4]] = getSubSurface([jNode, kNode, kkNode, jjNode],  ax, Style)
    [tempLines[5], tempSurfaces[5]] = getSubSurface([iNode, lNode, llNode, iiNode],  ax, Style)

        
    return tempLines, tempSurfaces

# =============================================================================
# Plotting function enablers
# =============================================================================

def update_Plot_Disp(nodes, elements, fig, ax, Style, DisplacementData = np.array([]), 
                     scale = 1):
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
    if DisplacementData.size == 0:    DisplacementData = np.zeros([Nnode, ndm])
    
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
        
        x0 = nodes[:, 1]
        y0 = nodes[:, 2]
        
        dx = DisplacementData[:,0]
        dy = DisplacementData[:,1]
        
        x = x0 + scale*dx
        y = y0 + scale*dy          
        
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
               
        x0 = nodes[:, 1]
        y0 = nodes[:, 2]
        z0 = nodes[:, 3]
        
        dx = DisplacementData[:,0]
        dy = DisplacementData[:,1]
        dz = DisplacementData[:,2]
        
        x = x0 + scale*dx
        y = y0 + scale*dy          
        z = z0 + scale*dz          
        
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
                
                [tempeles, tempVertices] = getCubeSurf(tempNodes, xyz_labels, ax, Style)
                
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

def setStandardViewport(fig, ax, Style, nodeCords, ndm, Disp = [], scale = 1):
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
    
    #TODO
    # Check if delta ani is a different format than the standard displacement
    
    # For the displacement function, the input is a vector
    
    # Adjust plot area, get the bounds on both x and y
    nodeMins = np.min(nodeCords, 0)
    nodeMaxs = np.max(nodeCords, 0)
    
    # Get the maximum displacement in each direction
    if Disp != []:
        
        # if it's the animation
        if len(Disp.shape) == 3:
            dispmax = np.max(np.abs(Disp), (0,1))*scale
        # if it's regular displacement 
        else:
            dispmax = np.max(np.abs(Disp), 0)*scale
        nodeMins = np.min(nodeCords, 0) - dispmax
        nodeMaxs = np.max(nodeCords, 0) + dispmax


    viewCenter = np.average([nodeMins, nodeMaxs], 0)
    viewDelta = 1.1*(nodeMaxs - nodeMins)
    viewRange = max(nodeMaxs - nodeMins)
        
    if ndm == 2:
        ax.set_xlim(viewCenter[0] - viewDelta[0]/2, viewCenter[0] + viewDelta[0]/2)
        ax.set_ylim(viewCenter[1] - viewDelta[1]/2, viewCenter[1] + viewDelta[1]/2)       
        		
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    if ndm == 3:
        ax.set_xlim(viewCenter[0]-(viewRange/4), viewCenter[0]+(viewRange/4))
        ax.set_ylim(viewCenter[1]-(viewRange/4), viewCenter[1]+(viewRange/4))
        ax.set_zlim(viewCenter[2]-(viewRange/3), viewCenter[2]+(viewRange/3))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
	
    if Style.axis_text == False:
        plt.axis('off')
    
    return fig, ax

def initializeFig(nodeCords,Style,ndm):
    
    # set the maximum figure size
    maxFigSize = 8
    
    # Find the node 
    nodeMins = np.min(nodeCords, 0)
    nodeMaxs = np.max(nodeCords, 0)    
    
    # 
    nodeDelta = nodeMaxs - nodeMins
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

def plot_active_model(CustomStyleFunction = None):
    """
    Plots an active model.

    Parameters
    ----------
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
    
    # Get nodes and elements
    nodes, elements = getNodesandElements()
    ndm = len(nodes[0,1:])        

    # Initialize model
    fig, ax = initializeFig(nodes[:,1:], BasicStyle, ndm)
    
    # Plot the first set of data
    update_Plot_Disp(nodes, elements, fig, ax, BasicStyle)
     
    # Adjust viewport size
    setStandardViewport(fig, ax, BasicStyle, nodes[:,1:], ndm)
    
    # Show the plot
    plt.show()
    
    return fig, ax

def plot_model_disp(LoadStep,    scale = 1, dispName = 'All_Disp', 
                    CustomStyleFunction = None, CustomDispStyleFunction = None,
                    nodeName = 'Nodes', eleName = 'Elements', delim = ',', 
                    dtype = 'float32', ftype = '.out'):
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
    
    # Get nodes and elements
    nodes, elements = readNodesandElements(nodeName, eleName, delim, dtype, ftype)
    
    # number of dimensions
    ndm = len(nodes[0,1:])
    Nnodes = len(nodes[:,0])
    
    # Get the displacements if there are any.
    Alldisp = readDisp(dispName)
    disp = np.zeros([Nnodes,ndm])
    
    
    #TODO
    # Consider scaling and resizing the displacement array in the read Disp 
    # Function. That way it's only done once.
    
    for ii in range(ndm):
        disp[:,ii] = Alldisp[LoadStep, (ii+ 1)::ndm]
    
    
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
    fig, ax = initializeFig(nodes[:,1:] + disp*scale, DispStyle, ndm)
    
    # Plot the first set of data
    update_Plot_Disp(nodes, elements, fig, ax, StaticStyle)
    
    # Plot the second set of data
    update_Plot_Disp(nodes, elements, fig, ax, DispStyle,  disp, scale)

	# Adjust plot area.
    setStandardViewport(fig, ax, DispStyle, nodes[:,1:], ndm, disp*scale)
	
    plt.show()
    
    return fig,  ax

def plot_model_eigen(LoadStep,    scale = 1, dispName = 'All_Disp', 
                    CustomStyleFunction = None, CustomDispStyleFunction = None,
                    nodeName = 'Nodes', eleName = 'Elements', delim = ',', 
                    dtype = 'float32', ftype = '.out'):
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
    
    # Get nodes and elements
    nodes, elements = readNodesandElements(nodeName, eleName, delim, dtype, ftype)
    
    # number of dimensions
    ndm = len(nodes[0,1:])
      
    # Get the displacements if there are any.
    Alldisp = readDisp(dispName)
    disp = Alldisp[LoadStep,:]
    
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
    fig, ax = initializeFig(nodes[:,1:], DispStyle, ndm)
    
    # Plot the first set of data
    StaticObjects = update_Plot_Disp(nodes, elements, fig, ax, StaticStyle)
    
    # Plot the second set of data
    DispObjects = update_Plot_Disp(nodes, elements, fig, ax, DispStyle,  disp, scale)

	# Adjust plot area.
    setStandardViewport(fig, ax, DispStyle, nodes[:,1:], ndm)
	
    plt.show()
    
    return fig, ax

def AnimateDisp(dt, deltaAni, nodes, elements, Scale = 1, 
                fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1, 
                CustomStyleFunction = None):
    """
    This function animates displacement in "real time". Model information
    is passed to the function
    
    For big models it's unlikely that the animation will actually run at the 
    desired fps in "real time". Matplotlib just isn't built for high fps 
    animation.
    

    Parameters
    ----------
    dt : array
        The input array of times for animation frames.
    deltaAni : 3D array, [NtimeAni,Nnodes,ndm]
        The input displacement of each node for all time, in every dimension.
    nodes : array
        The input ndoes in standard format:
            [node1, node1x, node1y]
            [....., ......, ......]
            [nodeN, nodeNx, nodeNy]
        The default name is 'Nodes'.
    elements : list
        The list of elments in standard format:
            [ele1Tag, ele1Node1, ..., ele1NodeN]
            [......., ........, ..., ........]
            [eleNTag, eleNNode1, ..., eleNNodeN]
        The default is 'Elements'.
    Scale :  float, optional
        The scale on the xy/xyz displacements. The default is 1.
    fps : TYPE, optional
        The frames per second to be displayed. This changes the number of 
        input data points to the animation.      
        
        It's dubious that the animation will actually display frames at teh
        correct rate, because of performance limitations in matplotlib's
        animation module.
        The default is 24.
    FrameInterval : float, optional
        The time interval between frames to be used. This is used if the user
        wants a certain density of frames, say 24 per input second, but wants 
        to display them at a different intervals than 1/fps. 
        
        The default is 0, which causes 1/fps to be used.
    skipFrame : int, optional
        This allows the user to skip a certain number of input frames
        and start the animation later. The default is 1.
    timeScale : float, optional
        This allows for the animation to be spead up. Note that the speed will
        ultimately be limited by peformance. The default is 1.


    Returns
    -------
    ani : Matplotlib Animation object
        The matplotlib animation object. This must be stored for the animation
        to work.

    """
    
    # Get nodes and elements
    ndm = len(nodes[0,1:])
    
    # Get Style Sheet
    if CustomStyleFunction == None:
        BasicStyle = Style.getStyle(Style.AniStyleSheet)
    else:
        BasicStyle = Style.getStyle(CustomStyleFunction)
    
    # initialize figure
    fig, ax = initializeFig(nodes[:,1:], BasicStyle, ndm)    
    
	# Adjust plot area.   
    setStandardViewport(fig, ax, BasicStyle, nodes[:,1:], ndm, deltaAni)
    
    # Get the animation
    ani = getDispAnimation(dt, deltaAni, nodes, elements, fig, ax, BasicStyle, Scale = Scale, fps = fps,
                           FrameInterval = FrameInterval, skipFrame = skipFrame, timeScale = timeScale)
    
    return ani

def getDispAnimation(dt, deltaAni, nodes, elements, fig, ax, Style, Scale = 1, 
                     fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1):
    """
    This defines the animation of an opensees model, given input data.
    
    For big models it's unlikely that the animation will actually run at the 
    desired fps in "real time". Matplotlib just isn't built for high fps 
    animation.

    Parameters
    ----------
    dt : 1D array
        The input time steps.
    deltaAni : 3D array, [NtimeAni,Nnodes,ndm]
        The input displacement of each node for all time, in every dimension.
    nodes: 
        The node list in standard format
    elements: 1D list
        The elements list in standard format.
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

    Returns
    -------
    TYPE
        Earthquake animation.

    """
    
    
    """
    This function animates an earthquake, given a set of input files.

    """
    #TODO
    # Consider removing dt, it isn't doing much right now.
    

    ndm = len(nodes[0,1:])
    Nnodes = len(nodes[:,0])
    Nele = len(elements)
    
    nodeLabels = nodes[:, 0]
    
    # ========================================================================
    # Initialize Plots
    # ========================================================================
        
    # Add Text
    if ndm == 2:
        time_text = ax.text(0.95, 0.01, '', verticalalignment='bottom', 
                            horizontalalignment = 'right', transform = ax.transAxes, color='grey')
        
    EQObjects = update_Plot_Disp(nodes, elements, fig, ax, Style)
    [EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText] = EQObjects    

    # EqfigNodes
    Nsurf = len(EqfigSurfaces)

    # =============================================================================
    # Animation
    # =============================================================================
   
    # Scale on displacement
    dtInput  = dt[1]
    dtFrames  = 1/fps
    Ntime = len(dt)
    
    deltaAni = deltaAni*Scale
    
    # If the interval is zero
    if FrameInterval == 0:
        FrameInterval = dtFrames*1000/timeScale
    else: 
        pass
    
    # in 3D, we need to use the "set data 3D" method.
    def animate2D(ii):
        # this is the most performance critical area of code
        
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + deltaAni[ii,:,:]
        # Update Plots
        
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
        
        # print('loop start')
        # update element locations
        for jj in range(Nele):
            # Get the node number for the first and second node connected by the element
            TempNodes = elements[jj][1:]
            # This is the xy coordinates of each node in the group
            TempNodeCoords = [xy_labels[node] for node in TempNodes] 
            coords_x = [xy[0] for xy in TempNodeCoords]
            coords_y = [xy[1] for xy in TempNodeCoords]
            
            # Update element lines    
            EqfigElements[jj].set_xdata(coords_x)
            EqfigElements[jj].set_ydata(coords_y)
            # print('loop start')
            # Update the surface if necessary
            if 2 < len(TempNodes):
                tempxy = np.column_stack([coords_x, coords_y])
                EqfigSurfaces[SurfCounter].xy = tempxy
                SurfCounter += 1
       
        # update time Text
        time_text.set_text(round(dtInput*ii,1))
        time_text.set_text(str(round(dtInput*ii,1)) )
        
        
        return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText

    def animate3D(ii):
        # this is the most performance critical area of code
        
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + deltaAni[ii,:,:]
        # Update Plots
        
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
            EqfigElements[jj].set_data_3d(coords_x, coords_y, coords_z)
            
            if len(TempNodes) > 2:
                # Update 3D surfaces
                tempVec = np.zeros([4,4])
                tempVec[0,:] = coords_x
                tempVec[1,:] = coords_y
                tempVec[2,:] = coords_z
                tempVec[3,:] = EqfigSurfaces[SurfCounter]._vec[3,:]
                EqfigSurfaces[SurfCounter]._vec = tempVec
                SurfCounter += 1
        
        
        
        
        return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText

    steps = np.arange(0,Ntime)

    if ndm == 2:
        ani = animation.FuncAnimation(fig, animate2D, steps, interval = FrameInterval)
    elif ndm == 3:
        ani = animation.FuncAnimation(fig, animate3D, steps, interval = FrameInterval)
    
    
    return ani

def AnimateDispSlider(dt, deltaAni, nodes, elements, Scale = 1, 
                fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1, 
                CustomStyleFunction = None):
    """
    This function animates displacement in "real time". Model information
    is passed to the function. A slider is included that can control the animation
    
    For big models it's unlikely that the animation will actually run at the 
    desired fps in "real time". Matplotlib just isn't built for high fps 
    animation.
    

    Parameters
    ----------
    dt : array
        The input array of times for animation frames.
    deltaAni : 3D array, [NtimeAni,Nnodes,ndm]
        The input displacement of each node for all time, in every dimension.
    nodes : array
        The input ndoes in standard format:
            [node1, node1x, node1y]
            [....., ......, ......]
            [nodeN, nodeNx, nodeNy]
        The default name is 'Nodes'.
    elements : list
        The list of elments in standard format:
            [ele1Tag, ele1Node1, ..., ele1NodeN]
            [......., ........, ..., ........]
            [eleNTag, eleNNode1, ..., eleNNodeN]
        The default is 'Elements'.
    Scale :  float, optional
        The scale on the xy/xyz displacements. The default is 1.
    fps : TYPE, optional
        The frames per second to be displayed. This changes the number of 
        input data points to the animation.      
        
        It's dubious that the animation will actually display frames at teh
        correct rate, because of performance limitations in matplotlib's
        animation module.
        The default is 24.
    FrameInterval : float, optional
        The time interval between frames to be used. This is used if the user
        wants a certain density of frames, say 24 per input second, but wants 
        to display them at a different intervals than 1/fps. 
        
        The default is 0, which causes 1/fps to be used.
    skipFrame : int, optional
        This allows the user to skip a certain number of input frames
        and start the animation later. The default is 1.
    timeScale : float, optional
        This allows for the animation to be spead up. Note that the speed will
        ultimately be limited by peformance. The default is 1.


    Returns
    -------
    ani : Matplotlib Animation object
        The matplotlib animation object. This must be stored for the animation
        to work.

    """
    
    # Get nodes and elements
    ndm = len(nodes[0,1:])
    
    # Get Style Sheet
    if CustomStyleFunction == None:
        BasicStyle = Style.getStyle(Style.AniStyleSheet)
    else:
        BasicStyle = Style.getStyle(CustomStyleFunction)
    
    # initialize figure
    fig, ax = initializeFig(nodes[:,1:], BasicStyle, ndm)    
    
	# Adjust plot area.   
    
    setStandardViewport(fig, ax, BasicStyle, nodes[:,1:], ndm, deltaAni)
    
    # Get the animation
    ani = getDispAnimationSlider(dt, deltaAni, nodes, elements, fig, ax, BasicStyle, Scale = Scale, fps = fps,
                                 FrameInterval = FrameInterval, skipFrame = skipFrame, timeScale = timeScale)
    
    return ani

def getDispAnimationSlider(dt, deltaAni, nodes, elements, fig, ax, Style, Scale = 1, 
                           fps = 24, FrameInterval = 0, skipFrame =1, timeScale = 1):
    """
    This defines the animation of an opensees model, given input data.
    
    For big models it's unlikely that the animation will actually run at the 
    desired fps in "real time". Matplotlib just isn't built for high fps 
    animation.

    Parameters
    ----------
    dt : 1D array
        The input time steps.
    deltaAni : 3D array, [NtimeAni,Nnodes,ndm]
        The input displacement of each node for all time, in every dimension.
    nodes: 
        The node list in standard format
    elements: 1D list
        The elements list in standard format.
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

    Returns
    -------
    TYPE
        Earthquake animation.

    """
    

    ndm = len(nodes[0,1:])
    Nnodes = len(nodes[:,0])
    Nele = len(elements)
    
    nodeLabels = nodes[:, 0]
    
    # ========================================================================
    # Initialize Plots
    # ========================================================================
        
    # Add Text
    if ndm == 2:
        time_text = ax.text(0.95, 0.01, '', verticalalignment='bottom', 
                            horizontalalignment='right', transform=ax.transAxes, color='grey')
        
    EQObjects = update_Plot_Disp(nodes, elements, fig, ax, Style)
    [EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText] = EQObjects    

    # EqfigNodes
    Nsurf = len(EqfigSurfaces)

    # ========================================================================
    # Animation
    # ========================================================================
   
    # Scale on displacement
    dtInput  = dt[1]
    dtFrames  = 1/fps
    Ntime = len(dt)
    Frames = np.arange(0,Ntime)
    
    deltaAni = deltaAni*Scale
    
    # If the interval is zero
    if FrameInterval == 0:
        FrameInterval = dtFrames*1000/timeScale
    else: 
        pass    
        
    FrameStart = Frames[0]
    FrameEnd = Frames[-1]
    
    # Slider Location and size relative to plot
    # [x, y, xsize, ysize]
    axSlider = plt.axes([0.25, .03, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Frame', FrameStart, FrameEnd, valinit=FrameStart)
    
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
        
    def animate2D_slider(TimeStep):
        """
        The slider value is liked with the plot - we update the plot by updating
        the slider.
        """
        global is_manual
        is_manual=True
        TimeStep = int(TimeStep)
               
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + deltaAni[TimeStep,:,:]
        # Update Plots
        
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
        
        # print('loop start')
        # update element locations
        for jj in range(Nele):
            # Get the node number for the first and second node connected by the element
            TempNodes = elements[jj][1:]
            # This is the xy coordinates of each node in the group
            TempNodeCoords = [xy_labels[node] for node in TempNodes] 
            coords_x = [xy[0] for xy in TempNodeCoords]
            coords_y = [xy[1] for xy in TempNodeCoords]
            
            # Update element lines    
            EqfigElements[jj].set_xdata(coords_x)
            EqfigElements[jj].set_ydata(coords_y)
            # print('loop start')
            # Update the surface if necessary
            if 2 < len(TempNodes):
                tempxy = np.column_stack([coords_x, coords_y])
                EqfigSurfaces[SurfCounter].xy = tempxy
                SurfCounter += 1
       
        # update time Text
        time_text.set_text(round(TimeStep*dtInput,1))
        time_text.set_text(str(round(TimeStep*dtInput,1)) )        
        
        # redraw canvas while idle
        fig.canvas.draw_idle()    
            
        return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText

    def animate3D_slider(TimeStep):
        
        global is_manual
        is_manual=True
        TimeStep = int(TimeStep)
        
        # this is the most performance critical area of code
        
        # The current node coordinants in (x,y) or (x,y,z)
        CurrentNodeCoords =  nodes[:,1:] + deltaAni[TimeStep,:,:]
        # Update Plots
        
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
            EqfigElements[jj].set_data_3d(coords_x, coords_y, coords_z)
            
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

        return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText

    def update_plot(ii):
    
        # If the control is manual, we don't change the plot    
        global is_manual
        if is_manual:
            return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText
       
        # Find the close timeStep and plot that
        CurrentFrame = int(np.floor(plotSlider.val))
        CurrentFrame += 1
        if CurrentFrame >= FrameEnd:
            CurrentFrame = 0
        
        # Update the slider
        plotSlider.set_val(CurrentFrame)
        is_manual = False # the above line called update_slider, so we need to reset this
        return EqfigNodes, EqfigElements, EqfigSurfaces, EqfigText

    if ndm == 2:
        plotSlider.on_changed(animate2D_slider)
    elif ndm == 3:
        plotSlider.on_changed(animate3D_slider)
    
    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)

    ani = animation.FuncAnimation(fig, update_plot, Frames, interval = FrameInterval)
    return ani




