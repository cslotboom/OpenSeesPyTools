# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

# =============================================================================
# Imports
# =============================================================================

import openseespytools.model as opm

import numpy as np

# =============================================================================
# Inputs
# =============================================================================

EqFileName = 'GM8ALL_DISP.out'
NodeFileName = 'Nodes.out'
ElementFileName = 'Elements.out'

NodeFileName = 'Nodes'
ElementFileName = 'Elements'

inputDt = 'TimeOut.csv'
dxName = 'DispOutx.csv'
dyName = 'DispOuty.csv'

fps = 24
dtFrames = 1/24

processData = True

# =============================================================================
# Analysis
# =============================================================================

# We process the input data. This can be turned off so we aren't always going 
# through the slow process of changing this data.
if processData == False:
    outputs = opm.getAnimationDisp('GM8ALL_DISP', dtFrames, 2)
    
# Read the processed input Data
# We could just read from the vairbale outputs, however that is generally slow
dt = np.loadtxt(inputDt, dtype ='float32', delimiter=',')
dx = np.loadtxt(dxName, dtype ='float32', delimiter=',')
dy = np.loadtxt(dyName, dtype ='float32', delimiter=',')

# Create the input variable
tempshape = dx.shape
deltaAni = np.zeros([*tempshape,2])
deltaAni[:,:,0] = dx
deltaAni[:,:,1] = dy

# Get nodes and elements
nodes, elements = opm.readNodesandElements()


# # Create the Animation. The animation object must be returned
ani = opm.AnimateDispSlider(dt, deltaAni, nodes, elements, fps=fps, 
                            timeScale = 1,Scale = 1)
