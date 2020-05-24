# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

# =============================================================================
# Imports
# =============================================================================

import openseesplt.openseesplt as opp
import numpy as np

# =============================================================================
# Inputs
# =============================================================================

EqFileName = 'GM8ALL_DISP.out'
NodeFileName = 'Nodes.out'
ElementFileName = 'Elements.out'

inputDt = 'TimeOut.csv'
dxName = 'DispOutx.csv'
dyName = 'DispOuty.csv'

fps = 24
dtFrames = 1/24

processData = False

# =============================================================================
# Analysis
# =============================================================================

# We process the input data. This can be turned off so we aren't always going 
# through the slow process of changing this data.
if processData == True:
    opp.GetEQDis2D(EqFileName, dtFrames)

# Read the processed input Data
dt = np.loadtxt(inputDt, dtype ='float32', delimiter=',')
dx = np.loadtxt(dxName, dtype ='float32', delimiter=',')
dy = np.loadtxt(dyName, dtype ='float32', delimiter=',')

# Create the Animation. The animation object must be returned
ani = opp.EQAnimation2D(dt, dx, dy, NodeFileName, ElementFileName, fps=fps, 
                      timeScale = 1,Scale = 1)



