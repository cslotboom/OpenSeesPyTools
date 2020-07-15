# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:48:25 2019
@author: Christian
"""

# =============================================================================
# Imports
# =============================================================================

from openseespytools import model
from openseespytools.model_database import OutputDatabase

import numpy as np

# =============================================================================
# Inputs
# =============================================================================

NodeFileName = 'Nodes.out'
ElementFileName = 'Elements.out'

modelName = 'Test'
LoadCase = 'LoadCase'

fps = 24
dtFrames = 1/24

processData = True

# =============================================================================
# Analysis
# =============================================================================

# Initialize Database Object
testDatabase = OutputDatabase(modelName, LoadCase)

# Process Displacement
model.getAnimationDisp(testDatabase, dtFrames, 2)

# # Create the Animation. The animation object must be returned
ani = model.animate_deformedshape(testDatabase, dtFrames, DispName = 'nodeDisp_Ani.out')