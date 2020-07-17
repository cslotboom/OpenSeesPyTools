import openseespy.opensees as op
import openseespy.postprocessing.Get_Rendering as opp
from openseespytools import model
from openseespytools.model_database import OutputDatabase
import numpy as np


import ModelFunctions as mf
import AnalysisFunctions as af

# =============================================================================
# Load control with Disp
# =============================================================================

op.wipe()

# Build Model
mf.getSections()
mf.buildModel()

# Prepare Outputs
Model = 'Cantilever'
LoadCase = 'Pushover'
element1 = 1
section1 = 1

# Creat Output Database
OutputDB = OutputDatabase(Model, LoadCase)
OutputDB.createOutputDatabase()
OutputDB.saveFiberData2D(element1, section1)

# Run Analysis
af.PushoverLcD(0.01)

op.wipe()

# =============================================================================
# Animation outputs
# =============================================================================

model.plot_fiberResponse2D(OutputDB, element1, section1, InputType = 'stress', tstep=1)
ani1 = model.animate_fiberResponse2D(OutputDB, element1, section1, InputType = 'stress', rFactor = 4)
