# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:51:12 2020

@author: Christian
"""

import openseespy as op


def defaultAnalysisParameters():
    # The default parameters for a analysis
    pass



def materialPlotter(arglist, LoadProtocol):
    
    # Create uniaxial material
    op.uniaxialMaterial(*arglist)
    
    
    # fix nodes
    
    
    # 
    
    
    
    # Default analysis settings
    defaultAnalysisParameters()
    
    # try to run analysis. If it fails return a message telling the user to 
    # try their own analysis function
    
    # read smallest step size in load protocol try some ratio of that