import openseespytools.response as opr
import numpy as np
import matplotlib.pyplot as plt
import os



def change_folder():
    # This is required to read the file correctly in pytest
    BaseDir = os.getcwd()
    folder = os.path.basename(BaseDir)
    
    if folder != 'fibre_tests':
        os.chdir('fibre_tests')

def fibre_plot():
    
    change_folder()
        
    FiberName = 'Fiber.out'
    LoadStep = 0
    fig, ax = opr.PlotFiberResponse(FiberName, 1400)
    
    check = ax.lines[0].get_xdata()

    return check[45]


def test_fibre_plot(monkeypatch):
    # repress the show plot attribute
    monkeypatch.setattr(plt, 'show', lambda: None)

    check = fibre_plot() 
    solution = 0.5002
    
    assert np.sum(np.abs(check  - solution)) < 10**-8
    
# def fibre_animation():

#     ybound = np.array([-2,.25])*10**7
#     ani = opr.AnimateFibre2DFile(FiberName, Ybound = ybound, rFactor = 2)
  
