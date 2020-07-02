import openseespytools.response as opr
import numpy as np


FiberName = 'Fiber.out'
LoadStep = 0
opr.PlotFiberResponse(FiberName, 1400)

ybound = np.array([-2,.25])*10**7
ani = opr.AnimateFiber2DFile(FiberName, Ybound = ybound, rFactor = 2)