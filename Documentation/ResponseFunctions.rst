=========
Response functions
=========


These functions allow you to see how your data is changing. 
This can be usful in understanding hystersis data, or seeing how a quantity changes over time.


AnimateXY
=========

The following code showcases the animation of two curves:

.. code:: python

  import numpy as np
  import openseespytools.response as rs
  npoints = 1000
  
  x = np.linspace(0,2*np.pi,npoints)
  y = np.sin(x)
  
  rs.AnimateXY(x,y, fps = 24, rFactor = 2)
  
We can also skip some start or end frames, as well as reduce the data, and only output a certain amount of frames:

.. code:: python

  # animiate a curve, but skipping some start and end frames
  rs.AnimateXY(x,y, fps = 24, skipStart = 300,  skipEnd = 500)

  # animate only the  a curve, but skipping the start frames
  rs.AnimateXY(x,y, fps = 24, rFactor = 2, outputFrames = 48)

AnimateCyclicXYCurves
=========

The 'AnimateCyclicXYCurves' function allows us to directly compare two data sets as they go through a cycle. 
This is useful for comparing experimental and analyitical data.:

.. code:: python

  import numpy as np
  import openseesplt.openseesplt as rs
  import scipy
  
  # Create a vector with some noise.
  permutate = np.random.normal(0,1,npoints)/4
  Ynoise = y + permutate
  Ynoise = scipy.signal.savgol_filter(Ynoise,53,2)

  # Define the first and second curve
  Curve1 = np.zeros([3*npoints,2])
  Curve2 = np.zeros([3*npoints,2])

  # Define the first and second curve
  Curve1[:,0] = np.concatenate([x,x[::-1],x])
  Curve1[:,1] = np.concatenate([y,-y[::-1],y])
  Curve2[:,0] = np.concatenate([x,x[::-1],x])
  Curve2[:,1] = np.concatenate([Ynoise,-Ynoise[::-1],Ynoise])


  rs.AnimateCyclicXYCurves(Curve1, Curve2,NFrames=300,fps = 60)
