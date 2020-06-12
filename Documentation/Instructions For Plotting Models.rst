Procedure for Plotting Displacements
====================================

To plot displacements, several functions need to be called while the opensees model is active.
The functions will work "as is", and where inputs and outputs will be saved in a standardized format.
The user can also specify different file names and types if they for some reason want to do this.

Fist the model will be made, then the functions **saveNodesandElements** are called. 
These save information about the node coordinates and element connectivity to files in a standardized format.
A **recoder** for all of the displacements will also need to be called. The final outputs will

.. code:: python

  import opensees.openseespy as op
  import openseestools.model as opm
    
  *** Code that builds model goes here ***
    
  op.recorder( 'Node' , '-file' , dispFileName , '-time' ,'-dof', 1,  2, 'disp')
  opm.saveNodesandElements()

  *** Code that analyzed model goes here ***
  
  opm.plot_model_disp(Loadstep)

There are many optional arguement that give the user control over the modle functionality.


Procedure for Animating Displacements
====================================

The procedure for plotting animations is similar to displacements. The nodes and elements must be saved then loaded (or just kept as variables).

.. code:: python

  import opensees.openseespy as op
  import openseestools.model as opm
    
  *** Code that builds model goes here ***
    
  op.recorder( 'Node' , '-file' , dispFileName , '-time' ,'-dof', 1,  2, 'disp')
  opm.saveNodesandElements()


Data then needs to be prepared to pass to the animation function.
When preparing a function for animation, it's recommended that the *getAnimationDisp* function is used to preprocess data.
This function shifts the input data into the animations time domain.
Output files will be created that represents the node displacements over all time


.. code:: python

  
  opm.getAnimationDisp(InputFileName, dtFrames)
    
  # Read the processed input Data
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
  
  # pass it to the animation file

  # ani = opm.AnimateDisp(dt, deltaAni, nodes, elements, fps=fps, 
                      timeScale = 1,Scale = 1)


Note that the output from *getAnimationDisp* could be used without reading in the data again.
However, it's generally much faster to read from the files saved by *getAnimationDisp* because they are often way smaller
