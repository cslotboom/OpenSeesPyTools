========================
Model Functions
========================

These functions can be used to get useful information from the opensees model.
This includes plots of the model as it displaces over time, such as in a pushover or earthquake.
StyleSheets then used to modify the appearance of the plot.

The functions will work "as is", and where inputs and outputs will be saved in a standardized format.
The user can also specify different file names and types if they for some reason want to do this.





======================
Enabling Functions Displacements
======================

There are several functions that can be used by to prepare the model for plotting.

getNodesandElements
===================

This function returns the nodes and elments for an active model, in a standardized format. 
The OpenSees model must be active in order for the function to work.
The standard format for nodes is *[[Nodetag1, xcoord, ycoord],[Nodetag1, xcoord, ycoord], ... , [NodetagN, xcoord, ycoord]]*
The standard format for elements is *[ele1, ele2, ... , eleN]*
where each element is  *[eletag1, elenode1, .... , elenodeN]*.
It's also possible to use different output file names, however this is not recommended.


saveNodesandElements
====================

This function saves the nodes and elements for an active model, in a standardized format. 
The OpenSeesPy model must be active in order for the function to work.
For elements, four total files are generated, one for each type of element (2Node, 3Node, 4Node, etc.).


readDisp
========

This functions reads an input displaement file. Including it is probably overkill.

readNodesandElements
====================

This functions reads files containing the node and element connectivity information.
It's assumed that the files are saved in the format output by *saveNodesandElements*


getAnimationDisp
================

This function prepars an input displacement  for animation.
Often the input file for displacement is very large, and takes a long time to read. 
Using this fuction will reduce the input animation time, and allows for data to be accessed more quickly.

It's assumed that the input file records the x,y (or x,y,z) of all nodes.
The time in the dispalcement file is shifted into the domain 
of the animiation.


getSubSurface
================

Not used by users. Creates and returns the sub-surface objects for a node list.
The input list of nodes corespond the the vertices of a quadralateral 

getCubeSurf
================

This functions plots the nodes and surfaces for a 8 node element, and returns the objects



======================
Plotting functions
======================

These files can be used to visualize the response of OpenSeesPy models


plot_active_model
======================

plots an active OpenSeesPy model. No files are required to be saved or read.


plot_model_disp
======================
This function plots a the displacement of a model. It's assumed that node
and element files are saved in the standard format.
It's also assumed that a displlacement recorder file is in the directory
and contains data in the standard format.


AnimateDisp
===========

This function animates displacement in "real time". Model information
is passed to the function from node, element, and displacment files

For big models it's unlikely that the animation will actually run at the 
desired fps in "real time". Matplotlib just isn't built for high fps 
animation.

getDispAnimation
================
Not used by users. This defines the animation of an opensees model, given input data.




======================
Plotting function enablers
======================

These enable the animation files, and aren't typically used by users.


setStandardViewport
===================
Not used by users. This function sets the standard viewport size of a function, using the
nodes as an input.


initializeFig
=============
Initializes the figure, this later will be updated.

update_Plot_Disp
================

Not used by users.
This functions plots an image of the model in it's current diplacement
state. If no displacement data is passed to the funtion, it plots the base
model.
It returns the plotted matplotlib objects, the object types will depend on
if the domain is in 2D or 3D.
