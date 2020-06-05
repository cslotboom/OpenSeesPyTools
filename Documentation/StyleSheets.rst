=========
Style Sheets
=========


Style sheets povide a convenient way to edit the appearance on opensees model plots.
By default, however it's also possible for the user to define a custom style sheet for their model.
The following steps can be used to define the custom style sheet. 
First, a new python file is created. 
This file will be in the plotting directory and contain a function that has the plotting information. 
In the file, import the StyleSheet object, then create a function that assigns the style dictionaries desiredto the style sheet.
Currently only a few keywords for style are supported. The supported keywords are initialized the style base class. 

.. code:: python

	from openseespytools_working.stylesheets import StyleSheet


	def customStyleSheet():
		
		style = StyleSheet()
		
		# background colour
		style.bg_colour = 'white'
		style.maxfigDimension = 8  
		style.axis_text = False
		
		# style for nodes
		style.node = {'color':'black', 'marker':'o', 'linewidth':0.,'ms':2}     
		style.node_tags = True
		style.node_tags_style = {'color':'black','fontsize':8, 
								 'fontweight':'regular'}     
		
		
		# 1D element style
		style.ele = {'color':'grey', 'linewidth':1, 'linestyle':'--'}

		# 1D element style for lines in surfaces
		style.ele_surf = {'color':'grey', 'linewidth':.5, 'linestyle':'--', 
					  'alpha':.2} 

		# 1D element style for lines in surfaces
		style.ele_surf_line = {'color':'grey', 'linewidth':1, 'linestyle':'-', 
						   'alpha':.2}

		# 3D element style for lines in 3D elements
		style.ele_solid = {'color':'grey', 'linewidth':.5, 'linestyle':'--', 
					   'alpha':.2}     
		style.ele_solid_line = {'color':'grey', 'linewidth':1, 'linestyle':'-', 
							'alpha':.2}  
		
		# style for element text   
		style.ele_tags = True    
		style.ele_tags_style = {'color':'black','fontsize':8, 'fontweight':'bold'} 


		StyleSheet.ele_2D_style = {'color':'g', 'linewidth':1, 'linestyle':'-', 
								   'alpha':.4}

		return style
  

From the main function, import the custom style sheet function


.. code:: python

  from CustomStlyeSheet import customStyleSheet

Then in the plot function add the custom style sheet as an arguement.

.. code:: python

  opm.plot_active_model(customStyleSheet)


