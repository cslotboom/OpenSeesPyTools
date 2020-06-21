from openseespytools_working.stylesheets import StyleSheet


def customStyleSheet():
    
    style = StyleSheet()
    
    # background colour
    style.bg_colour = 'lightgrey'
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