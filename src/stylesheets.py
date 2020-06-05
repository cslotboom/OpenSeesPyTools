"""
Some style sheets are stored here


"""


class StyleSheet:
    """ This is the basic style sheet object. These variales are here purely
    to showcase what standard names can be called. All variables will be
    overwritten
    """
    
    def __init__(self):
        
        # Plot overal style
        self.bg_colour = {}
        self.maxfigDimension = 8
        self.axis_text = True
        
        # Node styles
        self.node = {}
        self.node_tags = True
        self.node_tags_style = {}

        # element styles        
        self.ele = {}
        self.ele_surf = {}
        self.ele_surf_line = {}
        self.ele_solid = {}
        self.ele_solid_line = {}
        
        self.ele_tags = True
        self.ele_tags_style = {}
    
        
    pass

    
def BasicStyleSheet():
    
    style = StyleSheet()
    
    # background colour
    style.bg_colour = 'lightgrey'    
    style.maxfigDimension = 8  
    style.axis_text = False
    
    # style for nodes
    style.node = {'color':'black', 'marker':'o', 'linewidth':0.,'ms':2}     
    style.node_tags = True
    style.node_tags_style = {'color':'green','fontsize':8, 'fontweight':'regular'}     
    
    
    # 1D element style
    style.ele = {'color':'black', 'linewidth':2, 'linestyle':'-'}

    # 1D element style for lines in surfaces
    style.ele_surf = {'color':'blue', 'linewidth':.5, 'linestyle':'--', 
                  'alpha':.4} 

    # 1D element style for lines in surfaces
    style.ele_surf_line = {'color':'black', 'linewidth':1, 'linestyle':'-', 
                       'alpha':.4}

    # 3D element style for lines in 3D elements
    style.ele_solid = {'color':'red', 'linewidth':.5, 'linestyle':'--', 
                   'alpha':.4}     
    style.ele_solid_line = {'color':'black', 'linewidth':1, 'linestyle':'-', 
                        'alpha':.4}  
    
    # style for element text   
    style.ele_tags = True    
    style.ele_tags_style = {'fontsize':8, 'fontweight':'bold', 'color':'darkred'} 


    StyleSheet.ele_2D_style = {'color':'g', 'linewidth':1, 'linestyle':'-', 'alpha':.4}    

    return style


def StaticStyleSheet():
    
    style = StyleSheet()
    
    # background colour
    style.bg_colour = 'lightgrey'    
    style.maxfigDimension = 8  
    style.axis_text = False
    
    # style for nodes
    style.node = {'color':'black', 'marker':'o', 'linewidth':0.,'ms':1}     
    style.node_tags = False
    style.node_tags_style = {'color':'green','fontsize':8, 'fontweight':'regular'}     
    
    
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
    style.ele_tags = False    
    style.ele_tags_style = {'fontsize':8, 'fontweight':'bold', 'color':'darkred'} 


    StyleSheet.ele_2D_style = {'color':'g', 'linewidth':1, 'linestyle':'-', 'alpha':.4}    

    return style

def AniStyleSheet():
    
    style = StyleSheet()
    
    # background colour
    style.bg_colour = 'lightgrey'    
    style.maxfigDimension = 8  
    style.axis_text = True
    
    # style for nodes
    style.node = {'color':'black', 'marker':'o', 'linewidth':0.,'ms':2}     
    style.node_tags = False
    
    
    # 1D element style
    style.ele = {'color':'black', 'linewidth':2, 'linestyle':'-'}

    # 1D element style for lines in surfaces
    style.ele_surf = {'color':'C1', 'linewidth':.5, 'linestyle':'--'} 

    # 1D element style for lines in surfaces
    style.ele_surf_line = {'color':'black', 'linewidth':1, 'linestyle':'-'}

    # 3D element style for lines in 3D elements
    style.ele_solid = {'color':'red', 'linewidth':.5, 'linestyle':'--'}     
    style.ele_solid_line = {'color':'black', 'linewidth':1, 'linestyle':'-'}  
    
    # style for element text   
    style.ele_tags = False    


    StyleSheet.ele_2D_style = {'color':'g', 'linewidth':1, 'linestyle':'-'}  

    return style

def getStyle(StyleFunction):
    
    style = StyleFunction()
    
    return style
    
    
    
    
    
    
def BasicStyle():
    # Set viewport visual style
    bg_colour = 'lightgrey' # background colour
    # pl.rc('font', family='Monospace', size=10) # set font for labels
    node_style = {'color':'black', 'marker':'.', 'markersize':10} # nodes
    ele_style = {'color':'black', 'linewidth':1, 'linestyle':'-'} # elements
    # axis_style = {'color':'grey', 'linewidth':1, 'linestyle':'--'} # x=0, y=0 lines
    offset = 0.05 #offset for text
    
    
    
    # # 2D
    # bc_style = {'color':'black', 'markeredgewidth':1, 'markersize':9,
    #             'fillstyle':'none'} # node translation fixity (boundary conditions)
    # bcrot_style = {'color':'black', 'markeredgewidth':1, 'markersize':10,
    #                'fillstyle':'none'} # node rotation fixity (boundary conditions)
    # # 3D
    # azimuth = -50 #degrees
    # elevation = 20 #degrees
    # bc_style3d = {'length':0.3, 'arrow_length_ratio':0.5, 'colors':'black'}
    # bcrot_style3d = {}
    
    ChartName="Sample Chart"
    
    
    
    
    
    
    