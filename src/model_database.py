import os
import openseespy.opensees as op
import numpy as np
from math import asin
import warnings
import openseespytools.model as opm


class OutputDatabase:
    
    def __init__(self, ModelName, LoadCaseName):
        self.ModelName = ModelName
        self.LoadCaseName = LoadCaseName
        
        
        self.ODBdir = ModelName + "_ODB"
        self.LoadCaseDir = os.path.join(self.ODBdir, self.LoadCaseName)
        
        self.delim = ' '
        self.fmt = '%.5e'
        self.ftype = '.out'
        self.dtype = 'float32' 


    def createOutputDatabase(self, Nmodes=0, deltaT=0.0, recorders=[]):
        
        """
        This function creates a directory to save all the output data.
    
        Command: createODB("ModelName",<"LoadCase Name">, <Nmodes=Nmodes(int)>, <recorders=*recorder(list)>)
        
        ModelName    : (string) Name of the model. The main output folder will be named "ModelName_ODB" in the current directory.
        LoadCase Name: (string), Optional. Name of the load case forder to be created inside the ModelName_ODB folder. If not provided,
                        no load case data will be read.
        Nmodes         : (int) Optional key argument to save modeshape data. Default is 0, no modeshape data is saved.
        
        deltaT         : (float) Optional time interval for recording. will record when next step is deltaT greater than last recorder step. 
                        (default: records at every time step)
        
        recorders     : (string) A list of additional quantities a users would like to record in the output database.
                        The arguments for these additional inputs match the standard OpenSees arguments to avoid any confusion.
                        'localForce','basicDeformation', 'plasticDeformation','stresses','strains'
                        The recorders for node displacement and reactions are saved by default to help plot the deformed shape.
        
        Example: createODB(TwoSpanBridge, Pushover, Nmodes=3, recorders=['stresses', 'strains'])
        
        Future: The integrationPoints output works only for nonlinear beam column elements. If a model has a combination 
                of elastic and nonlienar elements, we need to create a method distinguish. 
        
        """
        
        ODBdir = self.ODBdir      # ODB Dir name
        if not os.path.exists(ODBdir):
                os.makedirs(ODBdir)
    
        nodeList = op.getNodeTags()
        eleList = op.getEleTags()
        
        dofList = [int(ii + 1) for ii in range(len(op.nodeCoord(nodeList[0]))) ]
        
        # Save node and element data in the main Output folder
        self.saveNodesandElements()
        
        #########################
        ## Create mode shape dir
        #########################
        if Nmodes > 0:
            ModeShapeDir = os.path.join(ODBdir,"ModeShapes")
            if not os.path.exists(ModeShapeDir):
                os.makedirs(ModeShapeDir)
                
            ## Run eigen analysis internally and get information to print
            Tarray = np.zeros([1,Nmodes])  # To save all the periods of vibration
            op.wipeAnalysis()
            eigenVal = op.eigen(Nmodes+1)
        
            for mm in range(1,Nmodes+1):
                Tarray[0,mm-1]=4*asin(1.0)/(eigenVal[mm-1])**0.5
            
            modeTFile = os.path.join(ModeShapeDir, "ModalPeriods.out")
            np.savetxt(modeTFile, Tarray, delimiter = self.delim, fmt = self.fmt)   
            
            ### Save mode shape data
            for ii in range(1,Nmodes+1):
                self.saveModeShapeData(ii)
            
            op.wipeAnalysis()
                  
        LoadCaseDir = self.LoadCaseDir
    
        if not os.path.exists(LoadCaseDir):
            os.makedirs(LoadCaseDir)
            
        NodeDispFile = os.path.join(LoadCaseDir, "NodeDisp_All.out")
        EleForceFile = os.path.join(LoadCaseDir, "EleForce_All.out")
        ReactionFile = os.path.join(LoadCaseDir, "Reaction_All.out")
        EleStressFile = os.path.join(LoadCaseDir, "EleStress_All.out")
        EleStrainFile = os.path.join(LoadCaseDir, "EleStrain_All.out")
        EleBasicDefFile = os.path.join(LoadCaseDir, "EleBasicDef_All.out")
        ElePlasticDefFile = os.path.join(LoadCaseDir, "ElePlasticDef_All.out")
        # EleIntPointsFile = os.path.join(LoadCaseDir,"EleIntPoints_All.out")
        
        # Save recorders in the ODB folder
        op.recorder('Node', '-file', NodeDispFile,  '-time', '-dT', deltaT, '-node', *nodeList, '-dof',*dofList, 'disp')
        op.recorder('Node', '-file', ReactionFile,  '-time', '-dT', deltaT, '-node', *nodeList, '-dof',*dofList, 'reaction')
        
        if 'localForce' in recorders:
            op.recorder('Element', '-file', EleForceFile,  '-time', '-dT', deltaT, '-ele', *eleList, '-dof',*dofList, 'localForce')   
        
        if 'basicDeformation' in recorders:
            op.recorder('Element', '-file', EleBasicDefFile,  '-time', '-dT', deltaT, '-ele', *eleList, '-dof',*dofList, 'basicDeformation')
    
        if 'plasticDeformation' in recorders:
            op.recorder('Element', '-file', ElePlasticDefFile,  '-time', '-dT', deltaT, '-ele', *eleList, '-dof',*dofList, 'plasticDeformation')  
    
        if 'stresses' in recorders:
            op.recorder('Element','-file', EleStressFile,  '-time', '-dT', deltaT, '-ele', *eleList,'stresses')
        
        if 'strains' in recorders:
            op.recorder('Element','-file', EleStrainFile,  '-time', '-dT', deltaT, '-ele', *eleList,'strains')
        
        # op.recorder('Element', '-file', EleIntPointsFile, '-time', '-dT', deltaT, '-ele', *eleList, 'integrationPoints')           # Records IP locations only in NL elements



    def saveFiberData2D(self, eleNumber, sectionNumber, baseFibreName = "FiberData", deltaT = 0.0):
        """
        Model : string
            The name of the input model database.    
        LoadCase : string
            The name of the input loadcase.    
        element : int
            The input element to be recorded
        section : int
            The section in the input element to be recorded.
        baseFibreName : str, optional
            The base name for the output file.
        deltaT : float, optional
            The time step to be plotted. The program will find the closed time 
            step to the input value. The default is -1.    
        """
               
        ftype = self.ftype
        ODBdir = self.ODBdir
        LoadCaseName = self.LoadCaseName
        
        FibreFileName = baseFibreName  + '_ele_' + str(eleNumber) + '_section_' + str(sectionNumber) + ftype
        FiberDir = os.path.join(ODBdir, LoadCaseName, FibreFileName)
    	
        op.recorder('Element' , '-file', FiberDir, '-time', '-dT', deltaT, '-ele', eleNumber, 'section', str(sectionNumber), 'fiberData')




# =============================================================================
# Nodes and elements
# =============================================================================

    def saveNodesandElements(self, nodeName = 'Nodes', eleName = 'Elements'):
        """   
        This file saves the node and element information for the structure. 
        For each node information is saved in the following format:
            Nodes:    [NodeID, xcord, ycord] or [NodeID, xcord, ycord, zcord]
        
        For elements, the element is saved with the element connectivity. 
        A different file is created for each type of element
        each possible element type.
            Elements: [EleID, eleNode1, eleNode2, ... , eleNodeN]
        Parameters
        ----------
        nodeName : str, optional
            The name of the file to be saved. The default is 'Nodes'.
        eleName : str, optional
            The name of the . The default is 'Elements'.
        delim : str, optional
            The delimeter for the output file. The default is ','.
        fmt : str, optional
            the format of the file to be saved in. The default is '%.5e'.
        """
    
        # Consider making these optional arguements
        delim = self.delim
        fmt = self.fmt
        ftype = self.ftype
        
        ODBdir = self.ODBdir
    
        # Read noades and elements
        nodes, elements = opm.getNodesandElements()
    
        # Sort through the element arrays
        ele2Node = np.array([ele for ele in elements if len(ele) == 3])
        ele3Node = np.array([ele for ele in elements if len(ele) == 4])
        ele4Node = np.array([ele for ele in elements if len(ele) == 5])
        ele8Node = np.array([ele for ele in elements if len(ele) == 9])
    
        nodeFile = os.path.join(ODBdir, nodeName + ftype)
        
        ele2File = os.path.join(ODBdir, eleName + "_2Node" + ftype)
        ele3File = os.path.join(ODBdir, eleName + "_3Node" + ftype)
        ele4File = os.path.join(ODBdir, eleName + "_4Node"  + ftype)
        ele8File = os.path.join(ODBdir, eleName + "_8Node"  + ftype)
    
        # SaveNodes
        np.savetxt(nodeFile, nodes, delimiter = delim, fmt = fmt)
        
        # Save element arrays
        np.savetxt(ele2File, ele2Node, delimiter = delim, fmt = fmt)
        np.savetxt(ele3File, ele3Node, delimiter = delim, fmt = fmt)
        np.savetxt(ele4File, ele4Node, delimiter = delim, fmt = fmt)
        np.savetxt(ele8File, ele8Node, delimiter = delim, fmt = fmt)


    def readNodesandElements(self, nodeName = 'Nodes', eleName = 'Elements'):
        """   
        This function reads input node/element information, assuming it is in the 
        standard format. 
        If outputDir == False, the base directory will be used.    
        
        Parameters
        ----------
        nodeName : str, optional
            The base name for the node file. It will be appended to include
            the file type. The default is 'Nodes.out'.
        eleName : str, optional
            The base nae for the element files. The default is 'Elements.out'.
        delim : str, optional
            The delimiter for files to be read. The default is ','.
        dtype : TYPE, optional
            The data type to read in. The default is 'float32'.
        Returns
        -------
        nodes : Array
            An output vector in standard format
        elements : List
            An output Element vector in standard format.
            elements = [ele1, ele2,..., elen], 
            ele1 = [element, node 1, node 2, ... , node n]
        """
    
        # Consider making these optional arguements
        delim = self.delim
        dtype = self.dtype 
        ftype = self.ftype
            
        ODBdir = self.ODBdir        # ODB Dir name
        
        # Check if output database exists
        if not os.path.exists(ODBdir):
            print('No directory found for nodes and elements')
            
        # Generate the file names
        nodeFile = os.path.join(ODBdir, nodeName + ftype)
        ele2File = os.path.join(ODBdir, eleName + "_2Node" + ftype)
        ele3File = os.path.join(ODBdir, eleName + "_3Node" + ftype)
        ele4File = os.path.join(ODBdir, eleName + "_4Node"  + ftype)
        ele8File = os.path.join(ODBdir, eleName + "_8Node"  + ftype)     
           
        eleFileNames = [ele2File, ele3File, ele4File, ele8File]    
        
        ## Load Node information
        try:
            nodes = np.loadtxt(nodeFile, dtype, delimiter = delim, unpack=False)
        except:
            print("Reading node data from a OpenSees Tcl model")
            nodes = np.transpose(np.loadtxt(nodeFile, dtype=float, delimiter=None, converters=None, unpack=True))
                
        # Populate an array with the input element information
        TempEle = [[]]*4
        
        # Check if the file exists, read it if it does. Ignore warnings if the files are empty
        for ii, FileName in enumerate(eleFileNames):
            if os.path.isfile(FileName):
                
                # Suppress the warning 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        TempEle[ii] = np.loadtxt(FileName, dtype, delimiter = delim, ndmin=2, unpack=False)
                    except:
                        print("Reading element data from a OpenSees Tcl model")
                        TempEle[ii] = np.transpose(np.loadtxt(FileName, dtype=float, delimiter=None,  skiprows=0, ndmin=2,converters=None, unpack=True))
    
        # define the final element array
        elements = [*TempEle[0], *TempEle[1], *TempEle[2], *TempEle[3]]
    
        # Check if any files were read
        if elements is []:
            raise Exception('No files were found!')
    
        return nodes, elements
    
# =============================================================================
# Mode shapes
# =============================================================================

    def saveModeShapeData(self, modeNumber, modeName = "ModeShape"):
        
        nodes_modeshape, Tn = opm.getModeShapeData(modeNumber)
               
        delim = self.delim
        fmt = self.fmt
        ftype = self.ftype
        ODBdir = self.ODBdir
        
        
        ModeShapeDir = os.path.join(ODBdir, "ModeShapes")
        modeFile = os.path.join(ModeShapeDir, modeName + str(modeNumber) + ftype)
        
        ## ModeShapeDir is a default name
        np.savetxt(modeFile, nodes_modeshape, delimiter = delim, fmt = fmt)    
    
    
    def readModeShapeData(self, modeNumber, modeName = "ModeShape"):
    
        # Consider making these optional arguements
        delim = self.delim
        dtype = self.dtype
        ftype = self.ftype
            
        ODBdir = self.ODBdir        # ODB Dir name
        ModeShapeDir = os.path.join(ODBdir, "ModeShapes")
        
        # Check if output database exists
        if not os.path.exists(ModeShapeDir):
            print('Error: No directory found for modeshapes. Use recordODB() command to save modeshapes.')
    
        modeFile = os.path.join(ModeShapeDir, modeName + str(modeNumber) + ftype)
        modeTFile = os.path.join(ModeShapeDir, "ModalPeriods.out")
        
        ## Read modal period data to display
        periods = np.loadtxt(modeTFile, dtype, delimiter = delim, unpack=False)
        period = periods[modeNumber - 1]
        
        ## Load Node information
        try:
            modeshape = np.loadtxt(modeFile, dtype, delimiter = delim, unpack=False)
        except:
            print("Reading modeshape data from a OpenSees Tcl model")
            modeshape = np.transpose(np.loadtxt(modeFile, dtype=float, delimiter=None, converters=None, unpack=True))
    
        return modeshape, period

# =============================================================================
# Displacement
# =============================================================================


    def readNodeDispData(self, DispName = "NodeDisp_All.out"):
        
        ODBdir =  self.ODBdir
        LoadCaseDir = self.LoadCaseName
        
        # Get number of nodes in the model to set a node displacement array
        nodes, elements = self.readNodesandElements()
        Nnodes = len(nodes)
        ndm = len(nodes[0,1:])
               
        NodeDispFile = os.path.join(ODBdir,LoadCaseDir, DispName)
        
        if not os.path.exists(NodeDispFile):
            print('Error: No directory found for Displacement.')        
        
        Disp = np.transpose(np.loadtxt(NodeDispFile, dtype=float, delimiter=None, converters=None, unpack=True))
                
        
        timeSteps = Disp[:,0]
        Ntime = len(Disp[:,0])
    
        tempDisp = np.zeros([Ntime,Nnodes,ndm])
        tempDisp[:,:,0] = Disp[:,1::ndm]
        tempDisp[:,:,1] = Disp[:,2::ndm]
        
        if ndm == 3:
             tempDisp[:,:,2] = Disp[:,3::ndm]
            
        nodes_displacement = tempDisp
        
        return timeSteps, nodes_displacement


# =============================================================================
# Fiber
# =============================================================================

    def saveFiberData2D(self, eleNumber, sectionNumber, deltaT = 0.0, FibreName = "FiberData"):
        """
        Model : string
            The name of the input model database.    
        LoadCase : string
            The name of the input loadcase.    
        element : int
            The input element to be recorded
        section : int
            The section in the input element to be recorded.
        deltaT : float, optional
            The time step to be plotted. The program will find the closed time 
            step to the input value. The default is -1.    
        """
        
        LoadCaseName = self.LoadCaseName
        ftype = self.ftype
        
        ODBdir = self.ODBdir		# ODB Dir name
        FibreFileName = FibreName  + '_ele_' + str(eleNumber) + '_section_' + str(sectionNumber) + ftype
        FiberDir = os.path.join(ODBdir, LoadCaseName, FibreFileName)
    	
        op.recorder('Element' , '-file', FiberDir, '-time', '-dT', deltaT, '-ele', eleNumber, 'section', str(sectionNumber), 'fiberData')


    def readFiberData2D(self, eleNumber, sectionNumber, FibreName = "FiberData"):
        
        # Consider making these optional arguements       
        LoadCaseName = self.LoadCaseName
        
        delim = self.delim
        dtype = self.dtype
        ftype = self.ftype 
        
        ODBdir = self.ODBdir       # ODB Dir name
        
        FibreFileName = FibreName  + '_ele_' + str(eleNumber) + '_section_' + str(sectionNumber) + ftype
        FiberDir = os.path.join(ODBdir, LoadCaseName, FibreFileName)
        
        # Check if output database exists
        if not os.path.exists(FiberDir):
            print('Error: No file for Fiber data. Use saveFiberData2D() to create a recorder.')    
        
        FiberData = np.loadtxt(FiberDir, dtype=dtype, delimiter=delim)
        timeSteps = FiberData[:,0]
        FiberData = FiberData[:,1:]
    
        return timeSteps, FiberData

