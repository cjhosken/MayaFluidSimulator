from maya import cmds
from maya.api import OpenMaya as om
from MFS_menu import MFS_create_menu, MFS_delete_menu
from MFS_solverNode import MFS_SolverNode

maya_useNewAPI = True

nodes = [MFS_SolverNode]

def initializePlugin(plugin):
    mPlugin = om.MFnPlugin(plugin)
    try:
        for node in nodes:
            mPlugin.registerNode(node.kPluginNodeName, node.kPluginNodeId, node.creator, node.initialize, om.MPxNode.kDependNode, node.kPluginNodeClassify)
    except:
        raise RuntimeError("Failed to register {0}".format(node.kPluginNodeName))

    MFS_create_menu()

def uninitializePlugin(plugin):
    MFS_delete_menu()
    
    mPlugin = om.MFnPlugin(plugin)
    try:
        for node in nodes:
            mPlugin.deregisterNode(node.kPluginNodeId)
    except:
        raise RuntimeError("Failed to deregister {0}".format(node.kPluginNodeName))

    