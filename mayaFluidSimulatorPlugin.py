from maya import cmds
from maya.api import OpenMaya as om

maya_useNewAPI = True

from MFS_menu import create_custom_menu

create_custom_menu()


from MFS_sourceNode import MFS_SourceNode
from MFS_solverNode import MFS_SolverNode

nodes = [MFS_SourceNode, MFS_SolverNode]

def initializePlugin(plugin):
    pluginFn = om.MFnPlugin(plugin, "Christopher Hosken", "0.0.1")

    for node in nodes:
        try:
            pluginFn.registerNode(node.typeName, node.typeId, node.creator, node.initialize, om.MPxNode.kDependNode)
        except:
            raise RuntimeError("Failed to register node {0}".format(node.typeName))

def uninitializePlugin(plugin):
    pluginFn = om.MFnPlugin(plugin)

    for node in reversed(nodes):
        try:
            pluginFn.unregisterNode(node.typeId)
        except:
            raise RuntimeError("Failed to unregister node {0}".format(node.typeName))