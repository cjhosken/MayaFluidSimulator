import os
import maya.cmds as cmds
from maya.api import OpenMaya as om

attributes = [
    ("domainSize", "float3"),
    ("gravity", "float"), 
    ("viscocity", "float"), 
    ("particleScale", "float"), 
    ("timeStep", "float"), 
    ("isInflow", "bool"), 
    ("velocity", "float3")
]

class MFS_SolverNode(om.MPxNode):
    kPluginNodeName="MFS_SolverNode"
    kPluginNodeId=om.MTypeId(0x00001)
    kPluginNodeClassify = 'utility/particle'

    domainSize = om.MObject()
    gravity = om.MObject()
    viscocity = om.MObject()
    particleScale = om.MObject()
    timeStep = om.MObject()

    isInflow = om.MObject()
    velocity = om.MObject()

    def __init__(self):
        om.MPxNode.__init__(self)

    @classmethod
    def creator(cls):
        return cls()

    @classmethod
    def initialize(cls):
        numericAttributeFn = om.MFnNumericAttribute()

        cls.domainSize = numericAttributeFn.create("domainSize", "dom", om.MFnNumericData.k3Float, 5.0)
        cls.addAttribute(cls.domainSize)

        cls.gravity = numericAttributeFn.create("gravity", "g", om.MFnNumericData.kFloat, 9.8)
        cls.addAttribute(cls.gravity)

        cls.viscocity = numericAttributeFn.create("viscocity", "r", om.MFnNumericData.kFloat, 0.1)
        cls.addAttribute(cls.viscocity)
        
        cls.particleScale = numericAttributeFn.create("particleScale", "ps", om.MFnNumericData.kFloat, 0.25)
        cls.addAttribute(cls.particleScale)

        cls.timeStep = numericAttributeFn.create("timeStep", "ts", om.MFnNumericData.kFloat, 0.1)
        cls.addAttribute(cls.timeStep)

        cls.isInflow = numericAttributeFn.create("isInflow", "if", om.MFnNumericData.kBoolean, False)
        cls.addAttribute(cls.isInflow)

        cls.velocity = numericAttributeFn.create("velocity", "v", om.MFnNumericData.k3Float, 0.0)
        cls.addAttribute(cls.velocity)

    def compute(self, pPlug, pData):
        return om.kUnknownParameter

def MFS_assign_SolverNode(*args):
    selected_objects = cmds.ls(selection=True)
    node_name = "MFS_SolverNode"

    if not selected_objects:
        cmds.confirmDialog(title="Solver Error!", 
            message="You need to use the solver on an object!",
            button="Sorry"
        )
        return

    active_object = selected_objects[0]

    if cmds.objectType(active_object) != "transform":
        cmds.confirmDialog(title="Solver Error!", 
            message="You need to use the solver on an object!",
            button="Sorry"
        )
        return

    node = cmds.createNode(node_name, name=active_object + "_" + node_name)

    for attr in attributes:
        attr_path = "{}.{}".format(node, attr[0])
        target_path = "{}.{}".format(active_object, attr[0])

        if attr[1] == "float3":
            cmds.addAttr(active_object, longName=attr[0], attributeType='compound', numberOfChildren=3)
            cmds.addAttr(active_object, longName='{}_X'.format(attr[0]), attributeType='float', parent=attr[0])
            cmds.addAttr(active_object, longName='{}_Y'.format(attr[0]), attributeType='float', parent=attr[0])
            cmds.addAttr(active_object, longName='{}_Z'.format(attr[0]), attributeType='float', parent=attr[0])
            
            cmds.setAttr('{}.{}_X'.format(active_object, attr[0]), cmds.getAttr(attr_path)[0][0])
            cmds.setAttr('{}.{}_Y'.format(active_object, attr[0]), cmds.getAttr(attr_path)[0][1])
            cmds.setAttr('{}.{}_Z'.format(active_object, attr[0]), cmds.getAttr(attr_path)[0][2])
        else:
            cmds.addAttr(active_object, longName=attr[0], attributeType=attr[1])
            cmds.setAttr(target_path, cmds.getAttr(attr_path))

        cmds.connectAttr(attr_path, target_path, force=True)

    wireframe = cmds.polyCube(width=1, height=1, depth=1, name="MFS_Domain")[0]

    cmds.setAttr(wireframe + '.overrideEnabled', 1)
    cmds.setAttr(wireframe + '.overrideShading', 0)

    cmds.setAttr(active_object + '.overrideEnabled', 1)
    cmds.setAttr(active_object + '.overrideShading', 0)

    cmds.connectAttr("{}.domainSize0".format(node), "{}.scaleX".format(wireframe))
    cmds.connectAttr("{}.domainSize1".format(node), "{}.scaleY".format(wireframe))
    cmds.connectAttr("{}.domainSize2".format(node), "{}.scaleZ".format(wireframe))