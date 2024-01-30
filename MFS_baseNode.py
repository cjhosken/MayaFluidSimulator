import maya.cmds as cmds
from maya.api import OpenMaya as om

class MFS_BaseNode(om.MPxNode):
    typeName="MFS_BaseNode"
    typeId=om.MTypeId(0x00000)

    def __init__(self):
        om.MPxNode.__init__(self)

    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def creator(cls):
        return cls()

    def compute(self, plug, dataBlock):
        return om.kUnkownParameter