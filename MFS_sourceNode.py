import maya.cmds as cmds
from maya.api import OpenMaya as om

from MFS_baseNode import MFS_BaseNode

class MFS_SourceNode(MFS_BaseNode):
    typeName="MFS_SourceNode"
    typeId=om.MTypeId(0x00001)