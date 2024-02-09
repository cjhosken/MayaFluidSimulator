import math
import maya.cmds as cmds
from maya.api import OpenMaya as om
import random

class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.position = [(0, 0, 0)]
        self.velocity = [(0, 0, 0)]

    def speed(self):
        t = int(cmds.currentTime(query=True)) - 1
        return math.sqrt(math.pow(self.velocity[t][0],2) + math.pow(self.velocity[t][1],2) + math.pow(self.velocity[t][2],2))
    
    def speedColor(self):
        s = max(3, min(self.speed(), 10)) * 0.1
        return cmds.colorIndex(1, 0.1, 0.2, s)

class MFS_Solver():
    points = []
    source_object = None

    def initialize(self):   
        pass

    def update(self):
        t = int(cmds.currentTime(query=True)) - 1

        for p in self.points:
            if (t > 0):
                p.velocity.append(
                    (
                        p.velocity[t-1][0],
                        p.velocity[t-1][1],
                        p.velocity[t-1][2] 
                    )
                )


                    # USER NAVIER STOKES HERE TO ADJUST THE PARTICLES
                    # SHOULD PROBABLY MAKE A CUSTOM CACHE FILE TYPE FOR POINTS TO READ / WRITE
                    
                p.position.append(
                    (
                        p.position[t-1][0] + p.velocity[t][0], 
                        p.position[t-1][1] + p.velocity[t][1], 
                        p.position[t-1][2] + p.velocity[t][2]
                    )
                )

            cmds.move(p.position[t][0], p.position[t][1], p.position[t][2], f"MFS_particle_{p.id}")
            cmds.setAttr(f"MFS_particle_{p.id}" + '.overrideColor', p.speedColor())

    def reset(self):
        self.clear()
        self.initialize()
        pass

    def clear(self):
        pass

# MAKE A FUNCTION THAT DISTRIBUTES POINTS EVENLY (OR RANDOMLY) INSIDE OF AN OBJECT
# MAY NEED TO SET UP CUDA AND NUMPY FOR THIS

    def point_distribute(self):
        if (self.points is not None):
            for p in self.points:
                cmds.delete(f"MFS_particle_{p.id}")
            self.points = []
        bounding_box = cmds.exactWorldBoundingBox(self.source_object)
        min_point = om.MPoint(bounding_box[0], bounding_box[1], bounding_box[2])
        max_point = om.MPoint(bounding_box[3], bounding_box[4], bounding_box[5])

        pscale = cmds.getAttr(f"{self.source_object}.particleScale")

        i = 0
        ix = min_point[0]
        while ix < max_point[0]:
            iy = min_point[1]
            while iy < max_point[1]:
                iz = min_point[2]
                while iz < max_point[2]:
                    pnt = MFS_Particle()

                    pnt.position[0] = (ix, iy, iz)
                    pnt.velocity[0] = (0, 0, 0)
                    pnt.id = i
                    self.points.append(pnt)

                    i += 1
                    iz += pscale
                iy += pscale 
            ix += pscale 
        
        for p in self.points:
            sphere_name = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=f"MFS_particle_{p.id}")[0]
            cmds.setAttr(sphere_name + '.overrideEnabled', 1)
            cmds.setAttr(sphere_name + '.overrideShading', 0)

            cmds.setAttr(sphere_name + '.overrideColor', p.speedColor())

            cmds.move(p.position[0][0], p.position[0][1], p.position[0][2], sphere_name)

            cmds.parent(sphere_name, "MFS_Domain")

        self.update()