import math
import maya.cmds as cmds
from maya.api import OpenMaya as om
import random

class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.position = [(0, 0, 0)]
        self.velocity = [(0, 0, 0)]
        self.simulating = False

    def speed(self, t):
        return math.sqrt(math.pow(self.velocity[t][0],2) + math.pow(self.velocity[t][1],2) + math.pow(self.velocity[t][2],2))
    
    def speedColor(self, t):
        s = max(3, min(self.speed(t), 10)) * 0.1
        return cmds.colorIndex(1, 210, 1/s, s, hsv=True)

class MFS_Solver():
    points = []
    source_object = None

    def initialize(self):   
        pass

    def update(self):
        t = int(cmds.currentTime(query=True)) - 1

        if (t > 0 and self.simulating):
            for p in self.points:
                p.velocity.append(
                            (
                                p.velocity[t-1][0] + (random.random()-0.5) * 0.2,
                                p.velocity[t-1][1] + (random.random()-0.5) * 0.2,
                                p.velocity[t-1][2] + (random.random()-0.5) * 0.2
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

                cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateX', t=t+1, v=p.position[t][0])
                cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateY', t=t+1, v=p.position[t][1])
                cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateZ', t=t+1, v=p.position[t][2])

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
        
        transform_node = cmds.createNode('transform', name='MFS_particles')

        for p in self.points:
            sphere_name = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=f"MFS_particle_{p.id}")[0]
            #cmds.setAttr(sphere_name + '.overrideEnabled', 1)
            #cmds.setAttr(sphere_name + '.overrideShading', 0)

            cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateX', t=1, v=p.position[0][0])
            cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateY', t=1, v=p.position[0][1])
            cmds.setKeyframe(f"MFS_particle_{p.id}", attribute='translateZ', t=1, v=p.position[0][2])

            cmds.parent(sphere_name, transform_node)

        self.update()