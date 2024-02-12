from maya import cmds
from maya.api import OpenMaya as om
import os
import math
import random

maya_useNewAPI = True

class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.position = [(0, 0, 0)]
        self.velocity = [(0, 0, 0)]

    def speed(self, t):
        return math.sqrt(math.pow(self.velocity[t][0],2) + math.pow(self.velocity[t][1],2) + math.pow(self.velocity[t][2],2))
    
    def speedColor(self, t):
        s = max(3, min(self.speed(t), 10)) * 0.1
        return cmds.colorIndex(1, 210, 1/s, s, hsv=True)

class MFS_Solver():
    points = []
    source_object = None
    domain_object = None
    solved = False
    initialized = False

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
                            (p.position[t-1][1] + p.velocity[t][1], 
                                p.position[t-1][2] + p.velocity[t][2]
                            )
                        )

                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=t+1, v=p.position[t][0])
                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=t+1, v=p.position[t][1])
                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=t+1, v=p.position[t][2])


    def clear(self):
        if (self.initialized):
            if (cmds.objExists(self.domain_object)):
                cmds.delete(self.domain_object)

        cmds.setAttr(self.source_object + '.overrideShading', 1)
        cmds.setAttr(self.source_object + '.overrideEnabled', 0)
        
        self.points = []
        self.source_object = None
        self.domain_object = None
        self.solved = False
        self.initialized = False

# MAKE A FUNCTION THAT DISTRIBUTES POINTS EVENLY (OR RANDOMLY) INSIDE OF AN OBJECT
# MAY NEED TO SET UP CUDA AND NUMPY FOR THIS

    def point_distribute(self, pscale):
        bounding_box = cmds.exactWorldBoundingBox(self.source_object)
        min_point = om.MPoint(bounding_box[0], bounding_box[1], bounding_box[2])
        max_point = om.MPoint(bounding_box[3], bounding_box[4], bounding_box[5])

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
            
        if (cmds.objExists(f"MFS_PARTICLES_{self.source_object}")):
            cmds.delete(f"MFS_PARTICLES_{self.source_object}")
        transform_node = cmds.createNode('transform', name=f'MFS_PARTICLES_{self.source_object}')

        for p in self.points:
            if (cmds.objExists(f"MFS_PARTICLE_{self.source_object}_{p.id:05}")):
                cmds.delete(f"MFS_PARTICLE_{self.source_object}_{p.id:05}")
            sphere_name = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=f"MFS_PARTICLE_{self.source_object}_{p.id:05}")[0]

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=1, v=p.position[0][0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=1, v=p.position[0][1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=1, v=p.position[0][2])

            cmds.parent(sphere_name, transform_node)

        #self.update()


solvers = []

def MFS_create_menu():

    MFS_delete_menu()

    MFS_menu = cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)

    cmds.menuItem(label="Open Maya Fluid Simulator", command=MFS_popup)


def MFS_popup(*args):

    cmds.window(title="Maya Fluid Simulator", widthHeight=(500, 500))
    cmds.columnLayout(adjustableColumn=True)
    
    pscaleCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.1, field=True, label="Particle Scale")
    cmds.button(label="Initialize!", command=lambda *args:MFS_initializeSolver(pscaleCtrl))

    # gravity
    cmds.floatFieldGrp( numberOfFields=3, label='Gravity', extraLabel='cm', value1=0, value2=-9.8, value3=0 )

    # vortcity
    cmds.floatSliderGrp(minValue=0, step=0.1, value=0.1, field=True, label="Vorticity")

    # velocity
    cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )

    # is inflow
    cmds.checkBox(label="Inflow", value=False)

    if (True):
        cmds.button(label="Solve!", command=MFS_runSolver)
    else:
        cmds.button(label="Reset")
        
    cmds.showWindow()

def MFS_delete_menu():
    if cmds.menu("MFS_menu", exists=True):
        cmds.deleteUI("MFS_menu", menu=True)


def MFS_initializeSolver(pscaleCtrl, *args):
    selected_objects = cmds.ls(selection=True)

    if not selected_objects:
        cmds.confirmDialog(title="Solver Error!", 
            message="You need to use the solver on an object!",
            button="Sorry"
        )
        return
    
    active_object = selected_objects[0]

    if cmds.objectType(active_object) != "transform" or "MFS_PARTICLE" in active_object or "MFS_DOMAIN" in active_object:
        cmds.confirmDialog(title="Solver Error!", 
            message="You need to use the solver on an object!",
            button="Sorry"
        )
        return
    
    for solver in solvers:
        if (solver.source_object == active_object):
            solver.clear()
            solvers.remove(solver)

    solver = MFS_Solver()
    solver.source_object = active_object
    solvers.append(solver)

    domain = cmds.polyCube(width=1, height=1, depth=1, name=f"MFS_DOMAIN_{active_object}")[0]
    solver.domain_object = domain


    cmds.setAttr(solver.domain_object + '.overrideEnabled', 1)
    cmds.setAttr(solver.domain_object + '.overrideShading', 0)

    cmds.setAttr(solver.source_object + '.overrideEnabled', 1)
    cmds.setAttr(solver.source_object + '.overrideShading', 0)
    
    pscale = cmds.floatSliderGrp(pscaleCtrl, query=True, value=True)

    solver.point_distribute(pscale)
    solver.initialized = True


def MFS_runSolver(*args):
    cmds.currentTime(1, edit=True)
    MFS_solve()

def MFS_solve():
    solver = solvers[0]

    t = int(cmds.currentTime(query=True))
    
    if (t < 1 or t > 30): 
        solver.solved = True
        return
    
    solver.update()
    cmds.currentTime(t + 1, edit=True)
    MFS_solve()

if __name__ == "__main__":
    MFS_create_menu()