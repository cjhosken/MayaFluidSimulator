from maya import cmds
from maya.api import OpenMaya as om
import os
import math
import random
import numpy as np

# ------ CLASSES ------


class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.position = [[0, 0, 0]]
        self.velocity = [[0, 0, 0]]

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
    pscale = 0

    def update(self, start, end, other_force, init_vel, viscosity, scale):
        t = (int(cmds.currentTime(query=True)) - start)

        mass = 0.1

        if (not self.solved):
            for p in self.points:        
                if (t==0):
                    p.velocity[0] = [
                        init_vel[0] * scale,
                        init_vel[1] * scale,
                        init_vel[2] * scale
                    ]

                    cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateX', option="keys" )
                    cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateY', option="keys" )
                    cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateZ', option="keys" )
                else:
                    bounding_box = cmds.exactWorldBoundingBox(self.domain_object)
                    min_point = om.MPoint(bounding_box[0], bounding_box[1], bounding_box[2])
                    max_point = om.MPoint(bounding_box[3], bounding_box[4], bounding_box[5])


                    # FOR EACH PARTICLE i
                    # find neighbors j

                    # pi = summationj of mj Wij
                    # compute pi using rho i

                    # Need to find mass/pressure * divergence(density)

                    density = [0, 0, 0]
                    velocity = [0, 0, 0]

                    pressure = 0.1

                    pressure_force = [
                        mass / pressure * divergence(density[0]), 
                        mass / pressure * divergence(density[1]), 
                        mass / pressure * divergence(density[2])
                    ]

                    pressure_force = [0, 0, 0]


                    # mass * viscosity * divergence(velocity)
                    viscosity_force = [
                        mass * viscosity * divergence(velocity[0]), 
                        mass * viscosity * divergence(velocity[1]), 
                        mass * viscosity * divergence(velocity[2]) 
                    ]

                    viscosity_force = [0, 0, 0]
                    

                    # CREATE A FORCE FACTOR
                    force = [
                        pressure_force[0] + viscosity_force[0] + (mass * other_force[0]),
                        pressure_force[1] + viscosity_force[1] + (mass * other_force[1]),
                        pressure_force[2] + viscosity_force[2] + (mass * other_force[2])
                    ]

                    # AFFECT THE VELOCITY WITH FORCE
                    p.velocity.append(
                                        [
                                            p.velocity[t-1][0] + (force[0] * scale)/mass,
                                            p.velocity[t-1][1] + (force[1] * scale)/mass,
                                            p.velocity[t-1][2] + (force[2] * scale)/mass
                                        ]
                                    )
                    




                    # CALCULATE AN "ADVECTED" POSITION (where the particle will move to)
                    advected = [
                        p.position[t-1][0] + (p.velocity[t][0]) * scale,
                        p.position[t-1][1] + (p.velocity[t][1]) * scale,
                        p.position[t-1][2] + (p.velocity[t][2]) * scale
                    ]
                
                    # CHECK FOR PARTICLE COLLISIONS
                    #for s in self.points:
                    #    if (s.id != p.id):
                    #        if (math.sqrt(math.pow(s.position[t-1][0] - p.position[t-1][0] + p.velocity[t][0], 2) + math.pow(s.position[t-1][1] - p.position[t-1][1] + p.velocity[t][1], 2) + math.pow(s.position[t-1][2] - p.position[t-1][2] + p.velocity[t][2], 2)) < self.pscale/2):
                    #            p.velocity[t][0] = -p.velocity[t][0] * viscosity
                    #            p.velocity[t][1] = -p.velocity[t][1] * viscosity
                    #            p.velocity[t][2] = -p.velocity[t][2] * viscosity

                    # CHECK BOUNDARIES
                    if (advected[0] < min_point[0] or advected[0] > max_point[0]):
                        p.velocity[t][0] = -p.velocity[t][0] * viscosity

                    if (advected[1] < min_point[1] or advected[1] > max_point[1]):
                        p.velocity[t][1] = -p.velocity[t][1] * viscosity
                        
                    if (advected[2] < min_point[2] or advected[2] > max_point[2]):
                        p.velocity[t][2] = -p.velocity[t][2] * viscosity


                    # UPDATE POSITION
                    p.position.append(
                                        [
                                            p.position[t-1][0] + (p.velocity[t][0]) * scale, 
                                            p.position[t-1][1] + (p.velocity[t][1]) * scale, 
                                            p.position[t-1][2] + (p.velocity[t][2]) * scale
                                        ]
                                    )

                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=t+start, v=p.position[t][0])
                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=t+start, v=p.position[t][1])
                cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=t+start, v=p.position[t][2])


    def clear(self, keepDomain):
        if (self.initialized):
            if (self.domain_object is not None and cmds.objExists(self.domain_object) and not keepDomain):
                cmds.delete(self.domain_object)

        cmds.setAttr(self.source_object + '.overrideShading', 1)
        cmds.setAttr(self.source_object + '.overrideEnabled', 0)
        
        self.points = []
        self.source_object = None
        self.domain_object = None
        self.solved = False
        self.initialized = False

    def clearSim(self, start, end):
        for p in self.points:
            p.velocity = [
                        p.velocity[0]
                    ]

            p.position = [
                p.position[0]
            ]

            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateX', option="keys" )
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateY', option="keys" )
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", time=(0,end), attribute='translateZ', option="keys" )

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=start, v=p.position[0][0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=start, v=p.position[0][1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=start, v=p.position[0][2])


# MAKE A FUNCTION THAT DISTRIBUTES POINTS EVENLY (OR RANDOMLY) INSIDE OF AN OBJECT
# MAY NEED TO SET UP CUDA AND NUMPY FOR THIS

    def point_distribute(self, pscale):
        self.pscale = pscale


        bounding_box = cmds.exactWorldBoundingBox(self.source_object)
        min_point = om.MPoint(bounding_box[0], bounding_box[1], bounding_box[2])
        max_point = om.MPoint(bounding_box[3], bounding_box[4], bounding_box[5])

        cmds.progressWindow(title='Distributing Points', progress=0, status='Progress: 0%', isInterruptable=False, maxValue=((max_point[0] - min_point[0]) * (max_point[1] - min_point[1]) * (max_point[2] - min_point[2]))/pscale)

        i = 1
        ix = min_point[0]
        while ix <= max_point[0]:
            iy = min_point[1]
            while iy <= max_point[1]:
                iz = min_point[2]
                while iz <= max_point[2]:
                    pnt = MFS_Particle()

                    pnt.position[0] = [ix, iy, iz]
                    pnt.velocity[0] = [0, 0, 0]
                    pnt.id = i

                    self.points.append(pnt)

                    i += 1
                    iz += pscale
                iy += pscale 
            ix += pscale 



        if (cmds.objExists(f"MFS_PARTICLES_{self.source_object}")):
            cmds.delete(f"MFS_PARTICLES_{self.source_object}")
        transform_node = cmds.createNode('transform', name=f'MFS_PARTICLES_{self.source_object}')

        progress = 0

        for p in self.points:
            if (cmds.objExists(f"MFS_PARTICLE_{self.source_object}_{p.id:05}")):
                cmds.delete(f"MFS_PARTICLE_{self.source_object}_{p.id:05}")
            
            sphere_name = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=f"MFS_PARTICLE_{self.source_object}_{p.id:05}")[0]
            
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=0, v=p.position[0][0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=0, v=p.position[0][1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=0, v=p.position[0][2])
            
            progress += 1

            cmds.parent(sphere_name, transform_node)
            cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')

        #self.update()
        cmds.progressWindow(endProgress=1)

solvers = []

# ------ MENUS & BUTTON FUNCTIONS------

def MFS_create_menu():

    MFS_delete_menu()

    cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)

    cmds.menuItem(label="Open Maya Fluid Simulator", command=MFS_popup, image="D:/MFS_icon_solver_512.png")

def MFS_popup(*args):

    cmds.window(title="Maya Fluid Simulator", widthHeight=(500, 500))
    col = cmds.columnLayout(adjustableColumn=True)

    image_path = "D:/MFS_banner.png"
    image = cmds.image(width=300, height=150, image=image_path)
    
    pscaleCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.25, field=True, label="Particle Scale")
    domainCtrl = cmds.checkBox(label="Keep Domain", value=True)
    
    cmds.rowLayout(numberOfColumns=2)

    cmds.button(label="Initialize", command=lambda *args:MFS_initializeSolver(pscaleCtrl, domainCtrl))
    cmds.button(label="X", command=lambda *args:MFS_deleteSolver())

    cmds.columnLayout(adjustableColumn=True, parent=col)

    # gravity
    forceCtrl = cmds.floatFieldGrp( numberOfFields=3, label='Force', extraLabel='cm', value1=0, value2=-9.8, value3=0 )

    # viscosity
    viscCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.1, field=True, label="Viscosity")

    # velocity
    velCtrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )
    
    cmds.rowLayout(numberOfColumns=3)
    timeCtrl = cmds.intFieldGrp(numberOfFields=2, value1=1, value2=120, label="Frame Range")
    tsCtrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.05, field=True, label="Time Scale")

    cmds.columnLayout(adjustableColumn=True, parent=col)

    cmds.rowLayout(numberOfColumns=2)
    cmds.button(label="Solve", command=lambda *args:MFS_runSolver(timeCtrl, forceCtrl, viscCtrl, velCtrl, tsCtrl))
    cmds.button(label="X", command=lambda *args:MFS_clearSolver(timeCtrl))
        
    cmds.showWindow()

def MFS_delete_menu():
    if cmds.menu("MFS_menu", exists=True):
        cmds.deleteUI("MFS_menu", menu=True)


def MFS_deleteSolver(*args):
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
    
    active_object = selected_objects[0]

    for solver in solvers:
        if (solver.source_object == active_object):
            solvers.remove(solver)
    
    if (cmds.objExists(f"MFS_PARTICLES_{active_object}")):
        cmds.delete(f"MFS_PARTICLES_{active_object}")
    
    if (cmds.objExists(f"MFS_DOMAIN_{active_object}")):
        cmds.delete(f"MFS_DOMAIN_{active_object}")
    

def MFS_initializeSolver(pscaleCtrl, domainCtrl, *args):
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
    
    keepDomain = cmds.checkBox(domainCtrl, query=True, value=True)

    domain = None

    for solver in solvers:
        if (solver.source_object == active_object):
            if (keepDomain and solver.domain_object is not None):
                domain = solver.domain_object
            solver.clear(keepDomain)
            solvers.remove(solver)

    solver = MFS_Solver()
    solver.source_object = active_object
    solvers.append(solver)

    bounding_box = cmds.exactWorldBoundingBox(solver.source_object)
    min_point = om.MPoint(bounding_box[0], bounding_box[1], bounding_box[2])
    max_point = om.MPoint(bounding_box[3], bounding_box[4], bounding_box[5])

    if (domain is None or not keepDomain):
        domain = cmds.polyCube(width=3 * (max_point[0] - min_point[0]), height=3 * (max_point[1] - min_point[1]), depth=3 * (max_point[2] - min_point[2]), name=f"MFS_DOMAIN_{active_object}")[0]
        solver.domain_object = domain
        cmds.setAttr(solver.domain_object + ".translateX", cmds.getAttr(solver.source_object + ".translateX"))
        cmds.setAttr(solver.domain_object + ".translateY", cmds.getAttr(solver.source_object + ".translateY"))
        cmds.setAttr(solver.domain_object + ".translateZ", cmds.getAttr(solver.source_object + ".translateZ"))

        cmds.setAttr(solver.domain_object + '.overrideEnabled', 1)
        cmds.setAttr(solver.domain_object + '.overrideShading', 0)
    else:
        solver.domain_object = domain


    cmds.setAttr(solver.source_object + '.overrideEnabled', 1)
    cmds.setAttr(solver.source_object + '.overrideShading', 0)
    
    pscale = cmds.floatSliderGrp(pscaleCtrl, query=True, value=True)

    solver.point_distribute(pscale)
    solver.initialized = True

    cmds.select(solver.source_object)


def MFS_runSolver(timeCtrl, forceCtrl, viscCtrl, velCtrl, tsCtrl, *args):
    selected_objects = cmds.ls(selection=True)
    frameRange = cmds.intFieldGrp(timeCtrl, query=True, value=True)
    force = cmds.floatFieldGrp(forceCtrl, query=True, value=True)
    scale = cmds.floatSliderGrp(tsCtrl, query=True, value=True)
    viscosity = cmds.floatSliderGrp(viscCtrl, query=True, value=True)
    velocity = cmds.floatFieldGrp(velCtrl, query=True, value=True)

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
    
    cmds.currentTime(frameRange[0], edit=True)

    for solver in solvers:
        if solver.source_object == active_object:
            index = solvers.index(solver)
            solver.solved = False
            cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frameRange[1]-frameRange[0]))
            MFS_solve(index, 0, frameRange[0], frameRange[1], force, velocity, viscosity, scale)

def MFS_solve(index, progress, start, end, force, velocity, viscosity, scale):        
    solver = solvers[index]

    t = int(cmds.currentTime(query=True))

    if cmds.progressWindow( query=True, isCancelled=True ):
        cmds.progressWindow(endProgress=1)
        solver.solved = False
        return
    
    if (t < start or t > end): 
        cmds.progressWindow(endProgress=1)
        solver.solved = True
        return
    
    solver.update(start, end, force, velocity, viscosity, scale)
    progress += 1

    cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')
    cmds.currentTime(t + 1, edit=True)
    MFS_solve(index, progress, start, end, force, velocity, viscosity, scale)

def MFS_clearSolver(timeCtrl):
    selected_objects = cmds.ls(selection=True)
    active_object = selected_objects[0]

    frameRange = cmds.intFieldGrp(timeCtrl, query=True, value=True)

    for solver in solvers:
        if solver.source_object == active_object:
            solver.clearSim(frameRange[0], frameRange[1])


def divergence(value):
    return 1

if __name__ == "__main__":
    MFS_create_menu()