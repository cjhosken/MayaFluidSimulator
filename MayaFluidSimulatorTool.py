from maya import cmds
from maya.api import OpenMaya as om
import os
import math
import numpy as np

# ------ MENUS & BUTTON FUNCTIONS------

def MFS_create_menu():

    MFS_delete_menu()

    cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)

    project_path = cmds.workspace(query=True, rootDirectory=True)

    cmds.menuItem(label="Open Maya Fluid Simulator", command=MFS_popup, image=os.path.join(project_path, "icons/MFS_icon_solver_512.png"))

def MFS_popup(*args):

    project_path = cmds.workspace(query=True, rootDirectory=True)

    cmds.window(title="Maya Fluid Simulator", widthHeight=(500, 600))
    col = cmds.columnLayout(adjustableColumn=True)

    cmds.image(width=300, height=150, image=os.path.join(project_path, "icons/MFS_banner.png"))

    initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)

    cmds.columnLayout(adjustableColumn=True, parent=initialize_section)

    pscaleCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.25, field=True, label="Particle Scale")
    domainCtrl = cmds.checkBox(label="Keep Domain", value=True)
    
    init_row = cmds.rowLayout(numberOfColumns=2, parent=initialize_section, adjustableColumn=True)
    cmds.button(label="Initialize", command=lambda *args:MFS_initializeSolver(pscaleCtrl, domainCtrl))
    cmds.button(label="X", command=lambda *args:MFS_deleteSolver())
    cmds.rowLayout(init_row, edit=True, columnWidth=[(1, 450), (2, 50)])

    simulate_section = cmds.frameLayout(label='Simulate', collapsable=True, collapse=False, parent=col)

    cmds.columnLayout(adjustableColumn=True, parent=simulate_section)

    # gravity
    forceCtrl = cmds.floatFieldGrp( numberOfFields=3, label='Force', extraLabel='cm', value1=0, value2=-9.8, value3=0 )

    # viscosity
    viscCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0, field=True, label="Viscosity")

    # velocity
    velCtrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )
    
    cmds.rowLayout(numberOfColumns=3)
    timeCtrl = cmds.intFieldGrp(numberOfFields=2, value1=1, value2=120, label="Frame Range")
    tsCtrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.1, field=True, label="Time Scale")

    advanced_section = cmds.frameLayout(label='Advanced', collapsable=True, collapse=False, parent=col)

    cmds.columnLayout(adjustableColumn=True, parent=advanced_section)

    densityCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=4.8, maxValue=10000, field=True, label="Rest Density")
    kFacCtrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=10, field=True, label="K Factor")
    searchCtrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.5, field=True, label="Search Distance")
    smoothCtrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0, field=True, label="Velocity Smoothing")
    dampCtrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.01, field=True, label="Floor Damping")
    minVelCtrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, field=True, label="Minimum Velocity")
    massCtrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.001, field=True, label="Particle Mass")

    solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)
    cmds.button(label="Solve", command=lambda *args:MFS_runSolver(timeCtrl, forceCtrl, viscCtrl, velCtrl, tsCtrl, densityCtrl, kFacCtrl, searchCtrl, smoothCtrl, dampCtrl, minVelCtrl, massCtrl))
    cmds.button(label="X", command=lambda *args:MFS_clearSolver(timeCtrl))
    cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, 450), (2, 50)])

    cmds.columnLayout(adjustableColumn=True, parent=col)
        
    cmds.showWindow()

def MFS_delete_menu():
    if cmds.menu("MFS_menu", exists=True):
        cmds.deleteUI("MFS_menu", menu=True)

# ------ CLASSES ------

class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.initial = np.zeros((2, 3))
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.mass = 0.001
        self.density = 0
        self.pressure = 0
        self.neighbor_ids = np.array()
        self.total_force = np.zeros(3)

    def speed(self):
        return np.linalg.norm(self.velocity)

# GLOBAL VARIABLE
solvers = []

class MFS_Solver():
    points = np.array()
    source_object = None
    domain_object = None
    solved = False
    initialized = False
    volume = 0
    pscale = 0

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

                    pnt.position = np.array([ix, iy, iz])
                    pnt.velocity = np.zeros(3)
                    pnt.id = i

                    pnt.initial = np.array([pnt.position, pnt.velocity])

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
            
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=0, v=p.position[0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=0, v=p.position[1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=0, v=p.position[2])
            
            progress += 1

            cmds.parent(sphere_name, transform_node)
            cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')

        self.volume = (max_point[0] - min_point[0]) * (max_point[1] - min_point[1]) * (max_point[2] - min_point[2])
        cmds.progressWindow(endProgress=1)

    def update(self, start, end, other_force, init_vel, viscosity_factor, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, max_vel, mass):
        t = (int(cmds.currentTime(query=True)) - start)

        bounding_box = cmds.exactWorldBoundingBox(self.domain_object)

        # taken from https://nccastaff.bournemouth.ac.uk/jmacey/MastersProject/MSc16/15/thesis.pdf
        # https://eprints.bournemouth.ac.uk/23384/1/2016%20Fluid%20simulation.pdf
        # https://nccastaff.bournemouth.ac.uk/jmacey/OldWeb/MastersProjects/MSc15/06Burak/BurakErtekinMScThesis.pdf


        if (not self.solved):
            if (t==0):
                for p in self.points:
                    p.position = p.initial[0]
                    p.velocity = p.initial[1]

                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=t+start, v=p.position[0])
                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=t+start, v=p.position[1])
                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=t+start, v=p.position[2])
            else:
                self.update_position(start, t, scale, bounding_box)

                h = self.find_neighbors(search_dist, mass)

                self.calc_density_and_pressure(h, rest_density, kfac)

                self.calc_forces(other_force, viscosity_factor, h)

                self.calc_velocity(bounding_box, scale, h, vel_smooth, floor_damping, max_vel)

    def find_neighbors(self, search_dist, mass):
        # TODO: The current neighbor search is to check points within a certain radius. However hashmaps are much faster. Look into implementing that.
        max_dist = 0

        for p in self.points:
            p.neighbor_ids = np.array()
            p.total_force = np.zeros(3)
            p.mass = mass

            for j in self.points:
                
                dist = np.linalg.norm(np.subtract(p.position, j.position))

                if (dist < search_dist):
                    p.neighbor_ids.append(j.id)
                    max_dist = max(dist, max_dist)
            
        return max_dist


    def calc_density_and_pressure(self, h, rest_density, kfac):

        for p in self.points:
            p.density = rest_density

            for j in self.points:
                if (j.id in p.neighbor_ids):
                    j_to_p = np.subtract(p.position, j.position)

                    p.density += j.mass * wpoly_6(j_to_p, h)

            p.pressure = kfac * p.density

    def calc_forces(self, other_force, viscosity_factor, h):
        for p in self.points:
            pressure_force = np.array([0, 0, 0])
            viscosity_force = np.array([0, 0, 0])
            external_force = np.array([0, 0, 0])

            for j in self.points:
                if (j.id in p.neighbor_ids and j.id != p.id):
                    j_to_p = np.subtract(p.position, j.position)

                    pressure_term = wspiky_grad(j_to_p, h)
                    pressure_const = (j.pressure + p.pressure)/2*(j.mass / j.density)

                    pressure_force = np.add(pressure_force, np.multiply(pressure_term, pressure_const))

                
                    velocity_diff = np.subtract(j.velocity, p.velocity)

                    viscosity_term = j.mass / j.density
                    viscosity_smoothing = wvisc_lap(j_to_p, h)

                    viscosity_force = np.add(viscosity_force, np.multiply(velocity_diff, viscosity_term * viscosity_smoothing))

            pressure_force = np.multiply(pressure_force, -1)
            viscosity_force = np.multiply(viscosity_force, viscosity_factor)
            external_force = np.multiply(other_force, p.mass)

            p.total_force = np.add(np.add(pressure_force, viscosity_force), external_force)
        
    def calc_velocity(self, bbox, scale, h, vel_smooth, floor_damping):
        min_point = om.MPoint(bbox[0], bbox[1], bbox[2])
        max_point = om.MPoint(bbox[3], bbox[4], bbox[5])

        for p in self.points:
            p.velocity = np.add(p.velocity, np.multiply(p.total_force, scale / p.mass))

            xsph_term = np.zeros(3)

            for j in self.points:
                j_to_p = np.subtract(p.position, j.position)
                velocity_diff = np.subtract(j.velocity, p.velocity)

                xsph_term = np.add(xsph_term, np.multiply(velocity_diff, ((2 * j.mass) / (p.density + j.density)) * wpoly_6(j_to_p, h)))

            p.velocity = np.add(np.multiply(xsph_term, vel_smooth))
            
            if (p.position[0] + p.velocity[0] * scale < min_point[0] or p.position[0] + p.velocity[0] * scale > max_point[0]):
                p.velocity[0] = -p.velocity[0] 

            if (p.position[1] + p.velocity[1] * scale < min_point[1]):
                p.velocity[1] = -p.velocity[1] * floor_damping

            if (p.position[1] + p.velocity[1] * scale > max_point[1]):
                p.velocity[1] = -p.velocity[1]
                                
            if (p.position[2] + p.velocity[2] * scale < min_point[2] or p.position[2] + p.velocity[2] * scale > max_point[2]):
                p.velocity[2] = -p.velocity[2]
        

    def update_position(self, start, t, scale):        
        for p in self.points:
            p.position = np.add(np.multiply(p.velocity, scale))

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=t+start, v=p.position[0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=t+start, v=p.position[1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=t+start, v=p.position[2])

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
            p.velocity = p.initial[1]
            p.position = p.position[0]

            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', clear=True)
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', clear=True )
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', clear=True )

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=start, v=p.position[0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=start, v=p.position[1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=start, v=p.position[2])

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

def MFS_runSolver(timeCtrl, forceCtrl, viscCtrl, velCtrl, tsCtrl, densityCtrl, kFacCtrl, searchCtrl, smoothCtrl, dampCtrl, minVelCtrl, massCtrl, *args):
    selected_objects = cmds.ls(selection=True)
    frameRange = cmds.intFieldGrp(timeCtrl, query=True, value=True)
    force = cmds.floatFieldGrp(forceCtrl, query=True, value=True)
    scale = cmds.floatSliderGrp(tsCtrl, query=True, value=True)
    viscosity = cmds.floatSliderGrp(viscCtrl, query=True, value=True)
    velocity = cmds.floatFieldGrp(velCtrl, query=True, value=True)
    rest_density = cmds.floatSliderGrp(densityCtrl, query=True, value=True)
    kfac = cmds.floatSliderGrp(kFacCtrl, query=True, value=True)
    search_dist = cmds.floatSliderGrp(searchCtrl, query=True, value=True)
    vel_smooth = cmds.floatSliderGrp(smoothCtrl, query=True, value=True)
    floor_damping = cmds.floatSliderGrp(dampCtrl, query=True, value=True)
    min_vel = cmds.floatSliderGrp(minVelCtrl, query=True, value=True)
    mass = cmds.floatSliderGrp(massCtrl, query=True, value=True)

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
            solver.clearSim(frameRange[0], frameRange[1])
            solver.solved = False
            cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frameRange[1]-frameRange[0]))
            MFS_solve(index, 0, frameRange[0], frameRange[1], force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, min_vel, mass)

def MFS_solve(index, progress, start, end, force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, min_vel, mass):        
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
    
    solver.update(start, end, force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, min_vel, mass)
    progress += 1

    cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')
    cmds.currentTime(t + 1, edit=True)
    MFS_solve(index, progress, start, end, force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, min_vel, mass)

def MFS_clearSolver(timeCtrl):
    selected_objects = cmds.ls(selection=True)
    active_object = selected_objects[0]

    frameRange = cmds.intFieldGrp(timeCtrl, query=True, value=True)

    for solver in solvers:
        if solver.source_object == active_object:
            solver.clearSim(frameRange[0], frameRange[1])

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


# Maths Functions

def wpoly_6(r, h):
    dist = np.linalg.norm(r)

    if (dist <= h):
        return (315/(64*math.pi*h**9)) * (h**2 - dist**2)**3
    else:
        return 0

def wpoly_6_grad(r, h):
    dist = np.linalg.norm(r)

    if (dist <= h):
        term = (-945/(32*math.pi*h**9)) * (h**2 - dist**2)**2
        return np.mult(r, term)
    else:
        return np.zeros(3)

def wpoly_6_lap(r, h):
    dist = np.linalg.norm(r)

    if (dist <= h):
        return (-945/(32*math.pi*h**9)) * (h**2 - dist**2) * (3*h**2 - 7*dist**2)
    else:
        return 0

def wspiky_grad(r, h):
    dist = np.subtract(r)

    if (dist <= h):
        term = -45/(math.pi * h**6) * (h-dist)**2
        return np.mult(np.divide(r[0], dist), term)
    else:
        return np.zeros(3)

def wvisc_lap(r, h):
    dist = np.linalg.norm(r)
    if (dist <= h):
        return 45/(math.pi*h**6) * (h-dist)
    else:
        return 0

def divergence(r):
    return 

if __name__ == "__main__":
    MFS_create_menu()
