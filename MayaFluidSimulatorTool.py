from maya import cmds
from maya.api import OpenMaya as om
import os, math
import numpy as np
from collections import defaultdict

class MFS_Plugin():
    project_path = cmds.workspace(query=True, rootDirectory=True)
    solvers = np.array([])
    # make this a dictionary
    MFS_objects = dict({
            "domain":"MFS_DOMAIN",
            "particles":"MFS_PARTICLES"
        }
    )

    popup_width = 500
    popup_height = 600
    button_ratio = 0.9

    pscale_ctrl = None
    domain_ctrl = None

    force_ctrl = None
    visc_ctrl = None
    vel_ctrl = None
    time_ctrl = None
    ts_ctrl = None

    density_ctrl = None
    kfac_ctrl = None
    search_ctrl = None
    smooth_ctrl = None
    bounce_ctrl = None
    mass_ctrl = None
    cells_ctrl = None

    def __init__(self):
        self.MFS_create_menu()

    def MFS_create_menu(self):
        self.MFS_delete_menu()
        cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)
        cmds.menuItem(label="Open Maya Fluid Simulator", command=lambda x:self.MFS_popup(), image=os.path.join(self.project_path, "icons/MFS_icon_solver_512.png"))

    def MFS_popup(self):
        cmds.window(title="Maya Fluid Simulator", widthHeight=(self.popup_width, self.popup_height))
        col = cmds.columnLayout(adjustableColumn=True)

        cmds.image(width=self.popup_width/2, height=self.popup_height/4, image=os.path.join(self.project_path, "icons/MFS_banner.png"))

        initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)

        cmds.columnLayout(adjustableColumn=True, parent=initialize_section)

        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.25, field=True, label="Particle Scale")
        self.domain_ctrl = cmds.checkBox(label="Keep Domain", value=True)
        
        init_row = cmds.rowLayout(numberOfColumns=2, parent=initialize_section, adjustableColumn=True)
        cmds.button(label="Initialize", command=lambda x:self.MFS_initializeSolver())
        cmds.button(label="X", command=lambda x:self.MFS_deleteSolver())
        cmds.rowLayout(init_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        simulate_section = cmds.frameLayout(label='Simulate', collapsable=True, collapse=False, parent=col)

        cmds.columnLayout(adjustableColumn=True, parent=simulate_section)

        self.force_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Force', extraLabel='cm', value1=0, value2=-980, value3=0 )
        self.visc_ctrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=0.5, field=True, label="Viscosity")
        self.vel_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )
        
        cmds.rowLayout(numberOfColumns=3)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=1, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.01, field=True, label="Time Scale")

        advanced_section = cmds.frameLayout(label='Advanced', collapsable=True, collapse=False, parent=col)

        cmds.columnLayout(adjustableColumn=True, parent=advanced_section)

        self.mass_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=1, field=True, label="Particle Mass")
        self.density_ctrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=998.2, maxValue=10000, field=True, label="Rest Density")
        self.kfac_ctrl = cmds.floatSliderGrp(minValue=0, step=0.1, value=10, maxValue=100000000000, field=True, label="K Factor")
        self.search_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.8, field=True, label="Search Distance")
        self.smooth_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=1, field=True, label="Velocity Smoothing")
        self.bounce_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.01, field=True, label="Floor Bounce")
        self.cells_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=1, field=True, label="Cell Size")

        solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)
        cmds.button(label="Solve", command=lambda x:self.MFS_runSolver())
        cmds.button(label="X", command=lambda x:self.MFS_clearSolver())
        cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        cmds.columnLayout(adjustableColumn=True, parent=col)
            
        cmds.showWindow()

    def MFS_delete_menu(self):
        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)

    def MFS_initializeSolver(self):
        active_object = self.get_active_object()
        
        keepDomain = cmds.checkBox(self.domain_ctrl, query=True, value=True)
        domain = None

        for solver in self.solvers:
            if (solver.source_object == active_object):
                if (keepDomain and solver.domain_object is not None):
                    domain = solver.domain_object
                solver.clear(keepDomain)
                self.solvers = np.delete(self.solvers, [solver])

        solver = MFS_Solver()
        solver.source_object = active_object
        self.solvers = np.append(self.solvers, np.array(solver))

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
        
        pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)

        solver.initialize(pscale)

        cmds.select(solver.source_object)

    def MFS_runSolver(self):
        frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
        force = cmds.floatFieldGrp(self.force_ctrl, query=True, value=True)
        time_scale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
        viscosity = cmds.floatSliderGrp(self.visc_ctrl, query=True, value=True)
        velocity = cmds.floatFieldGrp(self.vel_ctrl, query=True, value=True)
        rest_density = cmds.floatSliderGrp(self.density_ctrl, query=True, value=True)
        kfac = cmds.floatSliderGrp(self.kfac_ctrl, query=True, value=True)
        search_dist = cmds.floatSliderGrp(self.search_ctrl, query=True, value=True)
        vel_smooth = cmds.floatSliderGrp(self.smooth_ctrl, query=True, value=True)
        floor_bounce = cmds.floatSliderGrp(self.bounce_ctrl, query=True, value=True)
        mass = cmds.floatSliderGrp(self.mass_ctrl, query=True, value=True)
        cell_size = cmds.floatSliderGrp(self.cells_ctrl, query=True, value=True)

        active_object = self.get_active_object()
        
        cmds.currentTime(frame_range[0], edit=True)

        for solver in self.solvers:
            if solver.source_object == active_object:
                solver.clearSim(frame_range[0])
                solver.solved = False
                cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frame_range[1]-frame_range[0]))
                self.MFS_solve(solver, 0, frame_range, force, velocity, viscosity, time_scale, rest_density, kfac, search_dist, vel_smooth, floor_bounce, mass, cell_size)

    def MFS_solve(self, solver, progress, frame_range, force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, mass, cell_size):        
        t = int(cmds.currentTime(query=True))

        solver.solved = (t < frame_range[0] or t > frame_range[1])

        if cmds.progressWindow( query=True, isCancelled=True) or solver.solved:
            cmds.progressWindow(endProgress=1)
            return
        
        solver.update(frame_range[0], force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, mass, cell_size)
        progress += 1

        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')
        cmds.currentTime(t + 1, edit=True)
        self.MFS_solve(solver, progress, frame_range, force, velocity, viscosity, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, mass, cell_size)

    def MFS_clearSolver(self):
        active_object = self.get_active_object()

        frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)

        for solver in self.solvers:
            if solver.source_object == active_object:
                solver.clearSim(frame_range[0])

    def MFS_deleteSolver(self):
        active_object = self.get_active_object()
    
        for solver in self.solvers:
            if (solver.source_object == active_object):
                self.solvers = np.delete(self.solvers, np.where(self.solvers == solver)[0])

        for obj in self.MFS_objects.values():
            if (cmds.objExists(f"{obj}_{active_object}")):
                cmds.delete(f"{obj}_{active_object}")

    def get_active_object(self):
        selected_objects = cmds.ls(selection=True)
        active_object = None

        if (selected_objects):
            active_object = selected_objects[0]

            if (cmds.objectType(active_object) == "transform" and obj not in active_object for obj in self.MFS_objects.values()):
                return active_object
                
        cmds.confirmDialog(title="Solver Error!", 
            message="You need to use the solver on an object!",
            button="Sorry"
        )
        
        return None

class MFS_Particle():
    def __init__(self):
        self.id = -1
        self.initial = np.zeros((2, 3))
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.mass = 0
        self.density = 0
        self.pressure = 0
        self.neighbor_ids = np.array([])
        self.total_force = np.zeros(3)

    def speed(self):
        return np.linalg.norm(self.velocity)

class MFS_Solver():
    points = np.array([])
    source_object = None
    domain_object = None
    solved = False
    initialized = False
    pscale = 0
    cell_size = 0
    hash_table = defaultdict(list)

    def initialize(self, pscale):
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

                    self.points = np.append(self.points, np.array(pnt))

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


        cmds.progressWindow(endProgress=1)
        self.initialized = True

    def update(self, start, other_force, initial_velocity, viscosity_factor, scale, rest_density, kfac, search_dist, vel_smooth, floor_damping, mass, cells):
        t = (int(cmds.currentTime(query=True)) - start)

        bounding_box = cmds.exactWorldBoundingBox(self.domain_object)

        self.cell_size = cells

        if (not self.solved):
            if (t==0):
                for p in self.points:
                    p.position = p.initial[0]
                    p.velocity = initial_velocity
                    p.initial[1] = p.velocity

                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=t+start, v=p.position[0])
                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=t+start, v=p.position[1])
                    cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=t+start, v=p.position[2])
            else:
                self.key_position(start + t, scale)

                h = self.find_neighbors(search_dist, mass, bounding_box)

                self.calc_density_and_pressure(h, rest_density, kfac)

                self.calc_forces(other_force, viscosity_factor, h)

                self.calc_velocity(bounding_box, scale, h, vel_smooth, floor_damping)

    def find_neighbors(self, search_dist, mass, bbox):
        for p in self.points:
            p.neighbor_ids = np.array([])
            p.total_force = np.zeros(3)
            p.mass = mass

            pics = self.get_particles_in_same_cell(p.position)
            for j in pics:
                
                dist = np.linalg.norm(np.subtract(p.position, j[0]))
                
                if (dist <= search_dist):
                    p.neighbor_ids = np.append(p.neighbor_ids, np.array(j[1]))
            
        return search_dist


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

                    pressure_grad = wspiky_grad(j_to_p, h)
                    pressure_term = ((j.pressure + p.pressure)/2) * (j.mass / j.density)

                    pressure_force = np.add(pressure_force, np.multiply(pressure_grad, pressure_term))

                    velocity_diff = np.subtract(j.velocity, p.velocity)

                    viscosity_term = wvisc_lap(j_to_p, h) * (j.mass / j.density)

                    viscosity_force = np.add(viscosity_force, np.multiply(velocity_diff, viscosity_term))

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

            p.velocity = np.add(p.velocity, np.multiply(xsph_term, vel_smooth))
            
            if (p.position[0] + p.velocity[0] * scale < min_point[0] or p.position[0] + p.velocity[0] * scale > max_point[0]):
                p.velocity[0] = -p.velocity[0] 

            if (p.position[1] + p.velocity[1] * scale < min_point[1]):
                p.velocity[1] = -(p.position[1] + p.velocity[1] * scale - min_point[0]) * floor_damping

            if (p.position[1] + p.velocity[1] * scale > max_point[1]):
                p.velocity[1] = -p.velocity[1]
                                
            if (p.position[2] + p.velocity[2] * scale < min_point[2] or p.position[2] + p.velocity[2] * scale > max_point[2]):
                p.velocity[2] = -p.velocity[2]
        

    def key_position(self, frame, scale):        
        for p in self.points:
            p.position = np.add(p.position, np.multiply(p.velocity, scale))

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=frame, v=p.position[0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=frame, v=p.position[1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=frame, v=p.position[2])

        self.update_hashmap()

    def get_hash_id(self, position):
        return int(position[0]/self.cell_size) + self.cell_size * (int(position[1]/self.cell_size) + self.cell_size * int(position[2]/self.cell_size))

    def add_to_hashmap(self, particle):
        id = self.get_hash_id(particle.position)
        self.hash_table[id].append([particle.position, particle.id])

    def get_particles_in_same_cell(self, position):
        id = self.get_hash_id(position)
        return self.hash_table[id]

    def update_hashmap(self):
        self.hash_table = defaultdict(list)
        for particle in self.points:
            self.add_to_hashmap(particle)

    def clear(self, keepDomain):
        if (self.initialized):
            if (self.domain_object is not None and cmds.objExists(self.domain_object) and not keepDomain):
                cmds.delete(self.domain_object)

        cmds.setAttr(self.source_object + '.overrideShading', 1)
        cmds.setAttr(self.source_object + '.overrideEnabled', 0)
        
        self.points = np.array([])
        self.source_object = None
        self.domain_object = None
        self.solved = False
        self.initialized = False

    def clearSim(self, start):
        for p in self.points:
            p.position = p.initial[0]
            p.velocity = p.initial[1]
            p.solved = False

            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', clear=True)
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', clear=True )
            cmds.cutKey(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', clear=True )

            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateX', t=start, v=p.position[0])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateY', t=start, v=p.position[1])
            cmds.setKeyframe(f"MFS_PARTICLE_{self.source_object}_{p.id:05}", attribute='translateZ', t=start, v=p.position[2])
            

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
        return np.multiply(r, term)
    else:
        return np.zeros(3)

def wpoly_6_lap(r, h):
    dist = np.linalg.norm(r)

    if (dist <= h):
        return (-945/(32*math.pi*h**9)) * (h**2 - dist**2) * (3*h**2 - 7*dist**2)
    else:
        return 0

def wspiky_grad(r, h):
    dist = np.linalg.norm(r)

    if (dist <= h):
        term = -45/(math.pi * h**6) * (h-dist)**2
        return np.multiply(np.divide(r, dist), term)
    else:
        return np.zeros(3)

def wvisc_lap(r, h):
    dist = np.linalg.norm(r)
    if (dist <= h):
        return 45/(math.pi*h**6) * (h-dist)
    else:
        return 0

if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()
