''' Maya Fluid Simulator is a tool used to simulate PIC/FLIP particles.

    Users are able to source custom shaped objects and simulate them in a simple box domain. The script is split into 2 parts.

    The first part, MFS_Plugin() is the general tool that interacts with Maya. Here you can find code for the user interface, point sourcing inside objects, and setup for the simulation.

    The second, MFS_Particle() and MFS_Grid() are where more of the technical simulation code is held. If you're looking into doing fluid simulations, that area will be more of interest.
'''

# Imports
from maya import cmds
from maya.api import OpenMaya as om
import os
import random
import re

''' Maya Fluid Simulator uses numpy. Users will need to manually install numpy using the commands below.

    Windows:
    "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy

    Linux:
    /usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy

    Mac:
    /Applications/Autodesk/maya2023/Maya.app/Contents/bin/mayapy -m pip install –user numpy


    Make sure that the correct maya version is being used.    
'''

import numpy as np

''' The MFS Plugin is a general class that contains the (mainly) Maya side of the script. This allows the use of "global" variables in a contained setting. '''
class MFS_Plugin():
    
    # "global" variables
    project_path = cmds.workspace(query=True, rootDirectory=True)
    popup_size = (500, 600)
    button_ratio = 0.9


    ''' __init__ initializes the plugin by creating the header menu. '''
    def __init__(self):
        self.MFS_create_menu()


    ''' MFS_create_menu deletes any pre-existing Maya Fluid Simulator header menus, then creates a new header menu.'''
    def MFS_create_menu(self):
        self.MFS_delete_menu()
        cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)
        cmds.menuItem(label="Open Maya Fluid Simulator", command=lambda x:self.MFS_popup(), image=os.path.join(self.project_path, "icons/MFS_icon_solver_512.png"))
    

    ''' MFS_popip creates the main UI for Maya Fluid Simulator. 

        Particle Scale         : The visual size of the particles
        Cell Size              : The cell width for particle sourcing and simulation
        Random Sampling        : The sampling value used for generating particles
        Domain Size            : The size of the domain
        Keep Domain            : Whether or not to replace the sourced domain
        Initialize (X)         : Initialize the particles and create the fluid domain. X will remove the particles and domain.

        Force                  : External forces acting on the fluid. Usually gravity.
        Initial Velocity       : The initial velocity of the fluid particles.
        Fluid Density*         : The density of the fluid
        Viscosity Factor*      : The amount of viscosity that the fluid has. (Water) -> (Honey)
        Floor Damping          : The amount of damping when particles collide with the floor
        PIC/FLIP Mix           : The blending from PIC (0) -> (1) FLIP. PIC is often better for viscious fluids, while FLIP is good for splashy fluids.
        Frame Range            : The range in the which simulation should run between.
        Time Scale             : The speed of the simulation.
        Simulate (X)           : Simulate the fluid particles. X will remove the keyframed simulation. 

        Setting the Maya project dir to the script location shows the plugin icons.
    '''
    def MFS_popup(self):
        
        cmds.window(title="Maya Fluid Simulator", widthHeight=self.popup_size)
        col = cmds.columnLayout(adjustableColumn=True)

        cmds.image(width=self.popup_size[0]/2, height=self.popup_size[1]/4, image=os.path.join(self.project_path, "icons/MFS_banner.png"))

        initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)
        cmds.columnLayout(adjustableColumn=True, parent=initialize_section)
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.25, field=True, label="Particle scale")
        self.cell_size_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.25, field=True, label="Cell Size")
        self.random_sample_ctrl = cmds.intSliderGrp(minValue=0, value=0, field=True, label="Random Sampling")

        self.domain_size_ctrl = cmds.floatFieldGrp(numberOfFields=3, label='Domain Size', extraLabel='cm', value1=5, value2=5, value3=5)

        self.domain_ctrl = cmds.checkBox(label="Keep Domain", value=True)
    
        init_row = cmds.rowLayout(numberOfColumns=2, parent=initialize_section, adjustableColumn=True)
        cmds.button(label="Initialize", command=lambda x:self.MFS_initialize())
        cmds.button(label="X", command=lambda x:self.MFS_delete())
        cmds.rowLayout(init_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_size[0]), (2, (1-self.button_ratio) * self.popup_size[0])])

        simulate_section = cmds.frameLayout(label='Simulate', collapsable=True, collapse=False, parent=col)
        self.force_ctrl = cmds.floatFieldGrp(numberOfFields=3, label='Force', extraLabel='cm', value1=0, value2=-9.8, value3=0 )
        self.vel_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )

        self.density_ctrl = cmds.floatSliderGrp(minValue=0, maxValue=2000, step=0.01, value=998.2, field=True, label="Fluid Density")
        self.stiff_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=1, field=True, label="Stiffness")
        self.relax_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=1.5, field=True, label="Overrelaxation")
        self.iter_ctrl = cmds.intSliderGrp(minValue=0, value=5, field=True, label="Iterations")

        self.picflip_ctrl = cmds.floatSliderGrp(minValue=0, maxValue=1.0, step=0.01, value=0.5, field=True, label="PIC/FLIP Mix")
        
        cmds.rowLayout(numberOfColumns=2)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.1, field=True, label="Time Scale")
        

        solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)

        cmds.button(label="Simulate", command=lambda x:self.MFS_simulate())
        cmds.button(label="X", command=lambda x:self.MFS_reset())
        cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_size[0]), (2, (1-self.button_ratio) * self.popup_size[0])])

        cmds.columnLayout(adjustableColumn=True, parent=col)

        cmds.showWindow()


    ''' MFS_delete_menu checks if the Maya Fluid Simulator header menu exists and deletes it. '''
    def MFS_delete_menu(self):
        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)


    ''' MFS_initialize uses the initialization settings to fill a selected object with fluid particles and to create a domain object. '''
    def MFS_initialize(self):
        source = self.get_active_object()
        
        if (source is not None):
            cmds.setAttr(source + ".overrideEnabled", 1)
            cmds.setAttr(source + ".overrideShading", 0)

            keep_domain = cmds.checkBox(self.domain_ctrl, query=True, value=True)
            domain_name = source + "_domain"
            domain = cmds.objExists(domain_name)

            cell_size = cmds.floatSliderGrp(self.cell_size_ctrl, query=True, value=True)
            pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)

            # Check if the domain exists and whether it needs to be replaced or not.
            if (not keep_domain or not domain):
                if (domain):
                    cmds.delete(domain_name)
                
                domain_size = cmds.floatFieldGrp(self.domain_size_ctrl, query=True, value=True)
                domain_size[0] = abs(domain_size[0])
                domain_size[1] = abs(domain_size[1])
                domain_size[2] = abs(domain_size[2])

                domain_divs = (
                    int(domain_size[0] / cell_size),
                    int(domain_size[1] / cell_size),
                    int(domain_size[2] / cell_size)
                )

                domain = cmds.polyCube(name=domain_name, w=domain_size[0], h=domain_size[1], d=domain_size[2], sx=domain_divs[0], sy=domain_divs[1], sz=domain_divs[2])
                cmds.setAttr(domain[0] + ".overrideEnabled", 1)
                cmds.setAttr(domain[0] + ".overrideLevelOfDetail", 1)
            
            samples = cmds.intSliderGrp(self.random_sample_ctrl, query=True, value=True)

            # mesh_to_points returns a list of vector points, not maya objects.
            points = self.mesh_to_points(source, pscale, samples)

            # Create the particle group
            particle_group_name = source + "_particles"

            if (cmds.objExists(particle_group_name)):
                cmds.delete(particle_group_name)

            particle_group = cmds.createNode("transform", name=particle_group_name)

            # Convert the mesh_to_points points into physical particles.
            for i,p in enumerate(points):
                particle_name = f"{source}_particle_{i:09}"
                if (cmds.objExists(particle_name)):
                    cmds.delete(particle_name)
                
                particle = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=particle_name)
                cmds.xform(particle, translation=p)
                cmds.parent(particle[0], particle_group)
            
            cmds.select(source)


    ''' MFS_simulate begins the Maya Fluid Simulation from the given fluid settings. '''
    def MFS_simulate(self):

        source = self.get_active_object()

        if (source is not None and self.can_simulate(source)):
            self.MFS_reset() # Remove all pre-existing keyframes

            frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
            timescale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
            external_force = cmds.floatFieldGrp(self.force_ctrl, query=True, value=True)

            fluid_density = cmds.floatSliderGrp(self.density_ctrl, query=True, value=True)
            pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)
            flipFac = cmds.floatSliderGrp(self.picflip_ctrl, query=True, value=True)
            iterations = cmds.intSliderGrp(self.iter_ctrl, query=True, value=True)
            overrelaxation = cmds.floatSliderGrp(self.relax_ctrl, query=True, value=True)
            stiffness = cmds.floatSliderGrp(self.stiff_ctrl, query=True, value=True)

            maya_particles = cmds.listRelatives(source + "_particles", children=True) or []
            particles = []

            # To allow the tool to simulate, even after the user has restarted Maya, the plugin looks at all the Maya object particles and creates a MFS_Particle class. 
            # It then adds all the particles to a point array to be used in the simulation.
            for p in maya_particles:
                id = int(re.search(r"\d+$", p).group())
                position = np.array(cmds.xform(p, query=True, translation=True, worldSpace=True), dtype="float64")
                velocity = np.array(cmds.floatFieldGrp(self.vel_ctrl, query=True, value=True), dtype="float64")

                particle = MFS_Particle(
                    id=id,
                    pos=position,
                    vel=velocity
                )

                particles = np.append(particles, particle)

            # Using the size of the domain and its subdivisions, a cell_size can be constructed to be used for simulating.
            bbox = cmds.exactWorldBoundingBox(source + "_domain")
            min_x, min_y, min_z, max_x, max_y, max_z = bbox

            size = np.array([abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)])
            

            resx = int(cmds.polyCube(source + "_domain", query=True, subdivisionsX=True))
            resy = int(cmds.polyCube(source + "_domain", query=True, subdivisionsY=True))
            resz = int(cmds.polyCube(source + "_domain", query=True, subdivisionsZ=True))

            resolution = np.array([resx, resy, resz])
            cell_size = size/resolution

            # A simulation grid is then constructed. This grid will store all the velocity values used for moving the particles.
            grid = MFS_Grid(resolution, cell_size)

            external_force = np.array(external_force, dtype="float64")

            cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frame_range[1]-frame_range[0]))
            
            # The simulation is then properly started in update.
            self.update(source, particles, grid, frame_range, timescale, external_force, fluid_density, pscale, flipFac, iterations, overrelaxation, stiffness, 0)


    ''' update is the start of an actual Fluid Simulator. It simulates the particle position frame by frame, keyframing the positions each time. 
        The user is then able to cancel the simulation by pressing Esc, however they need to wait for the current frame to finish simulation.
    
        source                  : The fluid source object
        particles               : The array containing the MFS_Particles
        grid                    : The MFS_Grid which most of the calculations are done on.
        frame_range             : The range in the which simulation should run between.
        timescale               : The speed of the simulation.
        external_force          : The external forces acting on the fluid. Usually gravity.
        fluid_density*          : The density of the fluid.
        viscosity_factor*       : The amount of fluid viscosity.
        damping                 : The damping factor for particle-ground collisions.
        pscale*                 : The size of the particles.
        flipFac                 : The blending from PIC (0) -> (1) FLIP. 
        progress                : Used to track the progress bar.

        The reason why so many values are being parsed through the update function is to avoid settings changing midway though simulation.
    '''
    def update(self, source, particles, grid, frame_range, timescale, external_force, fluid_density, pscale, flipFac, iterations, overrelaxation, stiffness, progress):
        percent = (progress / (frame_range[1] - frame_range[0])) * 100

        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {percent:.1f}%')
        t = int(cmds.currentTime(query=True))
        solved = (t < frame_range[0] or t > frame_range[1])
        cancelled = cmds.progressWindow(query=True, isCancelled=True)

        # Get the domain bounding box
        bbox = cmds.exactWorldBoundingBox(source + "_domain")

        # This is the start of the simulator. The method is a python implementation of:
        #
        # 1. keyframe frame: copy the particle positions in the simulation onto the maya particle objects.
        # 2. enter the CFL domain. This is done to limit particles travelling only 1 cell at a time.
        # 3. from_particles: transfer the point velocities to the grid using trilinear interpolation.
        # 4. calc_timestep: find the timestep. This will differ from cfl iteration to iteration, but at maximum is the timescale.
        # 5. calc_forces: calculate external forces such as gravity.
        # 6. enforce_boundaries: stop any edge cell velocities from pointing out of the domain. This is done by setting the velocity component to 0.
        # 7. solve_poission: solve the possion equation that makes the fluid divergence free.
        # 8. to_particles: transfer the grid velocities back into the particles, then move the particles.
        #
        # Once the cfl iterations are complete, the next frame is done until all the frames within the frame range are simulated.

    
        if (not (solved or cancelled)):
            self.keyframe(source, particles, t)
            
            print(f"Maya Fluid Simulator | Simulating Frame: {t}")

            cfl = 0

            while(cfl < timescale):
                grid.clear()
                average_density = grid.particles_to_grid(particles, bbox)
                dt = grid.calc_dt(particles, timescale, external_force)
                grid.apply_forces(external_force, dt)
                grid.enforce_boundaries()
                grid.solve_divergence(iterations, overrelaxation, stiffness, average_density)
                grid.grid_to_particles(particles, bbox, flipFac, dt)
                grid.handle_collisions_and_boundary(particles, bbox, pscale)
                cfl += dt            

            cmds.currentTime(t + 1, edit=True)

            self.update(source, particles, grid, frame_range, timescale, external_force, fluid_density, pscale, flipFac, iterations, overrelaxation, stiffness, progress=progress+1)
        else:
            cmds.currentTime(frame_range[0], edit=True)
            cmds.progressWindow(endProgress=1)


    ''' keyframe take the simulation particles, copies their position into the corresponding Maya particles, then keyframes the positions.

        source          : The source object
        particles       : The array containing MFS_Particles
        t               : the current frame value

    '''
    def keyframe(self, source, particles, t):
        for p in particles:
            particle_name = f"{source}_particle_{p.id:09}"
            cmds.setKeyframe(particle_name, attribute='translateX', t=t, v=p.position[0])
            cmds.setKeyframe(particle_name, attribute='translateY', t=t, v=p.position[1])
            cmds.setKeyframe(particle_name, attribute='translateZ', t=t, v=p.position[2])

    ''' MFS_delete deletes all the particles and the domain, and revert back the source object.'''
    def MFS_delete(self):
        source = self.get_active_object()

        if (source is not None):
            cmds.setAttr(source + ".overrideShading", 1)
            cmds.setAttr(source + ".overrideEnabled", 0)

            particle_group_name = source + "_particles"
            domain_name = source + "_domain"

            if (cmds.objExists(particle_group_name)):
                cmds.delete(particle_group_name)

            if (cmds.objExists(domain_name)):
                cmds.delete(domain_name)
    

    ''' MFS_reset resets the simulation by deleting all the keyframes and setting the current time to the start of the frame range.'''
    def MFS_reset(self):
        source = self.get_active_object()
        frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
        cmds.currentTime(frame_range[0], edit=True)

        if (source is not None):
            particles = cmds.listRelatives(source + "_particles", children=True) or []

            for p in particles:
                cmds.cutKey(p, attribute='translateX', clear=True)
                cmds.cutKey(p, attribute='translateY', clear=True)
                cmds.cutKey(p, attribute='translateZ', clear=True)


    ''' get_active_object checks if the selected object can be a "source" object.
        An object can be a source object if it is not a domain or a particle.
        Otherwise None is returned.
    '''
    def get_active_object(self):
        selected_objects = cmds.ls(selection=True)

        if (selected_objects):
            active_object = selected_objects[0]

            forbidden_labels = ["_domain", "_particle"]
            if all(label not in active_object for label in forbidden_labels):
                return active_object
                
        cmds.confirmDialog(title="Source Error!", 
            message="You need to select a source object!",
            button="Oopsies"
        )
        
        return None


    ''' can_simulate checks if the selected object has been sourced. To do so, it looks to see if a domain and particle group exist for the object.
        
        source          : The selected object to test
    
    '''
    def can_simulate(self, source):
        can_simulate = cmds.objExists(source + "_domain") and cmds.objExists(source + "_particles")

        if (not can_simulate):
            cmds.confirmDialog(title="Simulation Error!", 
                message="You need to initialize a source object!",
                button="Oopsies"
            )

        return can_simulate
    

    ''' mesh_to_points generates points inside of a given object.
        It creates a domain around the source object and splits it into subdivisions, then It creates a domain around the source object and splits it into subdivisions. 

        mesh                : the mesh to convert to points
        cell_size           : the spacing between particles. This is a scalar as we want the point generation to be uniform.
        samples             : the number of particles to randomly generate inside a cell. When 0, the particles assume a grid formation.
    
    '''
    def mesh_to_points(self, mesh, cell_size, samples=0):
    
        bbox = cmds.exactWorldBoundingBox(mesh)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        subdivisions = (
            int((max_x - min_x) / cell_size),
            int((max_y - min_y) / cell_size),
            int((max_z - min_z) / cell_size)
        )

        points = []
        for i in range(subdivisions[0]):
            for j in range(subdivisions[1]):
                for k in range(subdivisions[2]):
                    if (samples>0):
                        for s in range(samples):
                            x = min_x + (i + random.random()) * cell_size
                            y = min_y + (j + random.random()) * cell_size
                            z = min_z + (k + random.random()) * cell_size
                            point = (x, y, z)

                            if self.is_point_inside_mesh(point, mesh):
                                points.append(point)
                    else:
                        x = min_x + (i + 0.5) * cell_size
                        y = min_y + (j + 0.5) * cell_size
                        z = min_z + (k + 0.5) * cell_size
                        point = (x, y, z)

                        if self.is_point_inside_mesh(point, mesh):
                            points.append(point)

        return points


    ''' is_point_inside_mesh checks if a point is inside a specific mesh.
        To do this, a ray is fired from (at a random direciton) the point. 
        If the ray interescts with the mesh an uneven number of times, the point is inside the mesh.

        point       : the point position
        mesh        : the mesh to check if the point is in
        
        This only works if the mesh is enclosed, any gaps in the geometry can lead to unstable results.
    '''
    def is_point_inside_mesh(self, point, mesh):
        direction = om.MVector(random.random(), random.random(), random.random())
        
        selection_list = om.MSelectionList()
        selection_list.add(mesh)
        dag_path = selection_list.getDagPath(0)
        
        fn_mesh = om.MFnMesh(dag_path)
        intersections = fn_mesh.allIntersections(
            om.MFloatPoint(point),
            om.MFloatVector(direction),
            om.MSpace.kWorld,
            999999,
            False
        )

        if (intersections is not None):
            num_intersections = len(intersections[0])
            return num_intersections % 2 != 0

        return False



""" ---------- The script below is the main fluid simulation code. In the case of needing this for future projects, this is where you should be looking. ---------- """



''' The MFS_Particle class is used to store the simulation particle data. '''
class MFS_Particle():

    ''' __init__ initializes the particle
        
        id              : The point id, this is used to match the MFS_Particle to the corresponding maya particle object.
        pos             : The position of the particle. Eg: (0.0, 2.0, 0.0)
        vel             : The velocity of the particle. Eg: (-3.0, -2.0, 0.0)
    
    '''
    def __init__(self, id, pos, vel) -> None:
        self.id = id
        self.position = pos
        self.velocity = vel
        self.mass = 0.1


    ''' advect takes the interpolated velocity from the grid and adds it to the particle.

        bbox            : The bounding box of the simulation domain.
        velocity        : The current velocity.
        last_velocity   : The velocity from last_velocity.
        damping         : The damping factor for particle-ground collisions.
        dt              : The timestep.
        flipFac         : The blending from PIC (0) -> (1) FLIP. 

    '''
    def integrate(self, current_velocity, last_velocity, flipFac, bbox, dt):
        # Update velocity based on interpolated grid velocity
        pic = current_velocity

        flip = self.velocity + (last_velocity - current_velocity)
        self.velocity = flipFac * flip + (1 - flipFac) * pic

        self.position += self.velocity * dt
        
        


''' The MFS_Grid does most of the calculations for the simulation. Specifically:

    Particle->Grid and Grid->Particle
    Time-step calculation
    Force calculation
    enforcing boundaries
    possion equation solving

'''
class MFS_Grid():
    def __init__(self, resolution, cell_size):
        self.resolution = resolution
        self.cell_size = cell_size

        self.velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        self.last_velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.last_velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.last_velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        self.density = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")
        self.type = np.full((self.resolution[0], self.resolution[1], self.resolution[2]), 0, dtype="int64")
        
        self.clear()

        self.particleHashTable = {}

    def particles_to_grid(self, particles, bbox):        
        weight_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")

        weight_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")

        weight_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        for p in particles:

            # SOLVE U VELOCITY
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.0, -0.5, -0.5]))
        
            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_u)

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1] - 1)][min(k, self.resolution[2]  - 1)] += p.velocity[0] * w000
            weight_u[min(i, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w000

            self.velocity_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2] - 1)] += p.velocity[0] * w100
            weight_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w100

            self.velocity_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1] - 1)][min(k, self.resolution[2]  - 1)] += p.velocity[0] * w010
            weight_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w010

            self.velocity_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += p.velocity[0] * w110
            weight_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w110

            self.velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += p.velocity[0] * w001
            weight_u[min(i, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w001

            self.velocity_u[min(i, self.resolution[0])][min(j+1, self.resolution[1]  - 1)][min(k+1, self.resolution[2]  - 1)] += p.velocity[0] * w011
            weight_u[min(i, self.resolution[0])][min(j+1, self.resolution[1] - 1)][min(k + 1, self.resolution[2]  - 1)] += w011

            self.velocity_u[min(i+1, self.resolution[0])][min(j, self.resolution[1]  - 1)][min(k+1, self.resolution[2]  - 1)] += p.velocity[0] * w101
            weight_u[min(i+1, self.resolution[0])][min(j, self.resolution[1] - 1)][min(k+1, self.resolution[2]  - 1)] += w101

            self.velocity_u[min(i+1, self.resolution[0] - 1)][min(j+1, self.resolution[1]  - 1)][min(k+1, self.resolution[2]  - 1)] += p.velocity[0] * w111
            weight_u[min(i+1, self.resolution[0])][min(j+1, self.resolution[1]  - 1)][min(k+1, self.resolution[2]  - 1)] += w111

            # SOLVE V VELOCITY
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([-0.5, 0, -0.5]))
        
            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_v)

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.velocity_v[min(i, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k, self.resolution[2] - 1)] += p.velocity[1] * w000
            weight_v[min(i, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k, self.resolution[2] - 1)] += w000

            self.velocity_v[min(i + 1, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k, self.resolution[2] - 1)] += p.velocity[1] * w100
            weight_v[min(i + 1, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k, self.resolution[2] - 1)] += w100

            self.velocity_v[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2] - 1)] += p.velocity[1] * w010
            weight_v[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2] - 1)] += w010

            self.velocity_v[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2] - 1)] += p.velocity[1] * w110
            weight_v[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2] - 1)] += w110

            self.velocity_v[min(i, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k + 1, self.resolution[2] - 1)] += p.velocity[1] * w001
            weight_v[min(i, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k + 1, self.resolution[2] - 1)] += w001

            self.velocity_v[min(i, self.resolution[0] - 1)][min(j+1, self.resolution[1])][min(k+1, self.resolution[2] - 1)] += p.velocity[1] * w011
            weight_v[min(i, self.resolution[0] - 1)][min(j+1, self.resolution[1])][min(k + 1, self.resolution[2] - 1)] += w011

            self.velocity_v[min(i+1, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k+1, self.resolution[2] - 1)] += p.velocity[1] * w101
            weight_v[min(i+1, self.resolution[0] - 1)][min(j, self.resolution[1])][min(k+1, self.resolution[2] - 1)] += w101

            self.velocity_v[min(i+1, self.resolution[0] - 1)][min(j+1, self.resolution[1])][min(k+1, self.resolution[2] - 1)] += p.velocity[1] * w111
            weight_v[min(i+1, self.resolution[0] - 1)][min(j+1, self.resolution[1])][min(k+1, self.resolution[2] - 1)] += w111


            # SOLVE W VELOCITY
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([-0.5, -0.5, 0]))
        
            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_w)

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.velocity_w[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2])] += p.velocity[2] * w000
            weight_w[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2])] += w000

            self.velocity_w[min(i + 1, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2])] += p.velocity[2] * w100
            weight_w[min(i + 1, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2])] += w100

            self.velocity_w[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k, self.resolution[2])] += p.velocity[2] * w010
            weight_w[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k, self.resolution[2])] += w010

            self.velocity_w[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k, self.resolution[2])] += p.velocity[2] * w110
            weight_w[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k, self.resolution[2])] += w110

            self.velocity_w[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k + 1, self.resolution[2])] += p.velocity[2] * w001
            weight_w[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k + 1, self.resolution[2])] += w001

            self.velocity_w[min(i, self.resolution[0] - 1)][min(j+1, self.resolution[1] - 1)][min(k+1, self.resolution[2])] += p.velocity[2] * w011
            weight_w[min(i, self.resolution[0] - 1)][min(j+1, self.resolution[1] - 1)][min(k + 1, self.resolution[2])] += w011

            self.velocity_w[min(i+1, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k+1, self.resolution[2])] += p.velocity[2] * w101
            weight_w[min(i+1, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k+1, self.resolution[2])] += w101

            self.velocity_w[min(i+1, self.resolution[0] - 1)][min(j+1, self.resolution[1] - 1)][min(k+1, self.resolution[2])] += p.velocity[2] * w111
            weight_w[min(i+1, self.resolution[0] - 1)][min(j+1, self.resolution[1] - 1)][min(k+1, self.resolution[2])] += w111

            # SOLVE DENSITY
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.5, 0.5, 0.5]))
            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.density)

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.density[min(i, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w000
            self.density[min(i + 1, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w100
            self.density[min(i, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w010
            self.density[min(i + 1, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w110
            self.density[min(i, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w001
            self.density[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1]  - 1)][min(k + 1, self.resolution[2] - 1)] += w011
            self.density[min(i + 1, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w101
            self.density[min(i + 1, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w111
            
        
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0, 0, 0]))

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.type[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2]-1)] == 1

        for u in range(self.resolution[0]):
            for v in range(self.resolution[1]):
                for w in range(self.resolution[2]):
                    if (weight_u[u][v][w] > 0):
                        self.velocity_u[u][v][w] /= weight_u[u][v][w]

        for u in range(self.resolution[0]):
            for v in range(self.resolution[1]):
                for w in range(self.resolution[2]):
                    if (weight_v[u][v][w] > 0):
                        self.velocity_v[u][v][w] /= weight_v[u][v][w]

        for u in range(self.resolution[0]):
            for v in range(self.resolution[1]):
                for w in range(self.resolution[2]):
                    if (weight_w[u][v][w] > 0):
                        self.velocity_w[u][v][w] /= weight_w[u][v][w]
        
        num_fluid_cells = 0
        average_density = 0

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (self.type[i][j][k] == 1):
                        num_fluid_cells += 1
                        average_density += self.density[i][j][k]

        if (num_fluid_cells > 0): 
            average_density /= num_fluid_cells

        self.last_velocity_u = np.array(self.velocity_u, copy=True)
        self.last_velocity_v = np.array(self.velocity_v, copy=True)
        self.last_velocity_w = np.array(self.velocity_w, copy=True)

        return average_density

    
    def in_bounds(self, i, j, k, li, lj, lk):
        return (
            0 <= i < li and
            0 <= j < lj and 
            0 <= k < lk
        )
                
        
    def get_trilinear_weights(self, x, y, z, i, j, k, array):
        dx = x - i
        dy = y - j
        dz = z - k

        w000 = (1 - dx) * (1 - dy) * (1 - dz) * self.in_bounds(i, j, k, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w100 = (dx) * (1 - dy) * (1 - dz) * self.in_bounds(i + 1, j, k, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w010 = (1 - dx) * (dy) * (1 - dz) * self.in_bounds(i, j + 1, k, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w110 = (dx) * (dy) * (1 - dz) * self.in_bounds(i + 1, j + 1, k, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w001 = (1 - dx) * (1 - dy) * (dz) * self.in_bounds(i, j, k + 1, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w011 = (1 - dx) * (dy) * (dz) * self.in_bounds(i, j + 1, k + 1, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w101 = (dx) * (1 - dy) * (dz) * self.in_bounds(i + 1, j, k + 1, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
        w111 = (dx) * (dy) * (dz) * self.in_bounds(i + 1, j + 1, k + 1, np.size(array, axis=0), np.size(array, axis=1), np.size(array, axis=2))
    
        return w000, w100, w010, w110, w001, w011, w101, w111

    def calc_dt(self, particles, timescale, external_force):
        max_speed = 0

        for particle in particles:
            speed = np.linalg.norm(particle.velocity)
            max_speed = max(max_speed, speed)

        max_dist = np.linalg.norm(self.cell_size) * np.linalg.norm(external_force)

        if (max_speed <= 0):
            max_speed = 1

        return min(timescale, max(timescale * max_dist / max_speed, 1))

    def apply_forces(self, external_force, dt):
        for u in range(self.resolution[0] + 1):
            for v in range(self.resolution[1]):
                for w in range(self.resolution[2]):
                    self.velocity_u[u][v][w] += external_force[0] * dt

        for u in range(self.resolution[0]):
            for v in range(self.resolution[1] + 1):
                for w in range(self.resolution[2]):
                    self.velocity_v[u][v][w] += external_force[1] * dt

        for u in range(self.resolution[0]):
            for v in range(self.resolution[1]):
                for w in range(self.resolution[2] + 1):
                    self.velocity_w[u][v][w] += external_force[2] * dt

    def enforce_boundaries(self):
        for uy in range(self.resolution[1] ):
            for uz in range(self.resolution[2] ):
                if (self.velocity_u[0][uy][uz] < 0): 
                    self.velocity_u[0][uy][uz] = 0
                if (self.velocity_u[self.resolution[0]][uy][uz] > 0):
                    self.velocity_u[self.resolution[0]][uy][uz] = 0

        for vx in range(self.resolution[0]):
            for vz in range(self.resolution[2]):
                if (self.velocity_v[vx][0][vz] < 0): 
                    self.velocity_v[vx][0][vz] = 0
                if (self.velocity_v[vx][self.resolution[1]][vz] > 0):
                    self.velocity_v[vx][self.resolution[1]][vz] = 0
        
        for wx in range(self.resolution[0]):
            for wy in range(self.resolution[0]):
                if (self.velocity_w[wx][wy][0] < 0): 
                    self.velocity_w[wx][wy][0] = 0
                if (self.velocity_w[wx][wy][self.resolution[2]] > 0):
                    self.velocity_w[wx][wy][self.resolution[2]] = 0

    def solve_divergence(self, iterations, overrelaxation, stiffness, rest_density):
        for n in range(iterations):
            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]):
                    for k in range(self.resolution[2]):
                        divergence = overrelaxation * (
                            (self.velocity_u[i+1][j][k] - self.velocity_u[i][j][k]) / self.cell_size[0] +
                            (self.velocity_v[i][j+1][k] - self.velocity_v[i][j][k]) / self.cell_size[1] +
                            (self.velocity_w[i][j][k+1] - self.velocity_w[i][j][k]) / self.cell_size[2]
                        ) - stiffness * (self.density[i][j][k] - rest_density)

                        borders = (
                            self.in_bounds(i-1, j, k, self.resolution[0], self.resolution[1], self.resolution[2]) + 
                            self.in_bounds(i+1, j, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j-1, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j+1, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j, k-1, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j, k+1, self.resolution[0], self.resolution[1], self.resolution[2])
                        )
            
                        self.velocity_u[i][j][k] += divergence * self.in_bounds(i-1, j, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_u[i+1][j][k] -= divergence * self.in_bounds(i+1, j, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders

                        self.velocity_v[i][j][k] += divergence * self.in_bounds(i, j-1, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_v[i][j+1][k] -= divergence * self.in_bounds(i, j+1, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders

                        self.velocity_w[i][j][k] += divergence * self.in_bounds(i, j, k-1, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_w[i][j][k+1] -= divergence * self.in_bounds(i, j, k+1, self.resolution[0], self.resolution[1], self.resolution[2])/borders



    def grid_to_particles(self, particles, bbox, flipFac, dt):
        for p in particles:
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0, -0.5, -0.5]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_u)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            velocity_u = (
                self.velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w000 +
                self.velocity_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w100 +
                self.velocity_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w010 +
                self.velocity_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w110 +
                self.velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w001 +
                self.velocity_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w011 +
                self.velocity_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w101 +
                self.velocity_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w111
            )

            if (total_weight > 0):
                velocity_u /= total_weight


            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([-0.5, 0.0, -0.5]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_v)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            velocity_v = (
                self.velocity_v[min(i, self.resolution[0]-1)][min(j, self.resolution[1])][min(k, self.resolution[2]-1)] * w000 +
                self.velocity_v[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1])][min(k, self.resolution[2]-1)] * w100 +
                self.velocity_v[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2]-1)] * w010 +
                self.velocity_v[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2]-1)] * w110 +
                self.velocity_v[min(i, self.resolution[0]-1)][min(j, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w001 +
                self.velocity_v[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w011 +
                self.velocity_v[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w101 +
                self.velocity_v[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w111
            )

            if (total_weight > 0):
                velocity_v /= total_weight


            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([-0.5, -0.5, 0.0]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_w)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            velocity_w = (
                self.velocity_w[min(i, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k, self.resolution[2])] * w000 +
                self.velocity_w[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k, self.resolution[2])] * w100 +
                self.velocity_w[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k, self.resolution[2])] * w010 +
                self.velocity_w[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k, self.resolution[2])] * w110 +
                self.velocity_w[min(i, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w001 +
                self.velocity_w[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w011 +
                self.velocity_w[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w101 +
                self.velocity_w[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w111
            )

            if (total_weight > 0):
                velocity_w /= total_weight

            current_velocity = np.array([velocity_u, velocity_v, velocity_w])
            last_velocity = np.zeros(3)

            p.integrate(current_velocity, last_velocity, flipFac, bbox, dt)
            self.insert_particle_into_hash_table(p, bbox, np.zeros(3))

    def handle_collisions_and_boundary(self, particles, bbox, pscale):
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        for particle in particles:
            # Update particle position based on velocity
            x, y, z = particle.position
            r = pscale / 2

            # Handle boundary conditions
            if x - r < min_x:
                particle.velocity[0] *= 0
                particle.position[0] = min_x + r
            if x + r > max_x:
                particle.velocity[0] *= 0
                particle.position[0] = max_x - r
            if y - r < min_y:
                particle.velocity[1] *= 0
                particle.position[1] = min_y + r
            if y + r > max_y:
                particle.velocity[1] *= 0
                particle.position[1] = max_y - r
            if z - r < min_z:
                particle.velocity[2] *= 0
                particle.position[2] = min_z + r
            if z + r > max_z:
                particle.velocity[2] *= 0
                particle.position[2] = max_z - r

            x, y, z, i, j, k = self.get_grid_coords(bbox, particle.position, np.zeros(3))

            # Push neighboring particles into the array
            neighboring_particles = self.get_particles_from_hash_table(i, j, k)

            # Handle particle collisions
            for other in neighboring_particles:
                if particle != other:  # Ensure we're not checking the particle against itself
                    dx = other.position[0] - particle.position[0]
                    dy = other.position[1] - particle.position[1]
                    dz = other.position[2] - particle.position[2]
                    dist_squared = dx ** 2 + dy ** 2 + dz ** 2
                    min_dist_squared = (pscale) ** 2

                    if dist_squared < min_dist_squared:
                        # Swap velocities
                        temp_velocity = particle.velocity
                        particle.velocity = other.velocity
                        other.velocity = temp_velocity

                        # Calculate the direction of the collision
                        dist = dist_squared ** 0.5
                        overlap = (pscale) - dist

                        if dist > 0:
                            dx /= dist
                            dy /= dist
                            dz /= dist

                        # Move particles apart proportionally to their overlap
                        moveX = dx * overlap * 0.5
                        moveY = dy * overlap * 0.5
                        moveZ = dz * overlap * 0.5

                        # Move the particles in opposite directions
                        particle.position[0] -= moveX
                        particle.position[1] -= moveY
                        particle.position[2] -= moveZ
                        other.position[0] += moveX
                        other.position[1] += moveY
                        other.position[2] += moveZ

            self.insert_particle_into_hash_table(particle, bbox, np.zeros(3))
    
    def get_velocity(self, velu, velv, velw, i, j, k):
        return np.array([
            velu[i + 1][j][k] - velu[i][j][k],
            velv[i][j+1][k] - velv[i][j][k],
            velw[i][j][k+1] - velw[i][j][k]
        ]) / self.cell_size
    
    def get_grid_coords(self, bbox, position, offset):
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        x = ((position[0] - min_x) / self.cell_size[0]) - offset[0]
        y = ((position[1] - min_y) / self.cell_size[1]) - offset[1]
        z = ((position[2] - min_z) / self.cell_size[2]) - offset[2]

        i = int(x)
        j = int(y)
        k = int(z)

        return x, y, z, i, j, k
    
    # CHAT GPT START
    
    def insert_particle_into_hash_table(self, particle, bbox, offset):
        x, y, z, i ,j, k = self.get_grid_coords(bbox, particle.position, offset)
        hash_val = self.hash_coords(i, j, k)

        if hash_val not in self.particleHashTable:
            self.particleHashTable[hash_val] = []

        self.particleHashTable[hash_val].append(particle)

    def get_particles_from_hash_table(self, i, j, k):
        hash_val = self.hash_coords(i, j, k)
        return self.particleHashTable.get(hash_val, [])
    
    def hash_coords(self, i, j, k):
        h = (i * 92837111) ^ (j * 689287499) ^ (k * 123456789)
        return abs(h) % (self.resolution[0] * self.resolution[1] * self.resolution[2])

    # CHATGPT END

    def clear(self):
        self.particleHashTable = {}
        self.velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")
        self.density = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")


# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()