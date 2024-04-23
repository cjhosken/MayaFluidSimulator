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

''' Maya Fluid Simulator uses two external modules, numpy and scipy. Users will need to manually install these modules using the commands below.

    Windows         : "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy scipy

    Linux           : /usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy scipy

    Make sure that the correct maya version is being used.    
'''

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

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

        cmds.image(width=self.popup_width/2, height=self.popup_height/4, image=os.path.join(self.project_path, "icons/MFS_banner.png"))

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
        self.visc_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, field=True, label="Viscosity Factor")
        self.damp_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.9, maxValue=1, field=True, label="Floor Damping")

        self.picflip_ctrl = cmds.floatSliderGrp(minValue=0, maxValue=1.0, step=0.01, value=0.5, field=True, label="PIC/FLIP Mix")
        
        cmds.rowLayout(numberOfColumns=2)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.01, field=True, label="Time Scale")
        

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
            viscosity_factor = cmds.floatSliderGrp(self.visc_ctrl, query=True, value=True)
            damping =  (1 - cmds.floatSliderGrp(self.damp_ctrl, query=True, value=True))
            pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)
            flipFac = cmds.floatSliderGrp(self.picflip_ctrl, query=True, value=True)

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
            scale = np.array(cmds.xform(source + "_domain", query=True, scale=True))
            sizex = cmds.polyCube(source + "_domain", query=True, width=True)
            sizey = cmds.polyCube(source + "_domain", query=True, height=True)
            sizez = cmds.polyCube(source + "_domain", query=True, depth=True)
            size = np.array([sizex, sizey, sizez]) * scale

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
            self.update(source, particles, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, 0)


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
    def update(self, source, particles, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, progress):
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

            cfl_time = 0

            while (cfl_time < timescale):
                grid.from_particles(bbox, particles)
                timestep = grid.calc_timestep(external_force, timescale)
                grid.calc_forces(external_force, timestep)
                #grid.enforce_boundaries()
                grid.solve_poisson()
                grid.to_particles(bbox, particles, timestep, damping, flipFac)
                cfl_time += timestep
            
            cmds.currentTime(t + 1, edit=True)

            self.update(source, particles, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, progress=progress+1)
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


    ''' advect takes the interpolated velocity from the grid and adds it to the particle.

        bbox            : The bounding box of the simulation domain.
        velocity        : The current velocity.
        last_velocity   : The velocity from last_velocity.
        damping         : The damping factor for particle-ground collisions.
        dt              : The timestep.
        flipFac         : The blending from PIC (0) -> (1) FLIP. 

    '''
    def advect(self, bbox, velocity, last_velocity, damping, dt, flipFac):
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # PIC replaces the velocity of the particle with the velocity interpolated from the grid. 
        # This often results in smoother fluid behavior, which works better for viscous fluids like honey.
        pic_vel = velocity

        # FLIP finds the change in velocity from before forces are applied and divergence is solved to after.
        # This often results in splashier fluid behvaior, which works better for water.

        flip_vel = self.velocity + (last_velocity - velocity)

        # The two techniques are blended together using flipFac.
        self.velocity = (flipFac) * flip_vel + (1-flipFac) * pic_vel
        # For more information on PIC and FLIP, read (https://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf, section 3.2.8)
        
        # Advect the particle to check if it collides with the simulation bounds.
        advected = self.position + self.velocity

        #TODO: Can this be removed once proper divergence is zeroed?
        if (min_x <= advected[0] <= max_x and
            min_y <= advected[1] <= max_y and
            min_z <= advected[2] <= max_z):
            self.position = advected
        else:
            if advected[0] < min_x or advected[0] > max_x:
                self.velocity[0] = -self.velocity[0]

            if advected[1] > max_y:
                self.velocity[1] = -self.velocity[1]
                
            if advected[2] < min_z or advected[2] > max_z:
                self.velocity[2] = -self.velocity[2] 

            #TODO: When reflecting the velocity, the particles seem to repeatdly move up slowly and then snap down. This is likely due to the code not conserving momentum properly.
            if advected[1] < min_y:
                self.velocity[1] = -self.velocity[1] * damping
                
            self.position += self.velocity



''' The MFS_Grid does most of the calculations for the simulation. Specifically:

    Particle->Grid and Grid->Particle
    Time-step calculation
    Force calculation
    enforcing boundaries
    possion equation solving

'''
class MFS_Grid():

    '''  __init__ initializes the grid values and sets up the velocity arrays.

        res         : Resolution of the grid. Eg: (64, 64, 64)
        cell_size   : The size (length) of the cell, Eg: (0.2, 0.2, 0.2)

    '''
    def __init__(self, res, cell_size) -> None:
        self.resolution = res
        self.cell_size = cell_size

        # velocity and last_velocity are both initialized with an extra value on each axis.
        # This is due to the velocity components being stored on the "cell borders", which involves half-indexes.
        # More about half-indexing can be found here:

        self.velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")
        self.last_velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")


    ''' from_particles trilinearly interpolates the particle velocities onto the grid.

        bbox               : The bounding box of the simulation domain.
        particles          : The array containing the MFS_Particles.

        This is also known as P2G (Particle to Grid) in some simulation papers.
    '''
    def from_particles(self, bbox, particles):
        # Reset the velocity components
        self.clear()

        # As we're dealing with many particles, the velocity components need to be averaged.
        # Usually, this is done by doing sum_of_particles / num_of_particles. 
        # In trilinear interpolation, we need to do sum_of_interpolated_velocities / sum_of_weights
        # Therefore, we need to keep track of the total weights in our grid cells

        weights = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1), dtype="float64")

        # Iterate through the particles
        for p in particles:
            # Map the point from worldspace to grid space
            current = self.particle_to_grid(p, bbox)

            # Get the trilinear weights
            w000, w100, w010, w110, w001, w101, w011, w111, i, j, k = self.get_trilinear_weights(current)

            # Update the velocity grid and weight grid using point velocity and weights.
            self.velocity[i][j][k] += p.velocity * w000
            weights[i][j][k] += w000

            self.velocity[min(i + 1, self.resolution[0] - 1)][j][k] += p.velocity * w100
            weights[min(i + 1, self.resolution[0] - 1)][j][k] += w100

            self.velocity[i][min(j + 1, self.resolution[1] - 1)][k] += p.velocity * w010
            weights[i][min(j + 1, self.resolution[1] - 1)][k] += w010

            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] += p.velocity * w110
            weights[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] += w110

            self.velocity[i][j][min(k + 1, self.resolution[2] - 1)] += p.velocity * w001
            weights[i][j][min(k + 1, self.resolution[2] - 1)] += w001

            self.velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] += p.velocity * w101
            weights[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] += w101

            self.velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += p.velocity * w011
            weights[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += w011

            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += p.velocity * w111
            weights[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += w111

        # Average the velocities using the calculated weights.
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (weights[i][j][k] != 0):
                        self.velocity[i][j][k] /= weights[i][j][k]

        # Store the velocity into a last_velocity grid so that we can calculate the change in velocity used in FLIP.
        self.last_velocity = np.array(self.velocity, copy=True)
    

    ''' calc_timestep is used to create a CFL timestep. 
    
        It looks at the maximum velocity in the grid and adjusts the timestep so that the velocity will only move one grid cell.
        The current implementation is by Bridson (2009) as discussed in: (https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf, Section 3.4.1)

        external_force      : The external force, used to predict a more stable timestep.
        timescale           : The speed of the simulation. The timestep will never be greater than timescale.

    '''
    def calc_timestep(self, external_force, timescale):

        #TODO: Is the timestep actually used to move the velocity one grid cell?

        max_vel = 0

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    max_vel = max(max_vel, np.linalg.norm(self.velocity[i][j][k]))

        max_vel += np.linalg.norm(external_force * self.cell_size)

        timestep = timescale

        if (max_vel > 0):
            timestep = max(timestep, timescale * max(np.linalg.norm(self.cell_size) / max_vel, 1))

        return timestep


    ''' calc forces calculates the total forces in the simulation.

        external_force      : Usually gravity.
        dt                  : timestep.
    
    '''
    def calc_forces(self, external_force, dt):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    total_force = external_force
                    self.velocity[i][j][k] += total_force * dt


    ''' enforce_boundaries checks to see if a border cell velocity is directing into the simulation bounds.
        If it is, the velocity component is set to 0.

        This is mentioned in (https://www.danenglesson.com/images/portfolio/FLIP/rapport.pdf, Section 3.2.5), However im not entirely sure that it is implemented correctly.
    '''
    def enforce_boundaries(self):

        #TODO: DOUBLE CHECK IF THE VELOCITY SET 0 IS THE CORRECT IMPLMENETATION.

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (i-1 < 0 or i+1 >= self.resolution[0]):
                        self.velocity[i][j][k][0] = 0

                    if (j-1 < 0 or j+1 >= self.resolution[1]):
                        self.velocity[i][j][k][1] = 0

                    if (k-1 < 0 or k+1 >= self.resolution[2]):
                        self.velocity[i][j][k][2] = 0
    
    
    '''solve_poisson is used to make the velocity grid divergence-free.

        THIS FUNCTION IS STILL UNDER IMPLEMENTATION
    
    '''
    def solve_poisson(self):
        divergence = self.compute_divergence()

        A = self.construct_coefficient_matrix()
        print(A)
        print(np.linalg.cond)
        b = divergence.flatten()
        print(b)
        pressure = spsolve(A, b)
        pressure = pressure.reshape(self.resolution)
        self.correct_velocity(pressure)

    
    '''compute_divergence finds the divergence of the velocity field.

        THIS FUNCTION IS STILL UNDER IMPLEMENTATION
    
    '''
    def compute_divergence(self):
        divergence = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")
        # Find the velocity divergence
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    divergence[i][j][k] = (self.velocity[i + 1][j][k][0] - self.velocity[i][j][k][0] +
                                           self.velocity[i][j + 1][k][1] - self.velocity[i][j][k][1] +
                                           self.velocity[i][j][k + 1][2] - self.velocity[i][j][k][2])
                    
        return divergence
    
    
    '''construct_coefficient_matrix creates a matrix that can be used for Preconditioned Conjugate Gradient Method. Not entirely sure what that means but we're working on it.

        THIS FUNCTION IS STILL UNDER IMPLEMENTATION
    
    '''
    def construct_coefficient_matrix(self):
        # Construct the coefficient matrix (sparse)
        # Assuming uniform grid spacing for simplicity
        num_cells = np.prod(self.resolution)

        diag_main = np.ones(num_cells) * -6

        diag_offsets = [-1, 1, -self.resolution[0], self.resolution[0], -self.resolution[0] * self.resolution[1],
                        self.resolution[0] * self.resolution[1]]
        data = np.ones((7, num_cells))

        # Include diag_main in the data array
        data[0] = diag_main  # Include self term for diagonal

        for i, offset in enumerate(diag_offsets):
            mask = np.ones(num_cells, dtype=bool)
            if offset < 0:
                mask[:abs(offset)] = False
            elif offset > 0:
                mask[-offset:] = False
            data[i + 1, ~mask] = 0

        # Create sparse matrix
        A = csr_matrix((data.ravel(), (np.arange(num_cells).repeat(7), np.tile(np.arange(num_cells), 7))))

        return A
    

    '''correct_velocity takes the pressure gradient and uses it to make the velocity divergence-free.

        pressure        : The pressure gradient.

        THIS FUNCTION IS STILL UNDER IMPLEMENTATION
    
    '''
    def correct_velocity(self, pressure):
        dx, dy, dz = self.cell_size

        for i in range(1, self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.velocity[i][j][k][0] -= (pressure[i][j][k] - pressure[i - 1][j][k]) / dx

        for i in range(self.resolution[0]):
            for j in range(1, self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.velocity[i][j][k][1] -= (pressure[i][j][k] - pressure[i][j - 1][k]) / dy

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(1, self.resolution[2]):
                    self.velocity[i][j][k][2] -= (pressure[i][j][k] - pressure[i][j][k - 1]) / dz


    '''to_particles does the reverse of from_particles, and trilinearly interpolates the grid velocities back into the particles.
        
        bbox            : The bounding box of the simulation domain.
        particles       : The array containing the MFS_Particles.
        dt              : The timestep.
        damping         : The damping factor for particle-ground collisions.
        flipFac         : The blending from PIC (0) -> (1) FLIP. 

        This is also known as G2P (Grid to Particle) in some simulation papers.

    '''
    def to_particles(self, bbox, particles, dt, damping, flipFac):
        # iterate through the particles
        for p in particles:
            # Map the point from worldspace to grid space
            current = self.particle_to_grid(p, bbox)

            # backward trace the particle and obtain its location in grid space
            last = current - p.velocity * dt / self.cell_size

            # find the velocities at both current and last
            velocity = self.trilinear_interpolate_velocity(self.velocity, current)
            last_velocity = self.trilinear_interpolate_velocity(self.last_velocity, last)

            # advect the particle
            p.advect(bbox, velocity, last_velocity, damping, dt, flipFac)


    ''' trilinear_interpolate_velocity obtains the trilinearly interpolated velocity value from the grid at (x, y, z)

        vel         : The velocity grid to access velocity from.
        pos     : The coordinates of the particle in grid space.
        
    '''
    def trilinear_interpolate_velocity(self, vel, pos):
        # Get the trilinear weights
        w000, w100, w010, w110, w001, w101, w011, w111, i, j, k = self.get_trilinear_weights(pos)

        # Get the velocity
        velocity = (
            vel[i][j][k] * w000 + 
            vel[min(i + 1, self.resolution[0] - 1)][j][k] * w100 +
            vel[i][min(j + 1, self.resolution[1] - 1)][k] * w010 +
            vel[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] * w110 +
            vel[i][j][min(k + 1, self.resolution[2] - 1)] * w001 +
            vel[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] * w101 +
            vel[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w011 +
            vel[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w111
        )

        # Compute the total weight
        weight = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111

        # Return the averaged velocity
        return velocity / weight
    

    '''get_trillinear_weights returns the weights for the 8 grid cells.

        pos     : The coordinates of the particle in grid space.
    
    '''
    def get_trilinear_weights(self, pos):
        # Snap the grid space location to a cell index
        ijk = np.array([int(pos[0]), int(pos[1]), int(pos[2])], dtype="float64")

        # The difference between grid space and snapped grid space is used for trilinear interpolation
        dc = pos - ijk

        # Clamp indices to stay within the grid boundaries.
        # We do this after to avoid screwing with the weighting.
        # TODO: THERE IS A POTENTIAL ERROR WITH NOT CLAMPING THE POSITIONS FROM THE START. CHECK IF THIS CAN BE IGNORED OR NOT
        i = max(0, min(ijk[0], self.resolution[0] - 1))
        j = max(0, min(ijk[1], self.resolution[1] - 1))
        k = max(0, min(ijk[2], self.resolution[2] - 1))

        # Calculate the weights for the surround 8 cells
        w000 = (1 - dc[0]) * (1 - dc[1]) * (1 - dc[2])
        w100 = dc[0] * (1 - dc[1]) * (1 - dc[2])
        w010 = (1 - dc[0]) * dc[1] * (1 - dc[2])
        w110 = dc[0] * dc[1] * (1 - dc[2])
        w001 = (1 - dc[0]) * (1 - dc[1]) * dc[2]
        w101 = dc[0] * (1 - dc[1]) * dc[2]
        w011 = (1 - dc[0]) * dc[1] * dc[2]
        w111 = dc[0] * dc[1] * dc[2]

        # return the weights and the snapped position in grid space.
        return w000, w100, w010, w110, w001, w101, w011, w111, i, j, k
    
    '''particle_to_grid converts the particle position from world space to grid space.

        particle        : The MFS_Particle particle
        bbox            : The bounding box the fluid simulation domain

    '''
    def particle_to_grid(self, particle, bbox):
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # the particle positions are first transformed by moving the world so that the minimum bound of the domain is at (0,0,0). 
        # This then makes obtaining the index by dividing by cell_size much easier.

        u = (particle.position[0] - min_x) / self.cell_size[0]
        v = (particle.position[1] - min_y) / self.cell_size[1]
        w = (particle.position[2] - min_z) / self.cell_size[2]

        return np.array([u, v, w], dtype="float64")
    

    '''clear resets the velocity grid.'''
    def clear(self):
        self.velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")

# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()