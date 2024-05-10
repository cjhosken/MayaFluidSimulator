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
import time

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

class MFS_Plugin():
    ''' The MFS Plugin is a general class that contains the (mainly) Maya side of the script. This allows the use of "global" variables in a contained setting. '''
    
    # "global" variables
    popup_size = (500, 450)
    button_ratio = 0.9

    def __init__(self):
        ''' __init__ initializes the plugin by creating the header menu. '''

        self.MFS_create_menu()


    def MFS_create_menu(self):
        ''' MFS_create_menu deletes any pre-existing Maya Fluid Simulator header menus, then creates a new header menu.'''

        self.MFS_delete_menu()
        cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)
        cmds.menuItem(label="Open Maya Fluid Simulator", command=lambda x:self.MFS_popup())
    

    def MFS_popup(self):
        ''' MFS_popip creates the main UI for Maya Fluid Simulator. 

        Particle Scale         : The visual size of the particles
        Cell Size              : The cell width for particle sourcing and simulation
        Random Sampling        : The sampling value used for generating particles
        Domain Size            : The size of the domain
        Keep Domain            : Whether or not to replace the sourced domain
        Initialize (X)         : Initialize the particles and create the fluid domain. X will remove the particles and domain.

        Force                  : External forces acting on the fluid. Usually gravity.
        Initial Velocity       : The initial velocity of the fluid particles.
        Stiffness              : The amount of pressure force
        Overrelaxation         : The amount of velocity divergence
        Iterations             : The number of iterations for the divergence solve
        PIC/FLIP Mix           : The blending from PIC (0) -> (1) FLIP. PIC is often better for viscious fluids, while FLIP is good for splashy fluids.
        Frame Range            : The range in the which simulation should run between.
        Time Scale             : The speed of the simulation.
        Simulate (X)           : Simulate the fluid particles. X will remove the keyframed simulation. 

        Setting the Maya project dir to the script location shows the plugin icons.
        '''
        
        cmds.window(title="Maya Fluid Simulator", widthHeight=self.popup_size)
        col = cmds.columnLayout(adjustableColumn=True)

        initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)
        cmds.columnLayout(adjustableColumn=True, parent=initialize_section)
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, field=True, label="Particle Scale")
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

        self.stiff_ctrl = cmds.floatSliderGrp(minValue=0, step=0.0001, value=1.0, field=True, label="Pressure")
        self.relax_ctrl = cmds.floatSliderGrp(minValue=0, step=0.0001, value=0.02, field=True, label="Overrelaxation")
        self.iter_ctrl = cmds.intSliderGrp(minValue=0, value=5, field=True, label="Iterations")

        self.picflip_ctrl = cmds.floatSliderGrp(minValue=0, maxValue=1.0, step=0.01, value=0.0, field=True, label="PIC/FLIP Mix")
        
        cmds.rowLayout(numberOfColumns=2)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.1, field=True, label="Time Scale")
        

        solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)

        cmds.button(label="Simulate", command=lambda x:self.MFS_simulate())
        cmds.button(label="X", command=lambda x:self.MFS_reset())
        cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_size[0]), (2, (1-self.button_ratio) * self.popup_size[0])])

        cmds.columnLayout(adjustableColumn=True, parent=col)

        cmds.showWindow()


    
    def MFS_delete_menu(self):
        ''' MFS_delete_menu checks if the Maya Fluid Simulator header menu exists and deletes it. '''

        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)


    def MFS_initialize(self):
        ''' MFS_initialize uses the initialization settings to fill a selected object with fluid particles and to create a domain object. '''
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


    def MFS_simulate(self):
        ''' simulate is the start of an actual Fluid Simulator. It simulates the particle position frame by frame, keyframing the positions each time. 
            The user is then able to cancel the simulation by pressing Esc, however they need to wait for the current frame to finish simulation.
        
            source                  : The fluid source object
            particles               : The array containing the MFS_Particles
            grid                    : The MFS_Grid which most of the calculations are done on.
            frame_range             : The range in the which simulation should run between.
            timescale               : The speed of the simulation.
            external_force          : The external forces acting on the fluid. Usually gravity.
            pscale                  : The size of the particles.
            flipFac                 : The blending from PIC (0) -> (1) FLIP. 
            iterations              : The number of iterations for the divergence solve.
            overrelaxation          : The amount of velocity divergence.
            stiffness               : The amount of pressure force.
            progress                : Used to track the progress bar.

            The reason why so many values are being parsed through the update function is to avoid settings changing midway though simulation.
        '''

        source = self.get_active_object()

        if (source is not None and self.can_simulate(source)):
            self.MFS_reset() # Remove all pre-existing keyframes

            frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
            timescale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
            external_force = cmds.floatFieldGrp(self.force_ctrl, query=True, value=True)

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
            
            progress = 0
            cmds.currentTime(frame_range[0], edit=True)
            t = frame_range[0]
            solved = False
            cancelled = False

            timer_start = time.time()
            # This is the start of the simulator. The method is a python implementation of:
            #
            # 1. keyframe frame: copy the particle positions in the simulation onto the maya particle objects.
            # 2. enter the CFL domain. This is done to limit particles travelling only 1 cell at a time.
            # 3. particles_to_grid: transfer the point velocities to the grid using trilinear interpolation.
            # 4. calc_dt: find the timestep. This will differ from cfl iteration to iteration, but at maximum is the timescale.
            # 5. apply_forces: calculate external forces such as gravity.
            # 6. enforce_boundaries: stop any edge cell velocities from pointing out of the domain. This is done by setting the velocity component to 0.
            # 7. solve_divergence: solve the possion equation that makes the fluid divergence free.
            # 8. grid_to_particles: transfer the grid velocities back into the particles, then move the particles.
            # 9. handle_collisions_and_boundary: Check for particle collisions and move any escaping particles back into the simulation domain.
            #
            # Once the cfl iterations are complete, the next frame is done until all the frames within the frame range are simulated.

            # Initialize an average pressure variable to be used throughout the whole simulation
            
            average_pressure = -1

            while (not (solved or cancelled)):
                percent = (progress / (frame_range[1] - frame_range[0])) * 100

                cmds.progressWindow(e=1, progress=progress, status=f'Progress: {percent:.1f}%')
                solved = (t < frame_range[0] or t > frame_range[1])
                cancelled = cmds.progressWindow(query=True, isCancelled=True)

                self.keyframe(source, particles, t)
                
                print(f"Maya Fluid Simulator | Simulating Frame: {t}")

                # Initialize the CFL counter to maintain single cell stepping
                cfl = 0

                while(cfl < timescale):
                    grid.clear()

                    grid.particles_to_grid(particles, bbox)

                    if (average_pressure < 0):
                        average_pressure = grid.average_pressure()

                    dt = grid.calc_dt(particles, timescale, external_force)
                    grid.apply_forces(external_force, dt)
                    grid.enforce_boundaries()
                    grid.solve_divergence(iterations, overrelaxation, stiffness, average_pressure, dt)
                    grid.grid_to_particles(particles, bbox, flipFac, dt)
                    grid.handle_collisions_and_boundary(particles, bbox, pscale)
                    cfl += dt            

                t += 1
                cmds.currentTime(t, edit=True)
                progress += 1

            else:
                cmds.currentTime(frame_range[0], edit=True)
                cmds.progressWindow(endProgress=1)

            timer_end = time.time()

            print(f"Maya Fluid Simulator | Simulation Complete! {timer_end-timer_start} seconds taken.")


    def keyframe(self, source, particles, t):
        ''' keyframe take the simulation particles, copies their position into the corresponding Maya particles, then keyframes the positions.

        source          : The source object
        particles       : The array containing MFS_Particles
        t               : the current frame value

        '''
        for p in particles:
            particle_name = f"{source}_particle_{p.id:09}"
            cmds.setKeyframe(particle_name, attribute='translateX', t=t, v=p.position[0])
            cmds.setKeyframe(particle_name, attribute='translateY', t=t, v=p.position[1])
            cmds.setKeyframe(particle_name, attribute='translateZ', t=t, v=p.position[2])

    
    def MFS_delete(self):
        ''' MFS_delete deletes all the particles and the domain, and revert back the source object.'''
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
    


    def MFS_reset(self):
        ''' MFS_reset resets the simulation by deleting all the keyframes and setting the current time to the start of the frame range.'''
        source = self.get_active_object()
        frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
        cmds.currentTime(frame_range[0], edit=True)

        if (source is not None):
            particles = cmds.listRelatives(source + "_particles", children=True) or []

            for p in particles:
                cmds.cutKey(p, attribute='translateX', clear=True)
                cmds.cutKey(p, attribute='translateY', clear=True)
                cmds.cutKey(p, attribute='translateZ', clear=True)



    def get_active_object(self):
        ''' get_active_object checks if the selected object can be a "source" object.
        An object can be a source object if it is not a domain or a particle.
        Otherwise None is returned.
        '''
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


    
    def can_simulate(self, source):
        ''' can_simulate checks if the selected object has been sourced. To do so, it looks to see if a domain and particle group exist for the object.
        
        source          : The selected object to test
    
        '''
        can_simulate = cmds.objExists(source + "_domain") and cmds.objExists(source + "_particles")

        if (not can_simulate):
            cmds.confirmDialog(title="Simulation Error!", 
                message="You need to initialize a source object!",
                button="Oopsies"
            )

        return can_simulate
    

    def mesh_to_points(self, mesh, cell_size, samples=0):
        ''' mesh_to_points generates points inside of a given object.
        It creates a domain around the source object and splits it into subdivisions, then It creates a domain around the source object and splits it into subdivisions. 

        mesh                : the mesh to convert to points
        cell_size           : the spacing between particles. This is a scalar as we want the point generation to be uniform.
        samples             : the number of particles to randomly generate inside a cell. When 0, the particles assume a grid formation.
    
        '''
    
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


    
    def is_point_inside_mesh(self, point, mesh):
        ''' is_point_inside_mesh checks if a point is inside a specific mesh.
        To do this, a ray is fired from (at a random direciton) the point. 
        If the ray interescts with the mesh an uneven number of times, the point is inside the mesh.

        point       : the point position
        mesh        : the mesh to check if the point is in
        
        This only works if the mesh is enclosed, any gaps in the geometry can lead to unstable results.
        '''
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



""" ---------- The script below is the main fluid simulation code. In the case of needing this for future projects, this is where you should be looking. ---------- 

    Alot of this code was inspired by Ten Minute Physics' video on making a FLIP simulator in a browser. You can watch his video here.
    URL: https://www.youtube.com/watch?v=XmzBREkK8kY
    Date Accessed: 05/05/2024
    
"""




class MFS_Particle():
    ''' The MFS_Particle class is used to store the simulation particle data. '''

    def __init__(self, id, pos, vel) -> None:
        ''' __init__ initializes the particle
        
        id              : The point id, this is used to match the MFS_Particle to the corresponding maya particle object.
        pos             : The position of the particle. Eg: (0.0, 2.0, 0.0)
        vel             : The velocity of the particle. Eg: (-3.0, -2.0, 0.0)
    
        '''

        self.id = id
        self.position = pos
        self.velocity = vel
        


    def integrate(self, current_velocity, last_velocity, flipFac, dt):
        ''' integrate takes the interpolated velocity from the grid and adds it to the particle.

        current_velocity: The current velocity.
        last_velocity   : The velocity from last_velocity.
        flipFac         : The blending from PIC (0) -> (1) FLIP. 
        dt              : The timestep.

        '''

        pic = current_velocity

        flip = self.velocity + (current_velocity - last_velocity)
        self.velocity = flipFac * flip + (1 - flipFac) * pic

        self.position += self.velocity * dt
        
        


class MFS_Grid():
    ''' The MFS_Grid does most of the calculations for the simulation. Specifically:

    Particle->Grid and Grid->Particle
    Time-step calculation
    Force calculation
    enforcing boundaries
    possion equation solving

    '''

    def __init__(self, resolution, cell_size):
        '''__init__ sets up the general arrays and grids used for the fluid simulation

        resolution            : The number of cells in each dimension. Eg: (16, 16, 16)
        cell_size             : The side lengths of a single cell.

        '''
        self.resolution = resolution
        self.cell_size = cell_size

        self.velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        self.last_velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.last_velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.last_velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        self.pressure = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")
        self.type = np.full((self.resolution[0], self.resolution[1], self.resolution[2]), 0, dtype="int64")
        
        self.clear()

        self.particleHashTable = {}

    def particles_to_grid(self, particles, bbox):
        '''particles_to_grid trilinearly interpolates the particle velocities onto the velocity grids.

        particles       : An array containing fluid particles
        bbox            : The bounding box domain

        '''        
        weight_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")

        weight_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")

        weight_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")

        for p in particles:

            # Trilinear interpolate particle u velocity to the grid (x velocity)
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.0, 0.5, 0.5]))
        
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

            # Trilinear interpolate particle v velocity to the grid (y velocity)
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.5, 0, 0.5]))
        
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


            # Trilinear interpolate particle w velocity to the grid (z velocity)
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.5, 0.5, 0]))
        
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

            # calculate particle pressure in each cell
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0.5, 0.5, 0.5]))
            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.pressure)

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.pressure[min(i, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w000
            self.pressure[min(i + 1, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w100
            self.pressure[min(i, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w010
            self.pressure[min(i + 1, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k, self.resolution[2]  - 1)] += w110
            self.pressure[min(i, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w001
            self.pressure[min(i, self.resolution[0] - 1)][min(j + 1, self.resolution[1]  - 1)][min(k + 1, self.resolution[2] - 1)] += w011
            self.pressure[min(i + 1, self.resolution[0]  - 1)][min(j, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w101
            self.pressure[min(i + 1, self.resolution[0]  - 1)][min(j + 1, self.resolution[1]  - 1)][min(k + 1, self.resolution[2]  - 1)] += w111
            
            # Set the cell containing the particle to "fluid"
            x, y, z, i, j, k = self.get_grid_coords(bbox, p.position, np.array([0, 0, 0]))

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            self.type[min(i, self.resolution[0] - 1)][min(j, self.resolution[1] - 1)][min(k, self.resolution[2]-1)] = 1

        # Average all the weights

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

        self.last_velocity_u = np.array(self.velocity_u, copy=True)
        self.last_velocity_v = np.array(self.velocity_v, copy=True)
        self.last_velocity_w = np.array(self.velocity_w, copy=True)

    def average_pressure(self):
        # Find the average pressure for the start of simulation
        num_fluid_cells = 0
        average_pressure = 0

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (self.type[i][j][k] == 1):
                        num_fluid_cells += 1
                        average_pressure += self.pressure[i][j][k]

        # Handle dividing by 0
        if (num_fluid_cells > 0): 
            average_pressure /= num_fluid_cells
        
        return average_pressure
    
    
    def in_bounds(self, i, j, k, li, lj, lk):
        '''in_bounds checks if the i, j, k index is within the specified bounds
        
        i, j, k         : The cell space coordinate
        li, lj, lk      : The maximum bounds for the grid.

        '''
        return (
            0 <= i < li and
            0 <= j < lj and 
            0 <= k < lk
        )
                
        
    def get_trilinear_weights(self, x, y, z, i, j, k, array):
        '''get_trilinear_weights uses the particle position (in grid/cell space), to calculated the weights for the surrounding cells.

        x, y, z     : The particle position in grid coords.
        i, j, k     : The particle position in cell coords.
        array       : The 3-Dimensional array to check the weight indexes.


        '''
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
        '''calc_dt finds a timestep suitable enough so that particles will only move 1 cell space at a time.

        particles           : The array of fluid particles
        timescale           : The speed of the simulation
        external_force      : External force acting on the simulation. eg: (0, -9.8, 0)

        '''
        max_speed = 0

        for particle in particles:
            speed = np.linalg.norm(particle.velocity)
            max_speed = max(max_speed, speed)

        max_dist = np.linalg.norm(self.cell_size) * np.linalg.norm(external_force)

        # Handle diving by 0 or negative numbers
        if (max_speed <= 0):
            max_speed = 1

        return min(timescale, max(timescale * max_dist / max_speed, 1))

    def apply_forces(self, external_force, dt):
        '''apply_forces adds the external forces to the velocity grids

        external-force      : External force acting on the simulation. eg: (0, -9.8, 0)
        dt                  : The simulation cfl timestep

        '''
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
        '''enforce_boundaries makes sure that border cells dont have velocities that point out of the simulation domain.
        '''
        # Check the bordering cells and if their velocity points outwards, set it to 0.
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

    def solve_divergence(self, iterations, overrelaxation, stiffness, average_pressure, dt, tolerance=0.0001):
        ''' solve_divergence makes the fluid incompressible.s

        iterations          : The number of iterations for the divergence solve
        overrelaxation      : Scalar value for the velocity difference
        stiffness           : Scalar value for the pressure difference force
        average_pressure    : The average pressure of the fluid
        tolerance           : tolerance value for finding convergence
        
        '''

        # Iterate over the number of interations specified.
        for n in range(iterations):          
            divergence = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")

            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]):
                    for k in range(self.resolution[2]):
                        # Dividing by 10 due to the scene scale. Ideally, this should be removed in future implementations.
                        divergence[i][j][k] = overrelaxation/10 * (
                            (self.velocity_u[i+1][j][k] - self.velocity_u[i][j][k]) / (self.cell_size[0]) +
                            (self.velocity_v[i][j+1][k] - self.velocity_v[i][j][k]) / (self.cell_size[1]) +
                            (self.velocity_w[i][j][k+1] - self.velocity_w[i][j][k]) / (self.cell_size[2])
                        ) - stiffness/10 * (self.pressure[i][j][k] - average_pressure)


            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]):
                    for k in range(self.resolution[2]):
                        # Find the number of bordering cells
                        borders = (
                            self.in_bounds(i-1, j, k, self.resolution[0], self.resolution[1], self.resolution[2]) + 
                            self.in_bounds(i+1, j, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j-1, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j+1, k, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j, k-1, self.resolution[0], self.resolution[1], self.resolution[2]) +
                            self.in_bounds(i, j, k+1, self.resolution[0], self.resolution[1], self.resolution[2])
                        )

                        # Average out the velocities so that the fluid is divergence free.
                        self.velocity_u[i][j][k] += divergence[i][j][k] * self.in_bounds(i-1, j, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_u[i+1][j][k] -= divergence[i][j][k] * self.in_bounds(i+1, j, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders

                        self.velocity_v[i][j][k] += divergence[i][j][k] * self.in_bounds(i, j-1, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_v[i][j+1][k] -= divergence[i][j][k] * self.in_bounds(i, j+1, k, self.resolution[0], self.resolution[1], self.resolution[2])/borders

                        self.velocity_w[i][j][k] += divergence[i][j][k] * self.in_bounds(i, j, k-1, self.resolution[0], self.resolution[1], self.resolution[2])/borders
                        self.velocity_w[i][j][k+1] -= divergence[i][j][k] * self.in_bounds(i, j, k+1, self.resolution[0], self.resolution[1], self.resolution[2])/borders
        


    def grid_to_particles(self, particles, bbox, flipFac, dt):
        '''grid_to_particles trilinearly interpolates the grid velocities onto the particle velocities.

        particles           : An array containing fluid particles
        bbox                : The bounding box domain
        flipFac             : The blending from PIC (0) -> (1) FLIP. 
        dt                  : The simulation cfl timestep

        '''      

        for p in particles:
            # Find the advected position
            position = p.position - p.velocity * dt

            # Tri-linearly interpolate the grid u velocity (x velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0, 0.5, 0.5]))

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

            # Tri-linearly interpolate the pre-calculated grid u velocity (x velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0, 0.5, 0.5]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_u)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)


            last_velocity_u = (
                self.last_velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w000 +
                self.last_velocity_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w100 +
                self.last_velocity_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w010 +
                self.last_velocity_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k, self.resolution[2] -1)] * w110 +
                self.last_velocity_u[min(i, self.resolution[0])][min(j, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w001 +
                self.last_velocity_u[min(i, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w011 +
                self.last_velocity_u[min(i + 1, self.resolution[0])][min(j, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w101 +
                self.last_velocity_u[min(i + 1, self.resolution[0])][min(j + 1, self.resolution[1] -1)][min(k+1, self.resolution[2] -1)] * w111
            )

            if (total_weight > 0):
                last_velocity_u /= total_weight

            # Tri-linearly interpolate the grid v velocity (y velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0.5, 0.0, 0.5]))

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


            # Tri-linearly interpolate the pre-calculated grid v velocity (y velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0.5, 0.0, 0.5]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_v)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            last_velocity_v = (
                self.last_velocity_v[min(i, self.resolution[0]-1)][min(j, self.resolution[1])][min(k, self.resolution[2]-1)] * w000 +
                self.last_velocity_v[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1])][min(k, self.resolution[2]-1)] * w100 +
                self.last_velocity_v[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2]-1)] * w010 +
                self.last_velocity_v[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k, self.resolution[2]-1)] * w110 +
                self.last_velocity_v[min(i, self.resolution[0]-1)][min(j, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w001 +
                self.last_velocity_v[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w011 +
                self.last_velocity_v[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w101 +
                self.last_velocity_v[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1])][min(k+1, self.resolution[2]-1)] * w111
            )

            if (total_weight > 0):
                last_velocity_v /= total_weight


            # Tri-linearly interpolate the grid w velocity (z velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0.5, 0.5, 0.0]))

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

            # Tri-linearly interpolate the pre-calculated grid w velocity (z velocity) to the particle

            x, y, z, i, j, k = self.get_grid_coords(bbox, position, np.array([0.5, 0.5, 0.0]))

            w000, w100, w010, w110, w001, w011, w101, w111 = self.get_trilinear_weights(x, y, z, i, j, k, self.velocity_w)

            total_weight = w000 + w100 + w010 + w110 + w001 + w011 + w101 + w111

            i = max(i, 0)
            j = max(j, 0)
            k = max(k, 0)

            last_velocity_w = (
                self.last_velocity_w[min(i, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k, self.resolution[2])] * w000 +
                self.last_velocity_w[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k, self.resolution[2])] * w100 +
                self.last_velocity_w[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k, self.resolution[2])] * w010 +
                self.last_velocity_w[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k, self.resolution[2])] * w110 +
                self.last_velocity_w[min(i, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w001 +
                self.last_velocity_w[min(i, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w011 +
                self.last_velocity_w[min(i + 1, self.resolution[0]-1)][min(j, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w101 +
                self.last_velocity_w[min(i + 1, self.resolution[0]-1)][min(j + 1, self.resolution[1]-1)][min(k+1, self.resolution[2])] * w111
            )

            if (total_weight > 0):
                last_velocity_w /= total_weight

            # get the 'current' and 'last' velocities, then intergrate the particle and put it into the hash table for collisions

            current_velocity = np.array([velocity_u, velocity_v, velocity_w])
            last_velocity = np.array([last_velocity_u, last_velocity_v, last_velocity_w])

            p.integrate(current_velocity, last_velocity, flipFac, dt)
            self.insert_particle_into_hash_table(p, bbox, np.zeros(3))


    def handle_collisions_and_boundary(self, particles, bbox, pscale):
        '''handle_collsisions_and_boundary keeps the particles within the simulation domain and handles all particle to particle collisions.

        particles           : An array containing fluid particles
        bbox                : The bounding box domain
        pscale              : The size of the particless

        '''
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

                    if dist_squared <= min_dist_squared:
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
        '''get_velocity obtains the 3D velocity at a certain cell (i, j, k)

        velu            : The u velocity grid
        velv            : The v velocity grid
        velw            : The w velocity grid
        i, j, k         : The cell coordinate
        
        '''
        return np.array([
            velu[i + 1][j][k] - velu[i][j][k],
            velv[i][j+1][k] - velv[i][j][k],
            velw[i][j][k+1] - velw[i][j][k]
        ]) / 2
    
    def get_grid_coords(self, bbox, position, offset):
        '''get_grid_coords gets the grid/cell space coordinates of a particle position.

        bbox                : The bounding box domain
        position            : The position of the particle
        offset              : The grid/cell space offset value

        '''
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        x = ((position[0] - min_x) / self.cell_size[0]) - offset[0]
        y = ((position[1] - min_y) / self.cell_size[1]) - offset[1]
        z = ((position[2] - min_z) / self.cell_size[2]) - offset[2]

        i = int(x)
        j = int(y)
        k = int(z)

        return x, y, z, i, j, k
    

    # Provided by OpenAI's ChatGPT
    # Original source: OpenAI ChatGPT model
    # URL: https://openai.com/chatgpt
    # Date Accessed: 05/05/2024
       
    # Code from ChatGPT starts here

    # Function to insert a particle into the hash table based on its bounding box and offset
    def insert_particle_into_hash_table(self, particle, bbox, offset):
        ''' Inserts a particle into the hash table based on its bounding box and offset.

        particle: The particle to insert into the hash table.
        bbox: The bounding box of the particle.
        offset: The offset to determine the grid coordinates.

        '''
        # Get grid coordinates of the particle based on its bounding box and offset
        x, y, z, i ,j, k = self.get_grid_coords(bbox, particle.position, offset)
        # Hash the grid coordinates to get the hash value
        hash_val = self.hash_coords(i, j, k)

        # If hash value not in the hash table, create a new entry
        if hash_val not in self.particleHashTable:
            self.particleHashTable[hash_val] = []

        # Append the particle to the appropriate entry in the hash table
        self.particleHashTable[hash_val].append(particle)

    # Function to retrieve particles from the hash table based on grid coordinates
    def get_particles_from_hash_table(self, i, j, k):
        ''' Retrieves particles from the hash table based on grid coordinates.

            i: The grid coordinate along the x-axis.
            j: The grid coordinate along the y-axis.
            k: The grid coordinate along the z-axis.

        '''
        # Hash the grid coordinates to get the hash value
        hash_val = self.hash_coords(i, j, k)
        # Retrieve particles from the hash table based on the hash value, return empty list if not found
        return self.particleHashTable.get(hash_val, [])

    # Function to hash grid coordinates to a single value
    def hash_coords(self, i, j, k):
        ''' Hashes grid coordinates to a single value for indexing the hash table.

            i: The grid coordinate along the x-axis.
            j: The grid coordinate along the y-axis.
            k: The grid coordinate along the z-axis.

        '''
        # Combine grid coordinates using bitwise XOR and a prime number, then take absolute value and modulo to fit within hash table size
        h = (i * 92837111) ^ (j * 689287499) ^ (k * 123456789)
        return abs(h) % ((self.resolution[0]/2 * self.resolution[1]/2 * self.resolution[2]/2))

    # Code from ChatGPT ends here

    def clear(self):
        '''clear resets the hash table and the velocity, pressure grids.
        '''
        self.particleHashTable = {}

        self.velocity_u = np.zeros((self.resolution[0]+1, self.resolution[1], self.resolution[2]), dtype="float64")
        self.velocity_v = np.zeros((self.resolution[0], self.resolution[1]+1, self.resolution[2]), dtype="float64")
        self.velocity_w = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]+1), dtype="float64")
        self.pressure = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")


# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()