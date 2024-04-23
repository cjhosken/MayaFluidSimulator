''' Maya Fluid Simulator

'''

from maya import cmds
from maya.api import OpenMaya as om
import os
import random
import re
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy
# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user scipy
# /usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy
# /usr/autodesk/maya2023/bin/mayapy -m pip install --user scipy

import numpy as np

''' The MFS Plugin is a general class that contains the (mainly) Maya side of the script. This allows the use of "global" variables in a contained setting. '''
class MFS_Plugin():
    
    # "global" variables
    project_path = cmds.workspace(query=True, rootDirectory=True)
    popup_size = (500, 600)
    button_ratio = 0.9


    ''' Initialize the plugin by creating the header menu. '''
    def __init__(self):
        self.MFS_create_menu()


    ''' Delete any pre-existing Maya Fluid Simulator header menus, then create a new header menu.'''
    def MFS_create_menu(self):
        self.MFS_delete_menu()
        cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)
        cmds.menuItem(label="Open Maya Fluid Simulator", command=lambda x:self.MFS_popup(), image=os.path.join(self.project_path, "icons/MFS_icon_solver_512.png"))
    
    
    ''' Creates the main UI for Maya Fluid Simulator. 

        PScale                 : the size of the 
        Cell Size              : the size of the 
        Random Sampling        : the size of the 
        Domain Size            : the size of the 
        Keep Domain            : the size of the 
        Initialize (X)         : the size of the 

        Force                  : the size of the 
        Initial Velocity       : the size of the 
        Fluid Density          : the size of the 
        Viscosity Factor       : the size of the 
        Floor Damping          : the size of the 
        PIC/FLIP Mix           : the size of the 
        Frame Range            : the size of the 
        Time Scale             : the size of the 
        Simulate (X)           : the size of the 



        Setting the Maya project dir to the script location shows the plugin icons.
    '''
    def MFS_popup(self):
        
        cmds.window(title="Maya Fluid Simulator", widthHeight=self.popup_size)
        col = cmds.columnLayout(adjustableColumn=True)

        cmds.image(width=self.popup_width/2, height=self.popup_height/4, image=os.path.join(self.project_path, "icons/MFS_banner.png"))

        initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)
        cmds.columnLayout(adjustableColumn=True, parent=initialize_section)
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.25, field=True, label="PScale")
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


    ''' Checks if the Maya Fluid Simulator header menu exists and deletes it. '''
    def MFS_delete_menu(self):
        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)


    ''' Initialize uses the initialization settings to fill a selected object with fluid particles and to create a domain object. '''
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


    ''' This begins the Maya Fluid Simulation from the given fluid settings. '''
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

            particles = cmds.listRelatives(source + "_particles", children=True) or []
            points = []

            # To allow the tool to simulate, even after the user has restarted Maya, the plugin looks at all the Maya object particles and creates a MFS_Particle class. 
            # It then adds all the particles to a point array to be used in the simulation.
            for p in particles:
                id = int(re.search(r"\d+$", p).group())
                position = np.array(cmds.xform(p, query=True, translation=True, worldSpace=True), dtype="float64")
                velocity = np.array(cmds.floatFieldGrp(self.vel_ctrl, query=True, value=True), dtype="float64")

                point = MFS_Particle(
                    id=id,
                    pos=position,
                    vel=velocity
                )

                points = np.append(point, points)

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
            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, 0)


    ''' Update is the start of an actual Fluid Simulator. It simulates the particle position frame by frame, keyframing the positions each time. 
    The user is then able to cancel the simulation by pressing Esc, however they need to wait for the current frame to finish simulation.
    '''
    def update(self, source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, progress):
        percent = (progress / (frame_range[1] - frame_range[0])) * 100

        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {percent:.1f}%')
        t = int(cmds.currentTime(query=True))
        solved = (t < frame_range[0] or t > frame_range[1])
        cancelled = cmds.progressWindow(query=True, isCancelled=True)

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
            self.keyframe(source, points, t)
            print(f"Simulating Frame: {t}")

            cfl_time = 0

            while (cfl_time < timescale):
                grid.from_particles(source, points)
                timestep = grid.calc_timestep(external_force, timescale)
                grid.calc_forces(external_force, timestep)
                #grid.enforce_boundaries()
                grid.solve_poisson()
                grid.to_particles(source, points, timestep, damping, flipFac)
                cfl_time += timestep
            
            cmds.currentTime(t + 1, edit=True)

            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, flipFac, progress=progress+1)
        else:
            cmds.currentTime(frame_range[0], edit=True)
            cmds.progressWindow(endProgress=1)


    ''' Take the simulation points, copy their positions into the Maya particles, then keyframe the positions.'''
    def keyframe(self, source, points, t):
        for p in points:
            particle_name = f"{source}_particle_{p.id:09}"
            cmds.setKeyframe(particle_name, attribute='translateX', t=t, v=p.position[0])
            cmds.setKeyframe(particle_name, attribute='translateY', t=t, v=p.position[1])
            cmds.setKeyframe(particle_name, attribute='translateZ', t=t, v=p.position[2])

    ''' Delete all the particles and the domain, and revert back the source object.'''
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
    

    ''' Reset the simulation by deleting all the keyframes and setting the current time to the start of the frame range.'''
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


    ''' This checks if the selected object can b a "source" object.
        An object can be a source object if it is not a domain or a particle.
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


    ''' This checks if the selected object has been sourced. To do so, it looks to see if a domain and particle group exist for the object.'''
    def can_simulate(self, source):
        can_simulate = cmds.objExists(source + "_domain") and cmds.objExists(source + "_particles")

        if (not can_simulate):
            cmds.confirmDialog(title="Simulation Error!", 
                message="You need to initialize a source object!",
                button="Oopsies"
            )

        return can_simulate
    

    ''' Mesh to points generates points inside of a given object.
        It creates a domain around the source object and splits it into subdivisions, then It creates a domain around the source object and splits it into subdivisions. 

        When Samples = 0, it creates a uniform grid insde the object. When samples > 0, it randomly generates n(samples) points.
    '''
    def mesh_to_points(self, mesh_name, cell_size, samples=0):
    
        bbox = cmds.exactWorldBoundingBox(mesh_name)
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

                            if self.is_point_inside_mesh(point, mesh_name):
                                points.append(point)
                    else:
                        x = min_x + (i + 0.5) * cell_size
                        y = min_y + (j + 0.5) * cell_size
                        z = min_z + (k + 0.5) * cell_size
                        point = (x, y, z)

                        if self.is_point_inside_mesh(point, mesh_name):
                            points.append(point)


        return points


    ''' This looks to see if a point is inside a specific mesh.
        To do this, a ray is fired from (at a random direciton) the point. 
        If the ray interescts with the mesh an uneven number of times, the point is inside the mesh.

        This only works if the mesh is enclosed, any gaps in the geometry can lead to unstable results.
    '''
    def is_point_inside_mesh(self, point, mesh_name):
        direction = om.MVector(random.random(), random.random(), random.random())
        
        selection_list = om.MSelectionList()
        selection_list.add(mesh_name)
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


class MFS_Particle():
    def __init__(self, id, pos, vel) -> None:
        self.id = id
        self.position = pos
        self.velocity = vel

    def advect(self, source, velocity, velocity_change, damping, dt, flipFac):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        flipFac = max(min(flipFac, 1.0), 0.0)


        pic_vel = velocity
        flip_vel = self.velocity + velocity_change

        self.velocity = (flipFac) * flip_vel + (1-flipFac) * pic_vel

        advected = self.position + self.velocity

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

class MFS_Grid():
    def __init__(self, res, cell_size) -> None:
        self.resolution = res
        self.cell_size = cell_size
        
        self.velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")
        self.last_velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")
        self.type = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2]), dtype="float64")

    def from_particles(self, source, points):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        self.clear()

        # This is also known as P2G. The velocity values of the particles need to be projected onto the grid.
        # This is usually done using trillinear interpolation.

        weights = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1), dtype="float64")

        for point in points:
            x = (point.position[0] - min_x) / self.cell_size[0]
            y = (point.position[1] - min_y) / self.cell_size[1]
            z = (point.position[2] - min_z) / self.cell_size[2]

            i = int(x)
            j = int(y)
            k = int(z)

            dx = x - i
            dy = y - j
            dz = z - k

            # Clamp indices to stay within the grid boundaries
            i = max(0, min(i, self.resolution[0] - 1))
            j = max(0, min(j, self.resolution[1] - 1))
            k = max(0, min(k, self.resolution[2] - 1))

            self.type[i][j][k] = 1

            # Trilinear interpolation for velocity components
            w000 = (1 - dx) * (1 - dy) * (1 - dz)
            w100 = dx * (1 - dy) * (1 - dz)
            w010 = (1 - dx) * dy * (1 - dz)
            w110 = dx * dy * (1 - dz)
            w001 = (1 - dx) * (1 - dy) * dz
            w101 = dx * (1 - dy) * dz
            w011 = (1 - dx) * dy * dz
            w111 = dx * dy * dz

            # Update velocity grid using trilinear interpolation
            self.velocity[i][j][k] += point.velocity * w000
            weights[i][j][k] += w000

            self.velocity[min(i + 1, self.resolution[0] - 1)][j][k] += point.velocity * w100
            weights[min(i + 1, self.resolution[0] - 1)][j][k] += w100

            self.velocity[i][min(j + 1, self.resolution[1] - 1)][k] += point.velocity * w010
            weights[i][min(j + 1, self.resolution[1] - 1)][k] += w010

            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] += point.velocity * w110
            weights[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] += w110

            self.velocity[i][j][min(k + 1, self.resolution[2] - 1)] += point.velocity * w001
            weights[i][j][min(k + 1, self.resolution[2] - 1)] += w001

            self.velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] += point.velocity * w101
            weights[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] += w101

            self.velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += point.velocity * w011
            weights[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += w011

            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += point.velocity * w111
            weights[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += w111

        # normalize the velocities
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (weights[i][j][k] != 0):
                        self.velocity[i][j][k] /= weights[i][j][k]

        self.last_velocity = np.array(self.velocity, copy=True)
    
    def calc_timestep(self, external_force, timescale):
        max_vel = 0

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    max_vel = max(max_vel, np.linalg.norm(self.velocity[i][j][k]))

        max_vel += np.linalg.norm(external_force * self.cell_size)

        timestep = timescale

        if (max_vel > 0):
            timestep = max(timestep, timescale * np.linalg.norm(self.cell_size) / max_vel)

        return timestep

    def calc_forces(self, external_force, dt):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    total_force = external_force
                    self.velocity[i][j][k] += total_force * dt

    def enforce_boundaries(self):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (i-1 < 0 or i+1 >= self.resolution[0]):
                        self.velocity[i][j][k][0] = 0

                    if (j-1 < 0 or j+1 >= self.resolution[1]):
                        self.velocity[i][j][k][1] = 0

                    if (k-1 < 0 or k+1 >= self.resolution[2]):
                        self.velocity[i][j][k][2] = 0

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

        # Find the pressyre divergence

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

    def correct_velocity(self, pressure):
        dx, dy, dz = self.cell_size

        for i in range(1, self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.velocity[i][j][k][0] -= (pressure[i][j][k] - pressure[i - 1][j][k]) / dx

        # Correct v-component of velocity (stored at y edges)
        for i in range(self.resolution[0]):
            for j in range(1, self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.velocity[i][j][k][1] -= (pressure[i][j][k] - pressure[i][j - 1][k]) / dy

        # Correct w-component of velocity (stored at z edges)
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(1, self.resolution[2]):
                    self.velocity[i][j][k][2] -= (pressure[i][j][k] - pressure[i][j][k - 1]) / dz
            

    def to_particles(self, source, points, dt, damping, flipFac):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        for point in points:
            x = (point.position[0] - min_x) / self.cell_size[0]
            y = (point.position[1] - min_y) / self.cell_size[1]
            z = (point.position[2] - min_z) / self.cell_size[2]

            last_x = x - point.velocity[0] * dt / self.cell_size[0]
            last_y = y - point.velocity[1] * dt / self.cell_size[1]
            last_z = z - point.velocity[2] * dt / self.cell_size[2]

        

            velocity, vw = self.trilinear_interpolate_velocity(x, y, z)
            last_velocity, lvw = self.trilinear_interpolate_last_velocity(last_x, last_y, last_z)

            velocity_change = velocity/vw - last_velocity/lvw

            point.advect(source, velocity/vw, velocity_change, damping, dt, flipFac)

    def trilinear_interpolate_velocity(self, x, y, z):
        i = int(x)
        j = int(y)
        k = int(z)
        dx = x - i
        dy = y - j
        dz = z - k

        # Clamp indices to stay within the grid boundaries
        i = max(0, min(i, self.resolution[0] - 1))
        j = max(0, min(j, self.resolution[1] - 1))
        k = max(0, min(k, self.resolution[2] - 1))

        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w100 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w110 = dx * dy * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w101 = dx * (1 - dy) * dz
        w011 = (1 - dx) * dy * dz
        w111 = dx * dy * dz

        velocity = (
            self.velocity[i][j][k] * w000 + 
            self.velocity[min(i + 1, self.resolution[0] - 1)][j][k] * w100 +
            self.velocity[i][min(j + 1, self.resolution[1] - 1)][k] * w010 +
            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] * w110 +
            self.velocity[i][j][min(k + 1, self.resolution[2] - 1)] * w001 +
            self.velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] * w101 +
            self.velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w011 +
            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w111
        )

        weight = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111

        # Trilinear interpolation for v and w components and interpolate them similarly

        # Return the interpolated velocity
        return (velocity, weight)
    
    def trilinear_interpolate_last_velocity(self, x, y, z):
        i = int(x)
        j = int(y)
        k = int(z)

        dx = x - i
        dy = y - j
        dz = z - k

        # Clamp indices to stay within the grid boundaries
        i = max(0, min(i, self.resolution[0] - 1))
        j = max(0, min(j, self.resolution[1] - 1))
        k = max(0, min(k, self.resolution[2] - 1))

        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w100 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w110 = dx * dy * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w101 = dx * (1 - dy) * dz
        w011 = (1 - dx) * dy * dz
        w111 = dx * dy * dz

        velocity = (
            self.last_velocity[i][j][k] * w000 + 
            self.last_velocity[min(i + 1, self.resolution[0] - 1)][j][k] * w100 +
            self.last_velocity[i][min(j + 1, self.resolution[1] - 1)][k] * w010 +
            self.last_velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] * w110 +
            self.last_velocity[i][j][min(k + 1, self.resolution[2] - 1)] * w001 +
            self.last_velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] * w101 +
            self.last_velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w011 +
            self.last_velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] * w111
        )

        weight = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111

        # Trilinear interpolation for v and w components and interpolate them similarly

        # Return the interpolated velocity
        return (velocity, weight)
  
    def clear(self):
        self.velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")

# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()