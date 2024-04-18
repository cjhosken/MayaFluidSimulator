from maya import cmds
from maya.api import OpenMaya as om
import os
import random
import re
import subprocess
import sys


# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy
# /usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy

import numpy as np

# To avoid creating Global variables that could be accessed by Maya, a Plugin class was implemented. This localizes all the variables to the script and reduces any script conflicts.
class MFS_Plugin():
    # For the addon to show images correctly, the project path must be set to the same folder as the script.
    project_path = cmds.workspace(query=True, rootDirectory=True)

    # Settings for the GUI
    popup_width = 500
    popup_height = 600
    button_ratio = 0.9

    def __init__(self):
        self.MFS_create_menu()

    # Create a menu bar heading at the top of the application.
    def MFS_create_menu(self):
        self.MFS_delete_menu()
        cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)
        cmds.menuItem(label="Open Maya Fluid Simulator", command=lambda x:self.MFS_popup(), image=os.path.join(self.project_path, "icons/MFS_icon_solver_512.png"))

    # Create the popup GUI in which users can control the fluid simulation settings.
    def MFS_popup(self):
        cmds.window(title="Maya Fluid Simulator", widthHeight=(self.popup_width, self.popup_height))
        col = cmds.columnLayout(adjustableColumn=True)

        cmds.image(width=self.popup_width/2, height=self.popup_height/4, image=os.path.join(self.project_path, "icons/MFS_banner.png"))

        initialize_section = cmds.frameLayout(label='Initialize', collapsable=True, collapse=False, parent=col)
        cmds.columnLayout(adjustableColumn=True, parent=initialize_section)
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.2, field=True, label="pscale")
        self.cell_size_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.3, field=True, label="Cell Size")
        self.random_sample_ctrl = cmds.intSliderGrp(minValue=0, value=0, field=True, label="Random Sampling")
        self.domain_size_ctrl = cmds.floatFieldGrp(numberOfFields=3, label='Domain Size', extraLabel='cm', value1=5, value2=5, value3=5)
        self.domain_ctrl = cmds.checkBox(label="Keep Domain", value=True)
    
        init_row = cmds.rowLayout(numberOfColumns=2, parent=initialize_section, adjustableColumn=True)
        cmds.button(label="Initialize", command=lambda x:self.MFS_initialize())
        cmds.button(label="X", command=lambda x:self.MFS_delete())
        cmds.rowLayout(init_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        simulate_section = cmds.frameLayout(label='Simulate', collapsable=True, collapse=False, parent=col)
        self.force_ctrl = cmds.floatFieldGrp(numberOfFields=3, label='Force', extraLabel='cm', value1=0, value2=-9.8, value3=0 )
        self.vel_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )

        self.density_ctrl = cmds.floatSliderGrp(minValue=0, maxValue=2000, step=0.01, value=998.2, field=True, label="Fluid Density")
        self.visc_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, field=True, label="Viscosity Factor")
        self.damp_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, maxValue=1, field=True, label="Floor Damping")
        
        cmds.rowLayout(numberOfColumns=3)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.1, field=True, label="Time Scale")

        solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)

        cmds.button(label="Simulate", command=lambda x:self.MFS_simulate())
        cmds.button(label="X", command=lambda x:self.MFS_reset())
        cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        cmds.columnLayout(adjustableColumn=True, parent=col)

        cmds.showWindow()

    # Garbage management. Helps with script reloads.
    def MFS_delete_menu(self):
        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)

    # Function for initializing the solver. This will create domains, a new solver, etc. At the moment, objects are limited to only one solver each.
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
            if (not keep_domain or not domain):
                if (domain):
                    cmds.delete(domain_name)
                
                domain_size = cmds.floatFieldGrp(self.domain_size_ctrl, query=True, value=True)
                domain_divs = (
                    int(domain_size[0] / cell_size),
                    int(domain_size[1] / cell_size),
                    int(domain_size[2] / cell_size)
                )

                domain = cmds.polyCube(name=domain_name, w=domain_size[0], h=domain_size[1], d=domain_size[2], sx=domain_divs[0], sy=domain_divs[1], sz=domain_divs[2])
                cmds.setAttr(domain[0] + ".overrideEnabled", 1)
                cmds.setAttr(domain[0] + ".overrideLevelOfDetail", 1)
            
            samples = cmds.intSliderGrp(self.random_sample_ctrl, query=True, value=True)
            points = self.mesh_to_points(source, pscale, samples)


            particle_group_name = source + "_particles"

            if (cmds.objExists(particle_group_name)):
                cmds.delete(particle_group_name)

            particle_group = cmds.createNode("transform", name=particle_group_name)
        
            
            for i,p in enumerate(points):
                particle_name = f"{source}_particle_{i:09}"
                if (cmds.objExists(particle_name)):
                    cmds.delete(particle_name)
                
                particle = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=particle_name)
                cmds.xform(particle, translation=p)
                cmds.parent(particle[0], particle_group)
            
            cmds.select(source)

    # The solver function is a container for the solver solve function. This allows for a progress window and the ability to quit once a frame is finished.
    def MFS_simulate(self):

        source = self.get_active_object()

        if (source is not None):
            
            self.MFS_reset()
            frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
            timescale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
            external_force = cmds.floatFieldGrp(self.force_ctrl, query=True, value=True)

            fluid_density = cmds.floatSliderGrp(self.density_ctrl, query=True, value=True)
            viscosity_factor = cmds.floatSliderGrp(self.visc_ctrl, query=True, value=True)
            damping =  (1 - cmds.floatSliderGrp(self.damp_ctrl, query=True, value=True))

            particles = cmds.listRelatives(source + "_particles", children=True) or []
            points = []

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
            grid = MFS_Grid(resolution, cell_size)

            cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frame_range[1]-frame_range[0]))
            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, 0)


    def update(self, source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, progress):
        percent = (progress / (frame_range[1] - frame_range[0])) * 100

        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {percent:.1f}%')
        t = int(cmds.currentTime(query=True))
        solved = (t < frame_range[0] or t > frame_range[1])
        cancelled = cmds.progressWindow(query=True, isCancelled=True)

        if (not (solved or cancelled)):
            self.keyframe(source, points, t)
            print(f"Simulating Frame: {t}")
            grid.clear()
            grid.from_particles(source, points)
            grid.calc_forces(external_force, viscosity_factor)
            grid.advect(source, timescale)
            grid.to_particles(source, points, timescale, damping)
            
            cmds.currentTime(t + 1, edit=True)

            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, progress=progress+1)
        else:
            cmds.currentTime(frame_range[0], edit=True)
            cmds.progressWindow(endProgress=1)

    def keyframe(self, source, points, t):
        for p in points:
            particle_name = f"{source}_particle_{p.id:09}"
            cmds.setKeyframe(particle_name, attribute='translateX', t=t, v=p.position[0])
            cmds.setKeyframe(particle_name, attribute='translateY', t=t, v=p.position[1])
            cmds.setKeyframe(particle_name, attribute='translateZ', t=t, v=p.position[2])

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

    def mesh_to_points(self, mesh_name, cell_size, samples=0):
        # Get mesh bounding box
        bbox = cmds.exactWorldBoundingBox(mesh_name)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # Calculate step sizes for each axis
        subdivisions = (
            int((max_x - min_x) / cell_size),
            int((max_y - min_y) / cell_size),
            int((max_z - min_z) / cell_size)
        )

        # Generate points inside the mesh
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

                            # Test if the point is inside the mesh
                            if self.is_point_inside_mesh(point, mesh_name):
                                points.append(point)
                    else:
                        x = min_x + (i + 0.5) * cell_size
                        y = min_y + (j + 0.5) * cell_size
                        z = min_z + (k + 0.5) * cell_size
                        point = (x, y, z)

                        # Test if the point is inside the mesh
                        if self.is_point_inside_mesh(point, mesh_name):
                            points.append(point)


        return points

    def is_point_inside_mesh(self, point, mesh_name):
        # Create a ray from the point in a specific direction (e.g., towards positive X)
        direction = om.MVector(random.random(), random.random(), random.random()) # make this a random direction
        
        # Get the DAG path for the mesh
        selection_list = om.MSelectionList()
        selection_list.add(mesh_name)
        dag_path = selection_list.getDagPath(0)
        
        # Perform ray intersection with the mesh
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

    def advect(self, source, velocity, damping, dt):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        self.velocity = velocity

        advected = self.position + self.velocity * dt

        if (advected[0] < min_x or advected[0] > max_x):
            self.velocity[0] = -self.velocity[0] 

        if (advected[1] < min_y):
            self.velocity[1] = -(advected[1] - min_y) * damping

        if (advected[1] > max_y):
            self.velocity[1] = -self.velocity[1]
                                
        if (advected[2] < min_z or advected[2] > max_z):
            self.velocity[2] = -self.velocity[2]

        self.position += self.velocity * dt

class MFS_Grid():
    def __init__(self, res, cell_size) -> None:
        self.resolution = res
        self.cell_size = cell_size
        self.bounds = cell_size * self.resolution

        initial_value = MFS_Cell()
        self.cells = np.full((self.resolution[0], self.resolution[1], self.resolution[2]), initial_value)

    def from_particles(self, source, points):
        for point in points:
            i, j, k = self.get_cell(source, point)

            self.cells[i][j][k].velocity += point.velocity
            self.cells[i][j][k].particle_count += 1

        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    if (self.cells[i][j][k].particle_count > 0):
                        self.cells[i][j][k].velocity /= self.cells[i][j][k].particle_count

    def calc_forces(self, external_force, viscosity):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.cells[i][j][k].force = np.array(external_force, dtype="float64")

    def advect(self, source, dt):
        new_cells = np.full((self.resolution[0], self.resolution[1], self.resolution[2]), MFS_Cell())
    
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    velocity = self.cells[i][j][k].velocity
                    force = self.cells[i][j][k].force

                    # Compute the advected position using the velocity
                    advected_i = i + int(velocity[0] * dt / self.cell_size[0])
                    advected_j = j + int(velocity[1] * dt / self.cell_size[1])
                    advected_k = k + int(velocity[2] * dt / self.cell_size[2])
                    
                    # Interpolate velocity and force from surrounding cells
                    if (0 <= advected_i < self.resolution[0] and
                        0 <= advected_j < self.resolution[1] and
                        0 <= advected_k < self.resolution[2]):
                        
                        # Update the velocity and force in the new cell
                        new_cells[advected_i][advected_j][advected_k].velocity = velocity + force * dt
                    
                    


        # Update the grid with the advected velocities and forces
        self.cells = new_cells

    def interpolate_velocity(self, i, j, k):
        # Implement interpolation method (e.g., trilinear interpolation)
        # Here you can use various interpolation techniques such as trilinear, tricubic, etc.
        # For simplicity, let's assume linear interpolation for now
        return self.cells[i][j][k].velocity

    def interpolate_force(self, i, j, k):
        # Implement interpolation method for force (similar to velocity interpolation)
        return self.cells[i][j][k].force


    def to_particles(self, source, points, dt, damping):
        for point in points:
            i, j, k = self.get_cell(source, point)
            velocity = self.cells[i][j][k].velocity
            point.advect(source, velocity, damping, dt)

    def get_cell(self, source, point):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        i = int((point.position[0] - min_x) / self.cell_size[0])
        j = int((point.position[1] - min_y) / self.cell_size[1])
        k = int((point.position[2] - min_z) / self.cell_size[2])
        return (i, j, k)

    def clear(self):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    self.cells[i][j][k].velocity = np.array([0, 0, 0], dtype="float64")
                    self.cells[i][j][k].force = np.array([0, 0, 0], dtype="float64")
                    self.cells[i][j][k].pressure = 0
                    self.cells[i][j][k].density = 0
                    self.cells[i][j][k].particle_count = 0

class MFS_Cell():
    def __init__(self) -> None:
        self.velocity = np.array([0, 0, 0], dtype="float64")
        self.pressure = 0
        self.density = 0
        self.force = np.array([0, 0, 0], dtype="float64")
        self.particle_count = 0

# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()