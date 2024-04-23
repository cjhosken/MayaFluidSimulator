from maya import cmds
from maya.api import OpenMaya as om
import os
import random
import re


# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy
# /usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy

import numpy as np

class MFS_Plugin():
    project_path = cmds.workspace(query=True, rootDirectory=True)

    popup_width = 500
    popup_height = 600
    button_ratio = 0.9

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
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.25, field=True, label="pscale")
        self.cell_size_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.25, field=True, label="Cell Size")
        self.random_sample_ctrl = cmds.intSliderGrp(minValue=0, value=0, field=True, label="Random Sampling")

        #TODO: Users can set the domain to have a negative size. handle this in the code. (maybe apply scale or something?)
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
        self.damp_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.9, maxValue=1, field=True, label="Floor Damping")
        
        cmds.rowLayout(numberOfColumns=3)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.1, field=True, label="Time Scale")

        solve_row = cmds.rowLayout(numberOfColumns=2, parent=simulate_section, adjustableColumn = True)

        cmds.button(label="Simulate", command=lambda x:self.MFS_simulate())
        cmds.button(label="X", command=lambda x:self.MFS_reset())
        cmds.rowLayout(solve_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        cmds.columnLayout(adjustableColumn=True, parent=col)

        cmds.showWindow()

    def MFS_delete_menu(self):
        if cmds.menu("MFS_menu", exists=True):
            cmds.deleteUI("MFS_menu", menu=True)


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

    
    def MFS_simulate(self):

        source = self.get_active_object()

        if (source is not None and self.can_simulate(source)):

            self.MFS_reset()
            frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
            timescale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
            external_force = cmds.floatFieldGrp(self.force_ctrl, query=True, value=True)

            fluid_density = cmds.floatSliderGrp(self.density_ctrl, query=True, value=True)
            viscosity_factor = cmds.floatSliderGrp(self.visc_ctrl, query=True, value=True)
            damping =  (1 - cmds.floatSliderGrp(self.damp_ctrl, query=True, value=True))
            pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)

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
            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, 0)


    def update(self, source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, progress):
        percent = (progress / (frame_range[1] - frame_range[0])) * 100

        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {percent:.1f}%')
        t = int(cmds.currentTime(query=True))
        solved = (t < frame_range[0] or t > frame_range[1])
        cancelled = cmds.progressWindow(query=True, isCancelled=True)

        if (not (solved or cancelled)):
            self.keyframe(source, points, t)
            print(f"Simulating Frame: {t}")

            cfl_time = 0

            while (cfl_time < timescale):
                timestep = grid.from_particles(source, points, timescale)
                grid.calc_forces(external_force, viscosity_factor, timestep)
                grid.enforce_boundaries()
                grid.solve_poisson()
                grid.to_particles(source, points, timestep, damping)
                cfl_time += timestep
            
            cmds.currentTime(t + 1, edit=True)

            self.update(source, points, grid, frame_range, timescale, external_force, fluid_density, viscosity_factor, damping, pscale, progress=progress+1)
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

    def can_simulate(self, source):
        can_simulate = cmds.objExists(source + "_domain") and cmds.objExists(source + "_particles")

        if (not can_simulate):
            cmds.confirmDialog(title="Simulation Error!", 
                message="You need to initialize a source object!",
                button="Oopsies"
            )

        return can_simulate
    

    # Mesh to points generates points inside of a given object.
    # It creates a domain around the source object and splits it into subdivisions. 
    # It then generates points inside the cells and checks if those points are inside the object
    # When Samples = 0, it creates a uniform grid insde the object. When samples > 0, it randomly generates n(samples) points.

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

    # Is point inside mesh takes a point and fires a random vector out from it. 
    # If the ray intersects with a source object an uneven number of times, then it is inside the object.

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

    def advect(self, source, velocity, damping, dt):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        print("PVEL: ", velocity)

        self.velocity = velocity

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
        self.type = np.zeros((self.resolution[0], self.resolution[1], self.resolution[2], 3), dtype="float64")

    def from_particles(self, source, points, timescale):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        self.clear()

        # This is also known as P2G. The velocity values of the particles need to be projected onto the grid.
        # This is usually done using trillinear interpolation.

        for point in points:
            x = (point.position[0] - min_x) / self.cell_size[0] - 0.5
            y = (point.position[1] - min_y) / self.cell_size[1] - 0.5
            z = (point.position[2] - min_z) / self.cell_size[2] - 0.5

            i = int(x + 0.5)
            j = int(y + 0.5)
            k = int(z + 0.5)

            dx = x - i
            dy = y - j
            dz = z - k

            self.type[i][j][k] = 1

            # Trilinear interpolation for velocity components
            v000 = point.velocity * (1 - dx) * (1 - dy) * (1 - dz)
            v100 = point.velocity * dx * (1 - dy) * (1 - dz)
            v010 = point.velocity * (1 - dx) * dy * (1 - dz)
            v110 = point.velocity * dx * dy * (1 - dz)
            v001 = point.velocity * (1 - dx) * (1 - dy) * dz
            v101 = point.velocity * dx * (1 - dy) * dz
            v011 = point.velocity * (1 - dx) * dy * dz
            v111 = point.velocity * dx * dy * dz

            # Update velocity grid using trilinear interpolation
            self.velocity[i][j][k] += v000
            self.velocity[min(i + 1, self.resolution[0] - 1)][j][k] += v100
            self.velocity[i][min(j + 1, self.resolution[1] - 1)][k] += v010
            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k] += v110
            self.velocity[i][j][min(k + 1, self.resolution[2] - 1)] += v001
            self.velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)] += v101
            self.velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += v011
            self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)] += v111

            print("VEL1: ", self.velocity[i][j][k])

        max_vel = 0.001

        # Calculate timestep based on maximum velocity
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    max_vel = max(max_vel, np.linalg.norm(self.velocity[i][j][k]))

        timestep = timescale * min(np.linalg.norm(self.cell_size) / max_vel, 1)

        return timestep


    def calc_forces(self, external_force, viscosity, dt):
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    total_force = np.array(external_force, dtype="float64")
                    self.velocity[i][j][k] += total_force * dt

    def enforce_boundaries(self):
        #TODO: Enforing boundary conditions so that particles arent projected out of the domain
        pass

    def solve_poisson(self):
        #TODO: Solve the poisson equation to make the fluid non-divergent.
        pass

    def to_particles(self, source, points, dt, damping):
        bbox = cmds.exactWorldBoundingBox(source + "_domain")
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        for point in points:
            x = (point.position[0] - min_x) / self.cell_size[0]
            y = (point.position[1] - min_y) / self.cell_size[1]
            z = (point.position[2] - min_z) / self.cell_size[2]

            i = int(x)
            j = int(y)
            k = int(z)

            #TODO: FLIP / PIC METHOD
            # FLIP interpolates the change of velocity and adds it to the existing velocity
            # PIC replaces the velocity

            #TODO: To avoid glitching, do a backwards trace to find the velocity.

            #TODO: The particles need to get the trillinearly interpolated velocity from the grid.

            velocity = self.trilinear_interpolate_velocity(x - 0.5, y - 0.5, z - 0.5)
            point.advect(source, velocity, damping, dt)

    def trilinear_interpolate_velocity(self, x, y, z):
        i = int(x + 0.5)
        j = int(y + 0.5)
        k = int(z + 0.5)
        dx = x - i
        dy = y - j
        dz = z - k

        # Clamp indices to stay within the grid boundaries
        i = max(0, min(i, self.resolution[0] - 1))
        j = max(0, min(j, self.resolution[1] - 1))
        k = max(0, min(k, self.resolution[2] - 1))

        # Trilinear interpolation for u component
        v000 = self.velocity[i][j][k]
        v100 = self.velocity[min(i + 1, self.resolution[0] - 1)][j][k]
        v010 = self.velocity[i][min(j + 1, self.resolution[1] - 1)][k]
        v110 = self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][k]
        v001 = self.velocity[i][j][min(k + 1, self.resolution[2] - 1)]
        v101 = self.velocity[min(i + 1, self.resolution[0] - 1)][j][min(k + 1, self.resolution[2] - 1)]
        v011 = self.velocity[i][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)]
        v111 = self.velocity[min(i + 1, self.resolution[0] - 1)][min(j + 1, self.resolution[1] - 1)][min(k + 1, self.resolution[2] - 1)]

        # Trilinear interpolation for v and w components and interpolate them similarly

        # Return the interpolated velocity
        return trilinear_interpolate(v000, v100, v010, v110, v001, v101, v011, v111, dx, dy, dz)
  

    def clear(self):
        self.velocity = np.zeros((self.resolution[0] + 1, self.resolution[1] + 1, self.resolution[2] + 1, 3), dtype="float64")

def linear_interpolate(p0, p1, t):
    weight = 1 - t if t > 0.5 else t
    return p0 * (1 - weight) + p1 * weight

def bilinear_interpolate(p00, p10, p01, p11, x, y):
    v0 = linear_interpolate(p00, p10, x)
    v1 = linear_interpolate(p01, p11, x)

    return linear_interpolate(v0, v1, y)

def trilinear_interpolate(p000, p100, p010, p110, p001, p101, p011, p111, x, y, z):
    v00 = bilinear_interpolate(p000, p100, p010, p110, x, y)
    v01 = bilinear_interpolate(p001, p101, p011, p111, x, y)
    return linear_interpolate(v00, v01, z)

# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()