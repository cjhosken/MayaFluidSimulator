from maya import cmds
from maya.api import OpenMaya as om
import os, math
import numpy as np
from collections import defaultdict
import random
import re

# To avoid creating Global variables that could be accessed by Maya, a Plugin class was implemented. This localizes all the variables to the script and reduces any script conflicts.
class MFS_Plugin():
    # For the addon to show images correctly, the project path must be set to the same folder as the script.
    project_path = cmds.workspace(query=True, rootDirectory=True)
    solvers = np.array([])

    # A dictionary that holds the names of generated objects. This can be customizable.
    names = dict({
            "domain":"_domain",
            "particles":"_particle"
        }
    )
    max_length = 9

    # Settings for the GUI
    popup_width = 500
    popup_height = 600
    button_ratio = 0.9

    pscale_ctrl = None
    domain_ctrl = None
    domain_size_ctrl = None
    domain_divs_ctrl = None

    force_ctrl = None
    visc_ctrl = None
    vel_ctrl = None
    time_ctrl = None
    ts_ctrl = None

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
        self.pscale_ctrl = cmds.floatSliderGrp(minValue=0, step=0.01, value=0.3, field=True, label="Particle Scale")
        self.domain_ctrl = cmds.checkBox(label="Keep Domain", value=True)
        self.domain_size_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Domain Size', extraLabel='cm', value1=5, value2=5, value3=5 )
        self.domain_divs_ctrl = cmds.intFieldGrp( numberOfFields=3, label='Domain Divisions', extraLabel='cm', value1=64, value2=64, value3=64 )
    
        init_row = cmds.rowLayout(numberOfColumns=2, parent=initialize_section, adjustableColumn=True)
        cmds.button(label="Initialize", command=lambda x:self.MFS_initialize())
        cmds.button(label="X", command=lambda x:self.MFS_delete())
        cmds.rowLayout(init_row, edit=True, columnWidth=[(1, self.button_ratio * self.popup_width), (2, (1-self.button_ratio) * self.popup_width)])

        simulate_section = cmds.frameLayout(label='Simulate', collapsable=True, collapse=False, parent=col)
        self.vel_ctrl = cmds.floatFieldGrp( numberOfFields=3, label='Initial Velocity', extraLabel='cm', value1=0, value2=0, value3=0 )
        
        cmds.rowLayout(numberOfColumns=3)
        self.time_ctrl = cmds.intFieldGrp(numberOfFields=2, value1=0, value2=120, label="Frame Range")
        self.ts_ctrl = cmds.floatSliderGrp(minValue=0, step=0.001, value=0.01, field=True, label="Time Scale")

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

            if (not keep_domain or not domain):
                if (domain):
                    cmds.delete(domain_name)
                
                domain_size = cmds.floatFieldGrp(self.domain_size_ctrl, query=True, value=True)
                domain_divs = cmds.intFieldGrp(self.domain_divs_ctrl, query=True, value=True)
                domain = cmds.polyCube(name=domain_name, w=domain_size[0], h=domain_size[1], d=domain_size[2], sx=domain_divs[0], sy=domain_divs[1], sz=domain_divs[2])
                cmds.setAttr(domain[0] + ".overrideEnabled", 1)
                cmds.setAttr(domain[0] + ".overrideLevelOfDetail", 1)

            pscale = cmds.floatSliderGrp(self.pscale_ctrl, query=True, value=True)

            subdivisions = int(1 / pscale) * 2

            points = self.mesh_to_points(source, subdivisions)


            particle_group_name = source + "_particles"

            if (cmds.objExists(particle_group_name)):
                cmds.delete(particle_group_name)

            particle_group = cmds.createNode("transform", name=particle_group_name)
        
            particle_id = 1

            for p in points:
                particle_name = f"{source}_particle_{particle_id:09}"
                if (cmds.objExists(particle_name)):
                    cmds.delete(particle_name)
                
                particle = cmds.polySphere(radius=pscale/2, subdivisionsY=4, subdivisionsX=6, name=particle_name)
                cmds.xform(particle, translation=p)
                cmds.parent(particle, particle_group)

                particle_id += 1
            
            cmds.select(source)

        else:
            pass

    # The solver function is a container for the solver solve function. This allows for a progress window and the ability to quit once a frame is finished.
    def MFS_simulate(self):
        source = self.get_active_object()

        if (source is not None):
            frame_range = cmds.intFieldGrp(self.time_ctrl, query=True, value=True)
            timescale = cmds.floatSliderGrp(self.ts_ctrl, query=True, value=True)
            cmds.currentTime(frame_range[0], edit=True)

            particles = cmds.listRelatives(source + "_particles", children=True) or []
            points = []

            for p in particles:

                id = int(re.search(r"\d+$", p).group())
                position = cmds.xform(p, query=True, translation=True, worldSpace=True)
                velocity = cmds.floatFieldGrp(self.vel_ctrl, query=True, value=True)
                mass = 1

                points.append(MFS_Particle(
                    id=id,
                    pos=position,
                    vel=velocity,
                    mass=mass
                ))

            resx = int(cmds.polyCube(source + "_domain", query=True, subdivisionsX=True))
            resy = int(cmds.polyCube(source + "_domain", query=True, subdivisionsY=True))
            resz = int(cmds.polyCube(source + "_domain", query=True, subdivisionsZ=True))
            resolution = np.array([resx, resy, resz])
            grid = MFS_Grid(resolution)

            cmds.progressWindow(title='Simulating', progress=0, status='Progress: 0%', isInterruptable=True, maxValue=(frame_range[1]-frame_range[0]))
            self.update(source, points, grid, frame_range, timescale, 0)

    def update(self, source, points, grid, frame_range, timescale, progress):
        cmds.progressWindow(e=1, progress=progress, status=f'Progress: {progress}%')
        t = int(cmds.currentTime(query=True))
        solved = (t < frame_range[0] or t > frame_range[1])
        cancelled = cmds.progressWindow(query=True, isCancelled=True)

        if (not (solved or cancelled)):
            self.keyframe(source, points, t)
            
            
            
            cmds.currentTime(t + 1, edit=True)
            self.update(source, points, grid, frame_range, timescale, progress=progress+1)
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
        

        pass
    
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
                # If none of the forbidden labels are found in the object's name, return the object
                return active_object
                
        cmds.confirmDialog(title="Source Error!", 
            message="You need to select a source object!",
            button="Oopsies"
        )
        
        return None

    def mesh_to_points(self, mesh_name, subdivisions):
        # Get mesh bounding box
        bbox = cmds.exactWorldBoundingBox(mesh_name)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # Calculate step sizes for each axis
        step_x = (max_x - min_x) / subdivisions
        step_y = (max_y - min_y) / subdivisions
        step_z = (max_z - min_z) / subdivisions

        # Generate points inside the mesh
        points = []
        for i in range(subdivisions):
            for j in range(subdivisions):
                for k in range(subdivisions):
                    # Calculate cell center
                    x = min_x + (i + 0.5) * step_x
                    y = min_y + (j + 0.5) * step_y
                    z = min_z + (k + 0.5) * step_z
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

            # Count the number of intersections
            num_intersections = len(intersections[0])
        
            # If the number of intersections is odd, the point is inside the mesh
            return num_intersections % 2 != 0

        return False


class MFS_Particle():
    def __init__(self, id, pos, vel, mass) -> None:
        self.id = id
        self.position = np.array(pos, dtype=float)
        self.velocity = np.array(vel, dtype=float)
        self.mass = mass

    def update(self, force, dt):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class MFS_Grid():

    def __init__(self, res) -> None:
        self.resolution = res
        self.cells = np.zeros(res[0], res[1], res[2])

# Create and initialize the plugin.
if __name__ == "__main__":
    plugin = MFS_Plugin()
    plugin.__init__()
