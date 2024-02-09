from maya import cmds
from MFS_solverNode import MFS_assign_SolverNode, MFS_simulation_callback
from MFS_fluids import MFS_Solver
import os

def MFS_create_menu():
    script_directory = os.path.dirname(__file__)

    MFS_delete_menu()

    MFS_menu = cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)

    solver_icon = os.path.join(script_directory, "icons", "MFS_icon_solver_512.png")

    cmds.menuItem(label="Create", divider=True)
    cmds.menuItem(label="Solver", image=solver_icon, command=MFS_assign_SolverNode)
    cmds.menuItem(label="Simulate!", image=solver_icon, command=MFS_simulation_callback)

def MFS_delete_menu():
    if cmds.menu("MFS_menu", exists=True):
        cmds.deleteUI("MFS_menu", menu=True)