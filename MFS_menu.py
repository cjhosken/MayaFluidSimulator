from maya import cmds

def create_custom_menu():
    if cmds.menu("MFS_menu", exists=True):
        cmds.deleteUI("MFS_menu", menu=True)

    MFS_menu = cmds.menu("MFS_menu", label="Maya Fluid Simulator", parent="MayaWindow", tearOff=False)

    # TODO: HAVE A 'CREATE' LABEL ABOVE THE OPTIONS
    # TODO: ICONS
    # TODO: CHANGE COMMANDS
    # TODO: REFACTOR TO FIT CODING CONVENTIONS

    cmds.menuItem(label="Domain", command="print('domain applied')")
    cmds.menuItem(label="Solver", command="print('solver applied')")