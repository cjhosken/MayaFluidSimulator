from maya import cmds

def create_custom_menu():
    # Check if the menu already exists, and delete it if it does
    if cmds.menu("customMenu", exists=True):
        cmds.deleteUI("customMenu", menu=True)

    # Create the menu
    custom_menu = cmds.menu("customMenu", label="Custom Menu", parent="MayaWindow", tearOff=True)

    # Add items to the menu
    cmds.menuItem(label="Item 1", command="print('Item 1 selected')")
    cmds.menuItem(label="Item 2", command="print('Item 2 selected')")
    cmds.menuItem(divider=True)
    cmds.menuItem(label="Item 3", command="print('Item 3 selected')")