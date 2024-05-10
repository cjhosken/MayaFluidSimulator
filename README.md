
# Maya Fluid Simulator User Manual

## 1. Installation

Before running the tool, you must install Numpy into your Maya distribution. Numpy is recommended for its significant speed advantages over regular Python lists (Verma, 2020). Follow the instructions below to install Numpy:

### Windows:
```bash
"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy
```

### Linux:
```bash
/usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy
```

### Mac:
```bash
/Applications/Autodesk/maya2023/Maya.app/Contents/bin/mayapy -m pip install –-user numpy
```

*Note: Ensure you adjust the paths in the installation commands to match your Maya version.*

Once Numpy is installed, launch Maya and load the script. A panel titled “Maya Fluid Simulator” should appear on the Maya top bar.

## 2. Usage

To use the tool, follow these steps:

1. Open the user interface by selecting "Maya Fluid Simulator" -> "Open Maya Fluid Simulator".
2. You'll see several options and buttons appear:

### Initialize
- **Particle Size (0.1):** Diameter of particles.
- **Cell Size (0.25):** Size of cells for hashmap searches and velocity transfer.
- **Random Sampling (0):** Number of particles to randomly sample inside the source object cells.
- **Domain Size (5, 5, 5):** Size of the simulation domain.
- **Keep Domain (True):** Keep the domain when re-initializing points.
- **Initialize | x:** Create particles in the source object and create a domain object | delete all the generated artifacts.

### Simulate
- **Force (0, -9.8, 0):** Global force acting on the particles.
- **Initial Velocity (0, 0, 0):** Initial particle velocity.
- **Pressure (0.1):** Pressure divergence.
- **Overrelaxation (0.02):** Velocity divergence.
- **Iterations (5):** Number of iterations for solving the divergence.
- **PIC / FLIP Mix (0.6):** Blending between PIC (smooth) / FLIP (splashy).
- **Frame Range (0, 120):** Start and end frames for the simulation.
- **Time Scale (0.1):** Speed of the simulation.
- **Simulate | x:** Simulate the fluid | clear the simulation.

## 3 Conclusion
This tool was written for L4 Technical Arts Production for Computer Animation Technical Arts at Bournemouth University. There is a very high chance that the algorithms implemented are not entirely correct, therefore be weary when re-using the code. If you wish to access (or correct) the source code, you can do so [here](https://github.com/cjhosken/MayaFluidSimulator).