# 🌊 Maya Fluid Simulator

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)

[![Maya](https://img.shields.io/badge/Autodesk_Maya-%233A95E3.svg?&logo=autodesk&logoColor=white)](#)

**Maya Fluid Simulator (MFS)** is a Python-based implementation of PIC/FLIP in **Autodesk Maya**.

📘 Learn more about the theory and implementation of Maya Fluid Simulator [here](https://cjhosken.github.io/blog/mfs).

## 🧩 1. Installation

Before running the tool, you must install **NumPy** into your Maya distribution.
NumPy provides significant performance advantages over standard Python lists (Verma, 2020).

Follow the instructions below to install NumPy:

### 🪟 Windows

```bash
"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" -m pip install --user numpy
```

### 🐧 Linux

```bash
/usr/autodesk/maya2023/bin/mayapy -m pip install --user numpy
```

### 🍎 Mac

```bash
/Applications/Autodesk/maya2023/Maya.app/Contents/bin/mayapy -m pip install --user numpy
```

> 💡 **Note:** Make sure to adjust the file paths to match your specific Maya version and installation directory.

Once NumPy is installed, launch Maya and load the script.
A new menu titled **“Maya Fluid Simulator”** should appear on the Maya top bar.

---

## 🧠 2. Usage

Follow the steps below to begin using the simulator:

1. Open the user interface:
   **Maya Fluid Simulator → Open Maya Fluid Simulator**
2. A new control panel will appear, containing several options and settings.

---

### ⚙️ Initialize

| Parameter                 | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| **Particle Size (0.1)**   | Diameter of individual particles.                                         |
| **Cell Size (0.25)**      | Size of grid cells used for hashmap searches and velocity transfer.       |
| **Random Sampling (0)**   | Number of particles randomly sampled within source object cells.          |
| **Domain Size (5, 5, 5)** | Defines the overall size of the simulation domain.                        |
| **Keep Domain (True)**    | Retains the domain object when re-initializing points.                    |
| **[Initialize]**        | *Create particles and domain objects* | *Delete all generated artifacts.* |

---

### 💧 Simulate

| Parameter                      | Description                                                     |
| ------------------------------ | --------------------------------------------------------------- |
| **Force (0, -9.8, 0)**         | Global force applied to all particles (e.g., gravity).          |
| **Initial Velocity (0, 0, 0)** | Initial velocity vector of particles.                           |
| **Pressure (0.1)**             | Controls the pressure divergence within the fluid.              |
| **Overrelaxation (0.02)**      | Controls the velocity divergence for stability.                 |
| **Iterations (5)**             | Number of solver iterations per frame.                          |
| **PIC / FLIP Mix (0.6)**       | Blending ratio between PIC (smooth) and FLIP (splashy) methods. |
| **Frame Range (0, 120)**       | Start and end frame range for the simulation.                   |
| **Time Scale (0.1)**           | Adjusts the speed of the simulation.                            |
| **[Simulate]**               | *Run the simulation* | *Clear the simulation results.*          |

---

## 🧾 3. Additional Information

An additional `index.html` file is included, containing a **2D JavaScript implementation** of the fluid simulation.

This tool was developed for by [Christopher Hosken](https://cjhosken.github.io) for **L4 Technical Arts Production**
as part of the **Computer Animation Technical Arts** course at *Bournemouth University*.

> ⚠️ **Disclaimer:**
> The implemented algorithms may not be fully accurate. Use caution when reusing or extending this code.

To access or contribute to the source code, visit the GitHub repository:
🔗 [github.com/cjhosken/MayaFluidSimulator](https://github.com/cjhosken/MayaFluidSimulator)