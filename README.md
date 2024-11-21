# ME 133a Final Project: **Robot with a Lightsaber**

## Group Name: **ERM**  
**Members**: Eric, Rex, Mark

Welcome to the final project for ME 133a at Caltech! Our project, **Robot with a Lightsaber**, showcases an advanced robotic system capable of reflecting laser blasters using a lightsaber. This project demonstrates cutting-edge control, simulation, and robotic manipulation technologies.

---

## **Directory Structure**

```
├── code/           # ROS 2 nodes and supporting scripts
├── launch/         # Launch files for starting the project
├── meshes/         # 3D models and meshes for the urdf
├── resource/       # Auxiliary resources
├── rviz/           # RViz configuration files
├── test/           # Formatting and Copyright Test Files
├── urdf/           # Robot URDF file (iiwa7.urdf)
├── package.xml     # ROS 2 package metadata
├── setup.cfg       # Configuration for setup
├── setup.py        # Python setup script
```

---

## **Getting Started**

### **Prerequisites**

Ensure you have the following installed:

- **ROS 2 (Humble or newer)**
- Python 3.10+
- `colcon` build tools
- RViz 2 for visualization

### **Build Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/ERM-Caltech/me133a-final-project.git
   cd me133a-final-project
   ```

2. Build the project:
   ```bash
   colcon build --symlink-install
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

---

## **Running the Project**

1. **Launch the main project**:
   ```bash
   ros2 launch me133a-final-project main.launch.py
   ```

2. **Visualize in RViz**:
   - The RViz configuration (`rviz/config.rviz`) will automatically load.
   - Ensure RViz is properly installed to view the simulation.

---

## **Robot Details**

- **Robot Model**: [iiwa7](https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa)
- **URDF File**: `urdf/iiwa7.urdf`
- **RViz Config**: `rviz/config.rviz`

---

## **Contributors**

This project was developed by **Eric, Rex, and Mark** as part of ME 133a at Caltech.
