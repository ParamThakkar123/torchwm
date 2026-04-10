Environment Configuration
=======================

The simulator is configured through a Python dictionary with three main sections: ``physics``, ``generator``, and ``camera``.

Basic Configuration Structure
---------------------------

.. code-block:: python

    config = {
        "physics": {...},
        "generator": {...},
        "camera": {...},
    }

Physics Configuration
--------------------

Controls the physics simulation parameters.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``timestep``
     - float
     - 1/60
     - Physics time step in seconds
   * - ``substeps``
     - int
     - 1
     - Number of internal steps per external step
   * - ``num_solver_iterations``
     - int
     - 50
     - Solver iterations for constraint solving
   * - ``gravity_z``
     - float
     - -9.81
     - Gravity in z-direction (m/s²)

Example:

.. code-block:: python

    "physics": {
        "timestep": 1/120,        # Higher precision
        "substeps": 2,            # More stable physics
        "num_solver_iterations": 100,  # Better accuracy
        "gravity_z": -9.81,
    },

Generator Configuration
----------------------

Defines objects to spawn in the environment.

Object Types
~~~~~~~~~~~~

**Box:**

.. code-block:: python

    {
        "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},  # x, y, z dimensions
        "position": [0, 0, 1],  # x, y, z position
        "mass": 1.0,
    }

**Sphere:**

.. code-block:: python

    {
        "shape": {"type": "sphere", "radius": 0.5},
        "position": [0, 0, 1],
        "mass": 1.0,
    }

**URDF (robots):**

.. code-block:: python

    {
        "urdf": "path/to/robot.urdf",
        "position": [0, 0, 0],
        "fixed_base": True,  # or False for mobile robot
    }

Object Properties
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``shape``
     - dict
     - Shape specification (box/sphere)
   * - ``position``
     - list
     - [x, y, z] position
   * - ``position_mean``
     - list
     - Center position for random placement
   * - ``position_jitter``
     - list
     - Random offset per spawn [x, y, z]
   * - ``random_orientation``
     - bool
     - Random rotation at spawn
   * - ``mass``
     - float
     - Object mass in kg
   * - ``dynamics``
     - dict
     - Physical properties (friction, etc.)

Random Placement Example:

.. code-block:: python

    "generator": {
        "objects": [
            {
                "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                "position_mean": [0, 0, 1],
                "position_jitter": [0.3, 0.3, 0],
                "random_orientation": True,
                "mass": 1.0,
                "dynamics": {"lateralFriction": 0.5},
            },
        ]
    }

Dynamics Properties
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``lateralFriction``
     - Friction coefficient (0-1)
   * - ``restitution``
     - Bounciness (0-1)

Camera Configuration
--------------------

Controls the rendered camera view.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``width``
     - int
     - 64
     - Image width in pixels
   * - ``height``
     - int
     - 64
     - Image height in pixels
   * - ``fov``
     - float
     - 60
     - Field of view in degrees
   * - ``near``
     - float
     - 0.01
     - Near clipping plane (meters)
   * - ``far``
     - float
     - 100
     - Far clipping plane (meters)
   * - ``position``
     - list
     - [1, 1, 1]
     - Camera position [x, y, z]
   * - ``target``
     - list
     - [0, 0, 0]
     - Camera look-at point [x, y, z]
   * - ``jitter``
     - dict
     - {}
     - Camera pose noise (see below)
   * - ``noise``
     - dict
     - {}
     - Pixel noise (see below)

Camera Jitter Example:

.. code-block:: python

    "camera": {
        "width": 128,
        "height": 128,
        "fov": 60,
        "position": [2, 2, 2],
        "target": [0, 0, 0],
        "jitter": {
            "pos": 0.01,   # Position std dev in meters
            "target": 0.02,  # Look-at std dev
        },
        "noise": {
            "rgb_std": 5.0,  # Gaussian noise std dev
        },
    },

Complete Example
---------------

.. code-block:: python

    config = {
        "physics": {
            "timestep": 1/60,
            "substeps": 1,
            "num_solver_iterations": 50,
            "gravity_z": -9.81,
        },
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position": [0, 0, 1],
                    "mass": 1.0,
                    "dynamics": {"lateralFriction": 0.5},
                },
                {
                    "shape": {"type": "sphere", "radius": 0.3},
                    "position": [0.5, 0, 0.5],
                    "mass": 0.5,
                },
            ]
        },
        "camera": {
            "width": 128,
            "height": 128,
            "fov": 60,
            "position": [1.5, 1.5, 1.5],
            "target": [0, 0, 0],
        },
    }
