LeKiwi Mobile Base Setup and Control
====================================

This doc walks you through setting up a LeKiwi omnidirectional mobile base
and its front webcam, running the RLinf hardware checks, and bringing up the
``LeKiwiDriveEnv-v1`` environment.

Overview
--------

The base is three Feetech STS3215 wheel servos at 120 degree spacing, driven
over one serial bus, with a plain UVC webcam as its front camera. RLinf
integrates it at two layers: the robotics layer composes ``LeKiwiRobot``
(base plus cameras), and the env layer drives it through
``LeKiwiDriveEnv-v1``. There is no supported training workflow yet; this
guide covers hardware checks and the drive environment.

An action is the body velocity ``[x, y, theta]`` in m/s, m/s, and deg/s,
clipped to the configured limits. A step holds the command for one control
period and then stops the wheels, so the base only moves while the
controller keeps stepping.

Hardware Setup
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Base
     - LeKiwi mobile base with its three wheel servos, servo controller, and
       the power supply specified for the kit.
   * - Front camera
     - UVC webcam connected to the controller machine.
   * - Controller machine
     - Linux computer with a USB connection to the servo controller.

Installation
------------

Install the robot dependencies on the controller machine using the
custom-environment workflow below. Run the checks from the same checkout and
activated environment.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~

Clone RLinf on the machine connected to the base, then install its
environment.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the robot dependencies from the repository root:

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env lekiwi
   source .venv/bin/activate
   export REPO_PATH="$PWD"

The installer includes LeRobot with Feetech support and OpenCV. Run
subsequent commands from this checkout with the environment activated. See
:doc:`Installation </rst_source/start/installation>` for platform
prerequisites.

Set the Wheel Motor IDs
~~~~~~~~~~~~~~~~~~~~~~~

Connect each wheel motor individually, in the order the script prompts, and
assign its ID: left 7, back 8, right 9. Edit the port at the top of the
script with the device your controller reports, then run it from
``toolkits/lekiwi``:

.. code-block:: bash

   cd toolkits/lekiwi
   python setup_base.py

Skip this for a base whose motors are already configured. Find the port by
unplugging and replugging the controller while watching ``ls /dev/tty*``.

Check the Wheels and the Camera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With all three wheels connected, test them from ``toolkits/lekiwi``. Edit
the port at the top of ``test_base.py`` first:

.. code-block:: bash

   python test_base.py

To drive the base from the keyboard, use the teleoperation script, which
takes the port as an argument. For the first movement test, raise the wheels
off the ground and keep servo power within reach:

.. code-block:: bash

   python teleoperate_base.py --port /dev/ttyACM0

See ``toolkits/lekiwi/README.md`` for the key mapping and the dry-run mode.

The front-camera check composes the robot through RLinf's robotics layer:
it discovers the webcam, connects the base, and reads one frame. Edit the
two ports at the top of the script, then run:

.. code-block:: bash

   python test_lekiwi_base_webcam.py

The Drive Environment
---------------------

``LeKiwiDriveEnv-v1`` is the registered Gymnasium task: it composes the
robot from the hardware config, reports the measured body velocity and the
camera frames, and commands bounded velocities. Select it through the env
eval config, the same way any real-world task is selected:

.. code-block:: yaml

   env:
     eval:
       main_image_key: front
       init_params:
         id: LeKiwiDriveEnv-v1
       override_cfg:
         camera_names:
           "<camera-serial>": front
         max_linear_velocity: 0.5
         max_angular_velocity: 90.0
         step_frequency: 10.0

Name the camera with ``camera_names`` so its frames appear under ``front``
(the name ``main_image_key`` points at); an unnamed camera reports under
``camera_1`` and so on. The reward is always zero and the episode never
terminates early: a task decides success itself, as
:doc:`New Real-World Tasks </rst_source/extending/new_task>` describes.

To use the base in your own program, see
:doc:`Robotics Interface </rst_source/concepts/robotics>`.
