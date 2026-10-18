LeKiwi 移动底盘配置与控制
==========================

本文介绍如何配置 LeKiwi 全向移动底盘及其前置摄像头，运行 RLinf 硬件检查，并启动
``LeKiwiDriveEnv-v1`` 环境。

概览
----

底盘由三个互成 120 度的 Feetech STS3215 轮电机组成，通过一条串口总线驱动，前置
一个普通 UVC 摄像头。RLinf 在两个层面集成它：机器人层组装 ``LeKiwiRobot``
（底盘加摄像头），环境层通过 ``LeKiwiDriveEnv-v1`` 驱动。目前尚未提供训练流程；
本指南涵盖硬件检查与驱动环境。

动作为机体坐标系速度 ``[x, y, theta]``，单位分别为 m/s、m/s 和 deg/s，并按配置的
上限裁剪。一个 step 将指令保持一个控制周期后停止车轮，因此只有控制器持续步进时
底盘才会移动。

硬件配置
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 底盘
     - LeKiwi 移动底盘及其三个轮电机、舵机控制板和套件指定的电源。
   * - 前置摄像头
     - 连接在控制主机上的 UVC 摄像头。
   * - 控制主机
     - 通过 USB 连接舵机控制板的 Linux 计算机。

安装
----

在控制主机上按下方自定义环境流程安装机器人依赖。后续检查均在同一仓库目录和已激活
的环境中运行。

机器人控制节点
~~~~~~~~~~~~~~

在连接底盘的主机上克隆 RLinf，然后安装对应环境。

.. include:: _setup_common.rst
   :end-before: 然后，使用

在仓库根目录安装机器人依赖：

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env lekiwi
   source .venv/bin/activate
   export REPO_PATH="$PWD"

安装脚本包含 LeRobot 及 Feetech 支持和 OpenCV。后续命令均在此仓库目录中运行，并
保持环境已激活。平台要求见 :doc:`安装 </rst_source/start/installation>`。

设置轮电机 ID
~~~~~~~~~~~~~

按脚本提示的顺序逐个连接轮电机并分配 ID：左轮 7、后轮 8、右轮 9。先按控制主机
报告的设备修改脚本顶部的串口，然后在 ``toolkits/lekiwi`` 目录下运行：

.. code-block:: bash

   cd toolkits/lekiwi
   python setup_base.py

电机已配置过的底盘可跳过此步。拔插控制板并观察 ``ls /dev/tty*`` 即可找到串口。

检查车轮与摄像头
~~~~~~~~~~~~~~~~

三个车轮全部接好后，在 ``toolkits/lekiwi`` 目录下测试。先修改 ``test_base.py``
顶部的串口：

.. code-block:: bash

   python test_base.py

用键盘遥控底盘时，使用接受串口参数的遥控脚本。首次运动测试请将车轮抬离地面，
并保持舵机电源触手可及：

.. code-block:: bash

   python teleoperate_base.py --port /dev/ttyACM0

按键映射与 dry-run 模式见 ``toolkits/lekiwi/README.md``。

前置摄像头检查通过 RLinf 的机器人层组装整机：自动发现摄像头、连接底盘并读取一帧。
先修改脚本顶部的两个串口，然后运行：

.. code-block:: bash

   python test_lekiwi_base_webcam.py

驱动环境
--------

``LeKiwiDriveEnv-v1`` 是已注册的 Gymnasium 任务：按硬件配置组装整机，上报实测的
机体速度和摄像头画面，并下发有界的速度指令。通过 env eval 配置选择它，方式与其
他真机任务相同：

.. code-block:: yaml

   env:
     eval:
       main_image_key: front
       init_params:
         id: LeKiwiDriveEnv-v1
       override_cfg:
         camera_names:
           "<摄像头序列号>": front
         max_linear_velocity: 0.5
         max_angular_velocity: 90.0
         step_frequency: 10.0

用 ``camera_names`` 为摄像头命名，使其画面以 ``front`` 上报（即 ``main_image_key``
指向的名字）；未命名的摄像头以 ``camera_1`` 等上报。奖励恒为零，回合也不会提前
终止：任务成功与否由任务本身判定，参见
:doc:`新真机任务 </rst_source/extending/new_task>`。

如需在自己的程序中使用底盘，参见
:doc:`机器人接口 </rst_source/concepts/robotics>`。
