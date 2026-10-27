# LTR M1 — verified facts (from vendor SDK + model files, 2026-10-19)

Source: ~/Desktop/yfw-workshop/{LTR_SDK-develop.zip, M1-robotdog-asm09.urdf, M1-robotdog-asm09.xml, M1-robotdog-asm09.zip}.
These override any web-search guesses about the robot's identity.

## Embodiment (from MJCF/URDF)
- Wheel-legged quadruped: 4 legs x (HAA abduction, HFE hip, KNEE, WHEEL) = 16 actuated joints. Each foot is a continuously rotating wheel, radius 0.095 m.
- Mass ~47 kg (trunk 13.9 kg + 4 x 8.27 kg legs). Base height at nominal stand 0.54 m (top of body ~0.68 m). Hip spacing 0.57 m front-to-rear, thigh 0.238 m, shank-to-wheel-center 0.365 m. Front/rear knees bend in opposite directions.
- Torque limits: abad/hip/wheel 40 Nm, knee 100 Nm. MJCF uses torque motors, timestep 1 ms, implicitfast integrator, elliptic cones.
- Sensors in model: IMU only (gyro, quaternion, accelerometer). No camera or LiDAR in URDF/MJCF.
- scene.xml: Perlin heightfield + 274 box geoms (rough terrain / obstacle course) — ready for MuJoCo sim2sim.

## SDK (LTR_SDK-develop)
- C++17, CycloneDDS 0.10.2, prebuilt static libltr_sdk.a for x86_64 and aarch64; Ubuntu 20.04+, GCC 9.4+. No Python bindings, no ROS package.
- Architecture is a near-copy of Unitree SDK2: ChannelFactory/ChannelPublisher/Subscriber, RPC over "rt/api/<service>", lease, MotionSwitcher, SportClient with API IDs 1001-1025 named like Unitree B2's (Damp, BalanceStand, StopMove, StandUp, StandDown, RecoveryStand, Euler, Move, SwitchGait, BodyHeight, SpeedLevel, TrajectoryFollow, ContinuousGait, MoveToPos, SwitchMoveMode, VisionWalk, HandStand, AutoRecoverySet, FreeWalk, ClassicWalk, FastWalk).
- DOCUMENTED and used by the only example (keyboard teleop over Ethernet, DDS domain 0):
  - topic rt/ltr/sportmode_cmd (SportModeCmd_: device_id=1, mode: 0 PASSIVE, 1 DAMPING, 2 SITDOWN, 3 STANDUP, 4 NORMAL_WALK, 5 LION_DANCE, 6 LION_DANCE_OFF)
  - topic rt/ltr/vel_cmd (VelCmd_: lin_vx, lin_vy, lin_vz, ang_wx, ang_wy, ang_wz; example uses |v| = 0.5)
  - Prerequisite: G20 remote selects onboard "RL Loco Classic" locomotion, HOME -> HIGHLEVEL mode. Locomotion itself is an onboard RL policy.
- DECLARED BUT UNDOCUMENTED (firmware support unknown): the SportClient RPCs above (BodyHeight, Euler, MoveToPos, TrajectoryFollow, SwitchGait...), LowCmd_/LowState_ (20 motor slots with q/dq/tau/kp/kd, foot_force, IMU, BMS). Low-level topic names are not in the library.
- DDS type names follow ROS 2 IDL conventions (ltr::msg::dds_::VelCmd_, ltr_m1::msg::dds_::LowState_) with the "rt/" prefix, so ROS 2 (rmw_cyclonedds_cpp) interop like unitree_ros2 is likely possible but untested.

## Implications for the paper
- Real-robot experiments should assume only: velocity commands + mode switching + IMU/joint state, plus an ADD-ON camera (e.g., RGB-D) and offboard compute (RTX 5060 workstation over network, or a laptop/mini-PC carried by the robot — 47 kg platform has payload headroom).
- Do not make the paper depend on BodyHeight/Euler/low-level joint control or on custom locomotion running on hardware until tested.
- The wheel-legged embodiment (fast efficient rolling on flat ground + stepping capability) is a differentiator vs. the Unitree Go1/Go2 legged robots most embodied-AI papers use.
- Locomotion training in simulation is possible with the provided MJCF (MuJoCo / MJX / Isaac Lab via URDF), but hardware deployment is blocked on undocumented low-level access and a 47 kg robot needs safety precautions.
