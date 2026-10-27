"""DDS message types for the LTR M1, mirrored from the vendor SDK (LTR_SDK-develop).

The vendor C++ headers register these as final (XTypes IS_FINAL) structs named
"ltr::msg::dds_::VelCmd_" / "ltr::msg::dds_::SportModeCmd_" with sequential member ids,
so the Python definitions must keep the same names, member order and types to match on the wire.

Note: no `from __future__ import annotations` here; cyclonedds reads the annotations at class creation.
"""
from dataclasses import dataclass

from cyclonedds.idl import IdlStruct, annotations as annotate
from cyclonedds.idl import types

TOPIC_VELCMD = "rt/ltr/vel_cmd"
TOPIC_SPORTMODECMD = "rt/ltr/sportmode_cmd"

# SportModeCmd_.device_id: the vendor example always sends 1 (HIGHLEVELSDK); 0 is the gamepad.
DEVICE_ID_HIGHLEVELSDK = 1

# SportModeCmd_.mode values (from m1_dds_command_example.cpp)
MODE_PASSIVE = 0
MODE_DAMPING = 1
MODE_SITDOWN = 2
MODE_STANDUP = 3
MODE_NORMAL_WALK = 4
MODE_LION_DANCE = 5
MODE_LION_DANCE_OFF = 6


@dataclass
@annotate.final
@annotate.autoid("sequential")
class VelCmd_(IdlStruct, typename="ltr.msg.dds_.VelCmd_"):
    lin_vx: types.float32 = 0.0
    lin_vy: types.float32 = 0.0
    lin_vz: types.float32 = 0.0
    ang_wx: types.float32 = 0.0
    ang_wy: types.float32 = 0.0
    ang_wz: types.float32 = 0.0


@dataclass
@annotate.final
@annotate.autoid("sequential")
class SportModeCmd_(IdlStruct, typename="ltr.msg.dds_.SportModeCmd_"):
    device_id: types.uint8 = DEVICE_ID_HIGHLEVELSDK
    mode: types.uint8 = 0
