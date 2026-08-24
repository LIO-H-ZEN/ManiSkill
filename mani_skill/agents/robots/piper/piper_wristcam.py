import numpy as np

from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .piper import Piper


PIPER_CAMERA_INTRINSIC = np.array(
    [
        [112.0, 0.0, 112.0],
        [0.0, 112.0, 112.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def build_piper_wrist_camera_config(mount) -> CameraConfig:
    pose = sapien_utils.look_at(
        eye=[0.045, 0.0, 0.045],
        target=[0.0, 0.0, 0.180],
        up=[0.0, 1.0, 0.0],
    )
    return CameraConfig(
        uid="wrist_camera",
        pose=pose,
        width=224,
        height=224,
        fov=None,
        near=0.01,
        far=100,
        intrinsic=PIPER_CAMERA_INTRINSIC.copy(),
        mount=mount,
    )


@register_agent()
class PiperWristCam(Piper):
    """PIPER with the simulation-only LiftCube wrist camera."""

    uid = "piper_wristcam"

    @property
    def _sensor_configs(self):
        return [
            build_piper_wrist_camera_config(mount=self.robot.links_map["gripper_base"])
        ]
