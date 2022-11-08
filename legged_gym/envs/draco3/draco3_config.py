# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Draco3Cfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env):
        num_envs = 2
        num_observations = 280
        num_actions = 27

    class init_state( LeggedRobotCfg.init_state ):
        hip_yaw_angle = 5
        pos = [0.0, 0.0, 2.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "l_shoulder_aa": np.pi / 6,
            "l_elbow_fe": -np.pi / 2,
            "r_shoulder_aa": -np.pi / 6,
            "r_elbow_fe": -np.pi / 2,
            "l_hip_aa": np.radians(hip_yaw_angle),
            "l_hip_fe": -np.pi / 4,
            "l_knee_fe_jp": np.pi / 4,
            "l_knee_fe_jd": np.pi / 4,
            "l_ankle_fe": -np.pi / 4,
            "l_ankle_ie": np.radians(-hip_yaw_angle),
            "r_hip_aa": np.radians(-hip_yaw_angle),
            "r_hip_fe": -np.pi / 4,
            "r_knee_fe_jp": np.pi / 4,
            "r_knee_fe_jd": np.pi / 4,
            "r_ankle_fe": -np.pi / 4,
            "r_ankle_ie": np.radians(hip_yaw_angle),

            # All joints not explicit in Draco3 startup are set to zero
            "l_hip_ie": 0.,
            "l_shoulder_fe": 0.,
            "l_shoulder_ie": 0.,
            "l_wrist_ps": 0.,
            "l_wrist_pitch": 0.,
            "r_hip_ie": 0.,
            "r_shoulder_fe": 0.,
            "r_shoulder_ie": 0.,
            "r_wrist_ps": 0.,
            "r_wrist_pitch": 0.,
            "neck_pitch": 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {
            "neck_pitch":       40.0,
            "l_shoulder_fe":    100.0,
            "l_shoulder_aa":    100.0,
            "l_shoulder_ie":    100.0,
            "l_elbow_fe":       50.0,
            "l_wrist_ps":       40.0,
            "l_wrist_pitch":    50.0,
            "l_knee_fe_jp":     75.0,
            "l_knee_fe_jd":     75.0,            
            "l_hip_aa":         100.0,
            "l_hip_fe":         100.0,
            "l_hip_ie":         100.0,            
            "l_ankle_fe":       50.0,
            "l_ankle_ie":       50.0,         
            "r_shoulder_fe":    100.0,
            "r_shoulder_aa":    100.0,
            "r_shoulder_ie":    100.0,
            "r_elbow_fe":       50.0,
            "r_wrist_ps":       40.0,
            "r_wrist_pitch":    50.0,
            "r_knee_fe_jp":     75.0,
            "r_knee_fe_jd":     75.0,            
            "r_hip_aa":         100.0,
            "r_hip_fe":         100.0,
            "r_hip_ie":         100.0,            
            "r_ankle_fe":       50.0,
            "r_ankle_ie":       50.0,            
        }
        damping = {
            "neck_pitch":       2.0,
            
            "l_shoulder_fe":    8.0,
            "l_shoulder_aa":    8.0,
            "l_shoulder_ie":    8.0,
            "l_elbow_fe":       3.0,
            "l_wrist_ps":       2.0,
            "l_wrist_pitch":    3.0,
            "l_knee_fe_jp":     5.0,
            "l_knee_fe_jd":     5.0,            
            "l_hip_aa":         8.0,
            "l_hip_fe":         8.0,
            "l_hip_ie":         8.0,            
            "l_ankle_fe":       3.0,
            "l_ankle_ie":       3.0,         
            
            "r_shoulder_fe":    8.0,
            "r_shoulder_aa":    8.0,
            "r_shoulder_ie":    8.0,
            "r_elbow_fe":       3.0,
            "r_wrist_ps":       2.0,
            "r_wrist_pitch":    3.0,
            "r_knee_fe_jp":     5.0,
            "r_knee_fe_jd":     5.0,            
            "r_hip_aa":         8.0,
            "r_hip_fe":         8.0,
            "r_hip_ie":         8.0,            
            "r_ankle_fe":       3.0,
            "r_ankle_ie":       3.0,             
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/draco3/draco3.urdf'
        name = "draco3"
        foot_name = 'foot_contact'
        terminate_after_contacts_on = [
            'torso_link',
            ]
        fix_base_link = True # fixe the base of the robot
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class rewards( LeggedRobotCfg.rewards ):
        class scales ( LeggedRobotCfg.rewards.scales ):
            pass

class Draco3CfgPPO(LeggedRobotCfgPPO):
    class runner (LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'draco3'
        load_run = -1