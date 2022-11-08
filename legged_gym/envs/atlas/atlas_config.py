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

class AtlasCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env):
        num_envs = 2048 # number robots
        num_observations = 289 # num steps
        num_actions = 30 # DOF

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state( LeggedRobotCfg.init_state ):
        hip_yaw_angle = 0.0 #5
        pos = [0.0, 0.0, 0.85] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
        # shoulder_x
        "l_arm_shx": -np.pi / 4,
        "r_arm_shx": np.pi / 4,
        # elbow_y
        "l_arm_ely": np.pi / 2,
        "r_arm_ely": np.pi / 2,
        # elbow_x
        "l_arm_elx": np.pi / 2,
        "r_arm_elx": -np.pi / 2,
        # hip_y
        "l_leg_hpy": -np.pi / 4,
        "r_leg_hpy": -np.pi / 4,
        # knee
        "l_leg_kny": np.pi / 2,
        "r_leg_kny": np.pi / 2,
        # ankle
        "l_leg_aky": -np.pi / 4,
        "r_leg_aky": -np.pi / 4,

        # # shoulder_x
        # "l_arm_shx": 0.0,
        # "r_arm_shx": 0.0,
        # # elbow_y
        # "l_arm_ely": 0.0,
        # "r_arm_ely": 0.0,
        # # elbow_x
        # "l_arm_elx": 0.0,
        # "r_arm_elx": 0.0,
        # # hip_y
        # "l_leg_hpy": 0.0,
        # "r_leg_hpy": 0.0,
        # # knee
        # "l_leg_kny": 0.0,
        # "r_leg_kny": 0.0,
        # # ankle
        # "l_leg_aky": 0.0,
        # "r_leg_aky": 0.0,

        # all joints not explicitly stated in atlas set_initial_config are set to zero
        
        "neck_ry": 0.,
        
        "back_bkx": 0.,
        "back_bky": 0.,
        "back_bkz": 0.,
        "pelvis_com_fixed": 0.,
        
        "l_leg_hpx": 0.,
        "l_leg_hpz": 0.,
        "l_arm_shz": 0.,
        "l_leg_akx": 0.,
        "l_sole_fixed": 0.,
        "l_arm_wrx": 0.,
        "l_arm_wry": 0.,
        "l_arm_wry2": 0.,

        "r_leg_hpx": 0.,
        "r_leg_hpz": 0.,
        "r_arm_shz": 0.,
        "r_leg_akx": 0.,
        "r_sole_fixed": 0.,
        "r_arm_wrx": 0.,
        "r_arm_wry": 0.,
        "r_arm_wry2": 0.,
        # 31 joints
    }

    class control( LeggedRobotCfg.control ):
        control_type = 'T'
        # PD Drive parameters:
        # stiffness = {
        #     "neck_ry":          40.0,
        #     "back_bkx":         100.,
        #     "back_bky":         100.,
        #     "back_bkz":         100.,
        #     "pelvis_com_fixed": 0.,
        #     "l_arm_shx":        100.0,
        #     "l_arm_shz":        100.0,
        #     "l_arm_elx":        50.0,
        #     "l_arm_ely":        50.0,
        #     "l_arm_wrx":        40.0,
        #     "l_arm_wry2":       50.0,
        #     "l_leg_kny":        75.0,  
        #     "l_leg_hpx":        100.0,
        #     "l_leg_hpy":        100.0,
        #     "l_leg_hpz":        100.0, 
        #     "l_leg_akx":        50.0,
        #     "l_leg_aky":        50.0,
        #     "l_sole_fixed":     0.,
        #     "r_arm_shx":        100.0,
        #     "r_arm_shz":        100.0,
        #     "r_arm_elx":        50.0,
        #     "r_arm_ely":        50.0,
        #     "r_arm_wrx":        40.0,
        #     "r_arm_wry2":       50.0,
        #     "r_leg_kny":        75.0, 
        #     "r_leg_hpx":        100.0,
        #     "r_leg_hpy":        100.0,
        #     "r_leg_hpz":        100.0,  
        #     "r_leg_akx":        50.0,
        #     "r_leg_aky":        50.0,
        #     "r_sole_fixed":     0.,
        # }
        # damping = {
        #     "neck_ry":          2.,
        #     "back_bkx":         1.,
        #     "back_bky":         1.,
        #     "back_bkz":         1.,
        #     "pelvis_com_fixed": 0.,
        #     "l_arm_shx":        8.,
        #     "l_arm_shz":        8.,
        #     "l_arm_elx":        3.,
        #     "l_arm_ely":        3.,
        #     "l_arm_wrx":        2.,
        #     "l_arm_wry2":       3.,
        #     "l_leg_kny":        2,  
        #     "l_leg_hpx":        100.0,
        #     "l_leg_hpy":        100.0,
        #     "l_leg_hpz":        100.0, 
        #     "l_leg_akx":        1.,
        #     "l_leg_aky":        1.,
        #     "l_sole_fixed":     0.,
        #     "r_arm_shx":        8.,
        #     "r_arm_shz":        8.,
        #     "r_arm_elx":        3.,
        #     "r_arm_ely":        3.,
        #     "r_arm_wrx":        2.,
        #     "r_arm_wry2":       3.,
        #     "r_leg_kny":        2,
        #     "r_leg_hpx":        100.0,
        #     "r_leg_hpy":        100.0,
        #     "r_leg_hpz":        100.0, 
        #     "r_leg_akx":        1.,
        #     "r_leg_aky":        1.,
        #     "r_sole_fixed":     0.,
        # }
        stiffness = {
            "neck_ry":          0.0,
            "back_bkx":         0.0,
            "back_bky":         0.0,
            "back_bkz":         0.0,
            "pelvis_com_fixed": 0.0,
            "l_arm_shx":        0.0,
            "l_arm_shz":        0.0,
            "l_arm_elx":        0.0,
            "l_arm_ely":        0.0,
            "l_arm_wrx":        0.0,
            "l_arm_wry2":       0.0,
            "l_leg_kny":        0.0,
            "l_leg_hpx":        0.0,
            "l_leg_hpy":        0.0,
            "l_leg_hpz":        0.0,
            "l_leg_akx":        0.0,
            "l_leg_aky":        0.0,
            "l_sole_fixed":     0.0,
            "r_arm_shx":        0.0,
            "r_arm_shz":        0.0,
            "r_arm_elx":        0.0,
            "r_arm_ely":        0.0,
            "r_arm_wrx":        0.0,
            "r_arm_wry2":       0.0,
            "r_leg_kny":        0.0,
            "r_leg_hpx":        0.0,
            "r_leg_hpy":        0.0,
            "r_leg_hpz":        0.0,
            "r_leg_akx":        0.0,
            "r_leg_aky":        0.0,
            "r_sole_fixed":     0.0,
        }
        damping = {
            "neck_ry":          0.0,
            "back_bkx":         0.0,
            "back_bky":         0.0,
            "back_bkz":         0.0,
            "pelvis_com_fixed": 0.0,
            "l_arm_shx":        0.0,
            "l_arm_shz":        0.0,
            "l_arm_elx":        0.0,
            "l_arm_ely":        0.0,
            "l_arm_wrx":        0.0,
            "l_arm_wry2":       0.0,
            "l_leg_kny":        0.0,
            "l_leg_hpx":        0.0,
            "l_leg_hpy":        0.0,
            "l_leg_hpz":        0.0,
            "l_leg_akx":        0.0,
            "l_leg_aky":        0.0,
            "l_sole_fixed":     0.0,
            "r_arm_shx":        0.0,
            "r_arm_shz":        0.0,
            "r_arm_elx":        0.0,
            "r_arm_ely":        0.0,
            "r_arm_wrx":        0.0,
            "r_arm_wry2":       0.0,
            "r_leg_kny":        0.0,
            "r_leg_hpx":        0.0,
            "r_leg_hpy":        0.0,
            "r_leg_hpz":        0.0,
            "r_leg_akx":        0.0,
            "r_leg_aky":        0.0,
            "r_sole_fixed":     0.0,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 50
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-0.1, 0.1]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.001

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/atlas/atlas_reduced_collision.urdf'
        name = "atlas"
        foot_name = 'foot'
        terminate_after_contacts_on = [
            'pelvis',
            "head",
            "pelvis_com",
            "utorso",
            "mtorso",
            "ltorso",
            "r_clav",
            "r_scap",
            # "r_uarm",
            # "r_ufarm",
            # "l_uarm",
            # "l_ufarm",
            "r_uleg",
            "l_clav",
            "l_scap",
            "l_uleg",
        ]

        penalize_contacts_on = [
            "l_uglut",
            "l_lglut",
            "l_lleg",
            "l_talus",
            # "l_larm",
            "r_uglut",
            "r_lglut",
            "r_lleg",
            "r_talus",
            # "r_larm",
            # "r_lfarm",
            # "l_lfarm",
        ]

        fix_base_link = False # fixe the base of the robot
        disable_gravity = False
        
        # replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        
        density = 0.001
        angular_damping = 0.1
        linear_damping = 0.1
        max_angular_velocity = 5.0
        max_linear_velocity = 5.0
        armature = 0.
        thickness = 0.01
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 100
        base_height_target = 0.8
        max_contact_force = 500.      # MAXX Changed for Atlas
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -200.
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 1.0
            # torques = -5.e-6
            # dof_acc = -2.e-7
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # feet_air_time = 0.0
            # dof_pos_limits = -1.
            # no_fly = 0.25
            # dof_vel = -0.0
            # ang_vel_xy = -0.0
            # feet_contact_forces = -0.
            # stand_still = 50.0

            # base_height = 100.0 
            # collision = -1.
            # feet_stumble = -0.0 
            # action_rate = -0.01

            termination = -20.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            torques = 0.0
            dof_acc = 0.0
            lin_vel_z = -1.0
            ang_vel_xy = -5.0
            orientation = -10.0
            feet_air_time = 0.0
            dof_pos_limits = -1.
            no_fly = -0.25
            dof_vel = -0.0
            feet_contact_forces = -0.5
            stand_still = -1.0
            base_height = 0.0
            collision = 0.0
            feet_stumble = 0.0 
            action_rate = 0.0
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = False           # MAXX Changed for Atlas
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class AtlasCfgPPO(LeggedRobotCfgPPO):
    class runner (LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'atlas'

        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 16 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.