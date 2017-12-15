from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
import math

class RoboschoolReacher(RoboschoolMujocoXmlEnv):
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)

    def set_targets_color(self,colors):
        self.colors = colors

    def set_goals(self,goals):
        self.goals = goals

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.21
    def robot_specific_reset(self):
        def _set_targets_loc():
            for i in range(4):
                r = self.np_random.uniform(low=self.TARG_LIMIT/1.5,high=self.TARG_LIMIT)
                th = self.np_random.uniform(low=0,high=2*3.1415)
                self.jdict["target_%d_x"%i].reset_current_position( r*math.cos(th), 0)
                self.jdict["target_%d_y"%i].reset_current_position( r*math.sin(th), 0)

        _set_targets_loc()
        for i,color in zip(range(4),self.colors):
            self.parts["target%d"%i].set_multiply_color('#ffffffff',color[0] << 16 | color[1] << 8 | color[2])

        self.current_goals = list(self.goals)
        self.wait_counter = 0

        self.fingertip = self.parts["fingertip"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint   = self.jdict["joint1"]
        self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.elbow_joint.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target = self.parts["target%d"%self.current_goals[0]]
        target_x, _ = self.jdict["target_%d_x"%self.current_goals[0]].current_position()
        target_y, _ = self.jdict["target_%d_y"%self.current_goals[0]].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
            ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def _step(self, a):
        assert(not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.calc_potential()

        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]))                                # stall torque require some energy
            )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]

        # record  current subtask
        if( self.wait_counter > 0 ):
            subtask = -1
        else:
            subtask = self.current_goals[0]

        # Change goals
        if( self.wait_counter > 0 ):
            self.wait_counter-=1
            if( self.wait_counter == 0):
                self.current_goals.pop(0)
        else:
            if( self.wait_counter == 0 and
                (np.linalg.norm(self.to_target_vec) <= 0.02) ):
                self.wait_counter = 30


        self.frame  += 1
        self.done   += len(self.current_goals) == 0
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        return state, sum(self.rewards), len(self.current_goals) == 0, {'subtask':subtask}

    def camera_adjust(self):
        #x, y, z = self.fingertip.pose().xyz()
        #x *= 0.5
        #y *= 0.5
        #self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
        self.camera.move_and_look_at(0.0, 0.0, 0.3, 0.0, 0.0, 0.01)
