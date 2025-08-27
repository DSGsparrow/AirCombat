import numpy as np
from gymnasium import spaces
from collections import deque
import logging
from typing import List, Tuple

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, PostureShootReward
from ..reward_functions import ShootPenaltyReward, ShootGapPenaltyReward, RelativeAltitudeReward, ShootEventDrivenReward
from ..reward_functions import ShootEnemyPostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, DodgeMissileSafeReturn, ShootWrong
from ..termination_conditions import SafeReturn, ShootSafeReturn
from ..reward_functions import EndAltitudeReward, ShootWaitReward
from ..reward_functions import (SelfPlayShootPenaltyReward, SelfPlayPostureReward, SelfPlayShootEventDrivenReward,
                                SelfPlayShootGapPenalty, SelfPlayShootPosturePenalty,
                                SelfPlayShootWaitReward, SelfPlayEnemyPostureReward, SelfPlayShootMissileRewardWithDistance)
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R
from .singlecombat_with_missle_task import SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, SingleCombatShootMissileTask
from utils.shoot_rule import fuzzy_should_attack

class SingleCombatShootMissileBackTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureShootReward(self.config),
            AltitudeReward(self.config),
            ShootEventDrivenReward(self.config),
            ShootPenaltyReward(self.config),
            ShootGapPenaltyReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            ShootSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.4, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 0.9, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：DodgeMissileAgent 调用的自己SB3上训练的PPO
            norm_action = self.baseline_agent.get_action(env, agent_id)
            # np.ndarray(4,)
            # norm_action = self.baseline_agent.normalize_action(env, agent_id, action)
            # norm_action = action[:-1]
            # 4位杆量，可以直接传
            # self._shoot_action[agent_id] = action[-1]
            # 不能起效

            return norm_action

        else:
            norm_action = self.baseline_agent.get_action(env, agent_id)
            # norm_action = np.zeros(4)
            # norm_action[0] = action[0]
            # norm_action[1] = action[1]
            # norm_action[2] = action[2]
            # norm_action[3] = action[3]
            self._shoot_action[agent_id] = 1 if action[-1] > 0.5 else 0

            return norm_action
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            obs_list = obs.tolist()
            state = self.get_states(env, agent_id)
            state_list = state.tolist()

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                logging.info(f'{agent_id} launch mission! '
                             f'Total Steps={env.current_step}, obs={obs_list}, state={state_list}, '
                             f'current_reward={env.cumulative_reward}')


class HierarchicalSingleCombatShootMissileSoloTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            # SelfPlayShootPenaltyReward(self.config),  # 打弹就有惩罚，且引诱对方打弹有奖励，零和
            # SelfPlayPostureReward(self.config),
            # SelfPlayShootEventDrivenReward(self.config),
            # SelfPlayShootGapPenalty(self.config),  # 打弹间隔
            # SelfPlayShootPosturePenalty(self.config),  # 打弹时姿势
            SelfPlayShootWaitReward(self.config),  # 等待奖励
            # SelfPlayEnemyPostureReward(self.config),  # 敌方躲弹
            # AltitudeReward(self.config),  # 防坠地
            SelfPlayShootMissileRewardWithDistance(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            ShootSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

        # return HierarchicalSingleCombatTask.load_action_space(self)
        return self.action_space

    def get_obs(self, env, agent_id):
        """
                Convert simulation states into the format of observation_space

                ------
                Returns: (np.ndarray)
                - ego info
                    - [0] ego altitude           (unit: 5km)
                    - [1] ego_roll_sin
                    - [2] ego_roll_cos
                    - [3] ego_pitch_sin
                    - [4] ego_pitch_cos
                    - [5] ego v_body_x           (unit: mh)
                    - [6] ego v_body_y           (unit: mh)
                    - [7] ego v_body_z           (unit: mh)
                    - [8] ego_vc                 (unit: mh)
                - relative enm info
                    - [9] delta_v_body_x         (unit: mh)
                    - [10] delta_altitude        (unit: km)
                    - [11] ego_AO                (unit: rad) [0, pi]
                    - [12] ego_TA                (unit: rad) [0, pi]
                    - [13] relative distance     (unit: 10km)
                    - [14] side_flag             1 or 0 or -1
                - relative missile info
                    - [15] delta_v_body_x
                    - [16] delta altitude
                    - [17] ego_AO
                    - [18] ego_TA
                    - [19] relative distance
                    - [20] side flag
                """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_launching()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(enm_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        else:
            pass
        return norm_obs

    def get_states(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_states(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
                """
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：DodgeMissileAgent 调用的自己SB3上训练的PPO
            norm_action = self.baseline_agent.get_action(env.agents[agent_id])

            return norm_action

        else:
            norm_action = self.baseline_agent.get_action(env.agents[agent_id])
            # norm_action = np.zeros(4)
            # norm_action[0] = action[0]
            # norm_action[1] = action[1]
            # norm_action[2] = action[2]
            # norm_action[3] = action[3]
            self._shoot_action[agent_id] = 1 if action[-1] > 0.5 else 0

            return norm_action

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}

        self.lock_duration_num = {agent_id: 0 for agent_id in env.agents.keys()}
        self.lose_lock_duration_num = {agent_id: 0 for agent_id in env.agents.keys()}

        return SingleCombatDodgeMissileTask.reset(self, env)

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        total_reward = 0.0

        # 确保初始化过（防止外部提前调用未初始化）
        if not hasattr(self, 'cumulative_rewards'):
            self.cumulative_rewards = {
                aid: [0.0 for _ in range(len(self.reward_functions))]
                for aid in env.agents.keys()
            }
            self.wait_reward_list = []

        for i, reward_function in enumerate(self.reward_functions):
            r = reward_function.get_reward(self, env, agent_id)
            total_reward += r
            self.cumulative_rewards[agent_id][i] += r
            if i == 0:
                self.wait_reward_list.append(r)

        return total_reward, info

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            state = self.get_states(env, agent_id)
            self.launch[agent_id] = False

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self.launch[agent_id] = True
                logging.info(f'{agent_id} launch mission! distance={obs[13] * 10000}'
                             f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')
                             # f'Total Steps={env.current_step}, current_reward={env.cumulative_reward}')

    # def step(self, env):
    #     SingleCombatTask.step(self, env)
    #     for agent_id, agent in env.agents.items():
    #         # [RL-based missile launch with limited condition]
    #         shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
    #
    #         obs = self.get_obs(env, agent_id)
    #         state = self.get_states(env, agent_id)
    #
    #         target = agent.enemies[0].get_position() - agent.get_position()
    #         heading = agent.get_velocity()
    #         distance = np.linalg.norm(target)
    #         attack_angle = np.rad2deg(
    #             np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
    #
    #         if attack_angle < self.max_attack_angle:
    #             # 超出视线角开始计时
    #             self.lock_duration_num[agent_id] += 1
    #         else:
    #             self.lock_duration_num[agent_id] = 0
    #             if self.remaining_missiles['A0100'] == 0 and agent_id == 'A0100':
    #                 # A已经发弹
    #                 self.lose_lock_duration_num['B0100'] += 1
    #
    #         if shoot_flag:
    #             new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
    #             env.add_temp_simulator(
    #                 MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
    #             self.remaining_missiles[agent_id] -= 1
    #             logging.info(f'{agent_id} launch mission! '
    #                          f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')