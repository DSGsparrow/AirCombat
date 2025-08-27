import os
import argparse
import numpy as np
import torch
import gymnasium as gym  # 新版 gymnasium
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn

from env_factory.env_factory_selfplay import make_env, make_normal_env

# === 你已有的：parse_args / make_normal_env 等 ===
# from your_code import parse_args, make_normal_env

MANEUVER_MODEL_PATH = "trained_model/shoot_static3/final_model.zip"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchySelfPlayShoot")
    parser.add_argument("--target_state", type=int, default=0)

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_static_solo_gap.log")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_static_solo_gap/checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_sp_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_static_solo_gap")
    parser.add_argument("--model_dir", type=str, default="./model_pool/shoot_static")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back_t2/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")
    parser.add_argument("--dodge_model_path", type=str, default="trained_model/dodge_missile/ppo_air_combat_dodge4.zip")

    # 环境参数
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--raw_obs_dim", type=int, default=21)
    parser.add_argument("--fly_act_dim", type=int, default=3)
    parser.add_argument("--fire_act_dim", type=int, default=2)
    parser.add_argument("--warmup_action", nargs='+', type=float, default=[1, 2, 1, 0.0, 0.0])

    # 多线程
    parser.add_argument("--num_envs", type=int, default=16)

    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=8_000_000)
    parser.add_argument("--save_interval", type=int, default=20_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # 3e-4
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.15)
    parser.add_argument("--ent_coef", type=float, default=0.02)

    # Transformer 网络参数
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()



class HybridActionWrapper(gym.Wrapper):
    """
    只训练“发射布尔值”，前三维机动动作恒为 0。
    新增功能：决策间隔（action repeat / frame skip）
      - 外部智能体每次决策后，环境内部自动执行 decision_interval 步
      - 本 wrapper 会累计这几步的 reward 并返回最后一步的 obs / done / info

    参数
    ----
    decision_interval : int >= 1
        间隔步数（例如 5 表示每次决策后自动走 5 步）
    edge_fire : bool
        True  ：仅在间隔内的第一步将 fire=1 传入底层，其余步强制 fire=0（避免多次发射）
        False ：在整个间隔内重复相同的 fire 值（可能导致多次发射）
    """

    def __init__(self, env, decision_interval: int = 1, edge_fire: bool = True):
        super().__init__(env)
        assert isinstance(decision_interval, int) and decision_interval >= 1, \
            f"decision_interval 必须是 >=1 的整数，给定 {decision_interval}"
        self.decision_interval = decision_interval
        self.edge_fire = bool(edge_fire)

        # 对外：观测不变，只训练二元发射
        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(2)  # 0=不发射, 1=发射

        # 底层动作空间信息（≥4维：前三维机动，最后一维 fire）
        self._is_multidiscrete = isinstance(env.action_space, spaces.MultiDiscrete)
        self._is_box = isinstance(env.action_space, spaces.Box)

        if self._is_multidiscrete:
            nvec = env.action_space.nvec
            assert len(nvec) >= 4, f"期望底层动作≥4维，现在是 {len(nvec)}"
        elif self._is_box:
            assert env.action_space.shape[0] >= 4, \
                f"期望底层连续动作≥4维，现在是 {env.action_space.shape}"
        else:
            raise TypeError("底层动作空间既不是 MultiDiscrete 也不是 Box。")

        self._last_obs = None

    # 运行时动态切换
    def set_decision_interval(self, n: int):
        assert isinstance(n, int) and n >= 1, f"decision_interval 必须 >=1，给定 {n}"
        self.decision_interval = n

    def set_edge_fire(self, flag: bool):
        self.edge_fire = bool(flag)

    # ---------- 兼容 gym / gymnasium ----------
    @staticmethod
    def _is_gymnasium_step_tuple(t):
        # gymnasium: (obs, reward, terminated, truncated, info)
        return isinstance(t, tuple) and len(t) == 5

    # ---------- 构造底层动作（前三维 0，最后一维为 fire） ----------
    def _build_action(self, fire_bool: int):
        fire = int(fire_bool)

        if self._is_multidiscrete:
            out = np.zeros_like(self.env.action_space.nvec, dtype=np.int64)
            out[:3] = 0
            out[3] = fire
            # 合法范围
            out = np.minimum(out, self.env.action_space.nvec - 1)
            out = np.maximum(out, 0)
            return out

        elif self._is_box:
            out = np.zeros((self.env.action_space.shape[0],), dtype=np.float32)
            out[:3] = 0.0
            out[3] = float(fire)
            return np.clip(out, self.env.action_space.low, self.env.action_space.high)

        else:
            raise RuntimeError("未知的动作空间类型。")

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:  # gymnasium
            obs, info = ret
            self._last_obs = obs
            return obs, info
        else:  # gym
            self._last_obs = ret
            return ret

    def step(self, action):
        # 解析外部二元动作
        if isinstance(action, (np.ndarray, list, tuple)):
            fire_bool = int(np.asarray(action).reshape(-1)[0])
        else:
            fire_bool = int(action)

        total_reward = 0.0
        any_launch = False
        last_info = {}
        terminated = False
        truncated = False
        done = False
        obs = self._last_obs

        # 在间隔内执行多步
        for k in range(self.decision_interval):
            # edge_fire=True 时，仅第一步传 fire_bool，其余步强制 0
            this_fire = fire_bool if (k == 0 or not self.edge_fire) else 0

            full_action = self._build_action(this_fire)
            step_out = self.env.step(full_action)

            if self._is_gymnasium_step_tuple(step_out):
                obs, r, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, r, done, info = step_out
                # 为统一，下两行提供占位
                terminated, truncated = done, False

            self._last_obs = obs
            total_reward += float(r)
            last_info = info if isinstance(info, dict) else {}
            any_launch = any_launch or bool(last_info.get("launch", False))

            if done:  # 若提前结束，跳出
                break

        # 补充一些统计信息（可选）
        if isinstance(last_info, dict):
            last_info = dict(last_info)
            last_info["decision_interval"] = self.decision_interval
            last_info["interval_steps_executed"] = k + 1  # 实际执行了几步
            last_info["interval_reward_sum"] = total_reward
            last_info["interval_any_launch"] = any_launch
        else:
            last_info = {
                "decision_interval": self.decision_interval,
                "interval_steps_executed": k + 1,
                "interval_reward_sum": total_reward,
                "interval_any_launch": any_launch,
            }

        # 返回最后一步的 obs / flags，以及累计奖励
        if self._is_gymnasium_step_tuple(step_out):
            return obs, total_reward, terminated, truncated, last_info
        else:
            return obs, total_reward, done, last_info
# class HybridActionWrapper(gym.Wrapper):
#     """
#     只训练“发射布尔值”，前三维机动动作恒为 0。
#     若 action=1 且底层返回 info['launch']=True：
#       - rollout_after_launch=True: 在内部 roll out 到回合结束，累计奖励一次性返回
#       - rollout_after_launch=False: 不跳过，按普通一步返回
#     """
#
#     def __init__(self, env, rollout_after_launch: bool = True):
#         super().__init__(env)
#         self.rollout_after_launch = rollout_after_launch
#
#         # 对外：观测不变，只训练二元发射
#         self.observation_space = env.observation_space
#         self.action_space = spaces.Discrete(2)  # 0=不发射, 1=发射
#
#         # 底层动作空间信息
#         self._is_multidiscrete = isinstance(env.action_space, spaces.MultiDiscrete)
#         self._is_box = isinstance(env.action_space, spaces.Box)
#
#         if self._is_multidiscrete:
#             nvec = env.action_space.nvec
#             assert len(nvec) >= 4, f"期望底层动作≥4维，现在是 {len(nvec)}"
#         elif self._is_box:
#             assert env.action_space.shape[0] >= 4, \
#                 f"期望底层连续动作≥4维，现在是 {env.action_space.shape}"
#         else:
#             raise TypeError("底层动作空间既不是 MultiDiscrete 也不是 Box。")
#
#         self._last_obs = None
#
#     # 运行时动态切换
#     def set_rollout_after_launch(self, flag: bool):
#         self.rollout_after_launch = bool(flag)
#
#     # ---------- 构造底层动作（前三维 0，最后一维为 fire） ----------
#     def _build_action(self, fire_bool: int):
#         fire = int(fire_bool)
#
#         if self._is_multidiscrete:
#             out = np.zeros_like(self.env.action_space.nvec, dtype=np.int64)
#             out[:3] = 0
#             out[3] = fire
#             # 合法范围
#             out = np.minimum(out, self.env.action_space.nvec - 1)
#             out = np.maximum(out, 0)
#             return out
#
#         elif self._is_box:
#             out = np.zeros((self.env.action_space.shape[0],), dtype=np.float32)
#             out[:3] = 0.0
#             out[3] = float(fire)
#             return np.clip(out, self.env.action_space.low, self.env.action_space.high)
#
#         else:
#             raise RuntimeError("未知的动作空间类型。")
#
#     # ---------- 兼容 gym / gymnasium ----------
#     @staticmethod
#     def _is_gymnasium_step_tuple(t):
#         # gymnasium: (obs, reward, terminated, truncated, info)
#         return isinstance(t, tuple) and len(t) == 5
#
#     def reset(self, **kwargs):
#         ret = self.env.reset(**kwargs)
#         if isinstance(ret, tuple) and len(ret) == 2:
#             obs, info = ret
#             self._last_obs = obs
#             return obs, info
#         else:
#             self._last_obs = ret
#             return ret
#
#     def step(self, action):
#         # 上层只传 0/1
#         if isinstance(action, (np.ndarray, list, tuple)):
#             fire_bool = int(np.asarray(action).reshape(-1)[0])
#         else:
#             fire_bool = int(action)
#
#         # 第一步：按上层选择是否发射
#         full_action = self._build_action(fire_bool)
#         step_out = self.env.step(full_action)
#
#         if self._is_gymnasium_step_tuple(step_out):
#             obs, reward, terminated, truncated, info = step_out
#             self._last_obs = obs
#             launched = bool(info.get("launch", False))
#             done = terminated or truncated
#         else:
#             obs, reward, done, info = step_out
#             self._last_obs = obs
#             launched = bool(info.get("launch", False))
#             terminated, truncated = done, False  # 统一用
#
#         # 未发射：直接返回本步
#         if not launched:
#             return (obs, reward, terminated, truncated, info) if self._is_gymnasium_step_tuple(step_out) \
#                    else (obs, reward, done, info)
#
#         # 发射了，但选择“不跳过”：返回本步（保持与底层一致）
#         if not self.rollout_after_launch:
#             return (obs, reward, terminated, truncated, info) if self._is_gymnasium_step_tuple(step_out) \
#                    else (obs, reward, done, info)
#
#         # 发射且“跳过后续”：内部 roll out 到回合结束并累加奖励
#         cum_reward = float(reward)
#         rolled_steps = 0
#
#         while True:
#             if self._is_gymnasium_step_tuple(step_out):
#                 if terminated or truncated:
#                     break
#             else:
#                 if done:
#                     break
#
#             # 后续固定 fire=0，前三维仍为 0
#             full_action = self._build_action(0)
#             step_out = self.env.step(full_action)
#
#             if self._is_gymnasium_step_tuple(step_out):
#                 obs, r, terminated, truncated, info = step_out
#                 self._last_obs = obs
#                 cum_reward += float(r)
#                 rolled_steps += 1
#             else:
#                 obs, r, done, info = step_out
#                 self._last_obs = obs
#                 cum_reward += float(r)
#                 rolled_steps += 1
#                 terminated, truncated = done, False  # 仅为统一标记
#
#         # 在最后一步的 info 里补充一些记录（只在“跳过”模式下添加）
#         info = dict(info) if isinstance(info, dict) else {}
#         info.setdefault("launched", True)
#         info["rolled_out_steps"] = rolled_steps
#         info["cum_reward_after_launch"] = cum_reward
#
#         # 返回“终止时刻”的 obs / flags，以及累计奖励
#         if self._is_gymnasium_step_tuple(step_out):
#             return obs, cum_reward, terminated, truncated, info
#         else:
#             return obs, cum_reward, True, info




def make_wrapped_env(env_id, args):
    """工厂函数：创建单个底层 env 并包上 HybridActionWrapper。"""
    base = make_normal_env(env_id, args)  # 你原来的环境创建
    wrapped = HybridActionWrapper(base, decision_interval=5, edge_fire=True)
    return wrapped


def main_shoot_static():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # === 修改处：用包装后的工厂函数，仍然支持多进程 SubprocVecEnv ===
    env_fns = [lambda env_id=i: make_wrapped_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # Checkpoint 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval,
        save_path=args.checkpoint_path,
        name_prefix="ppo_model"
    )

    # ✅ 只训练“发射布尔值”的 PPO，因此 policy 仍用 MLP，但动作空间来自 wrapper 的 Discrete(2)
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        activation_fn=nn.ReLU
    )

    # 加载或新建模型（这里指“发射决策”模型）
    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            args.model_path,
            env=env,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，创建新 PPO 模型（仅训练发射布尔值）")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs
        )

    # 训练（其余与过去一致）
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )

    # 最后保存一次
    final_model_path = os.path.join(args.save_model_path, "final_model.zip")
    model.save(final_model_path)
    print(f"✅ 最终模型已保存到 {final_model_path}")


if __name__ == "__main__":
    main_shoot_static()