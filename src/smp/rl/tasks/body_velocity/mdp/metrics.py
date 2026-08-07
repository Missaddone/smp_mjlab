"""Experiment 15 per-step metrics exposed through mjlab's metrics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smp.rl.tasks.body_velocity.mdp.diagnostics import env_action_diagnostic_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _values(env: ManagerBasedRlEnv):
  return env_action_diagnostic_values(env)


def exp15_action_delta_rms(env: ManagerBasedRlEnv):
  return _values(env)["action_delta_rms"]


def exp15_action_delta_peak(env: ManagerBasedRlEnv):
  return _values(env)["action_delta_peak"]


def exp15_action_accel_proxy_rms(env: ManagerBasedRlEnv):
  return _values(env)["action_accel_proxy_rms"]


def exp15_action_accel_proxy_peak(env: ManagerBasedRlEnv):
  return _values(env)["action_accel_proxy_peak"]


def exp15_joint_vel_rms(env: ManagerBasedRlEnv):
  return _values(env)["joint_vel_rms"]


def exp15_joint_vel_peak(env: ManagerBasedRlEnv):
  return _values(env)["joint_vel_peak"]


def exp15_leg_action_delta_rms(env: ManagerBasedRlEnv):
  return _values(env)["leg_action_delta_rms"]


def exp15_leg_action_delta_peak(env: ManagerBasedRlEnv):
  return _values(env)["leg_action_delta_peak"]


def exp15_leg_action_accel_proxy_rms(env: ManagerBasedRlEnv):
  return _values(env)["leg_action_accel_proxy_rms"]


def exp15_leg_action_accel_proxy_peak(env: ManagerBasedRlEnv):
  return _values(env)["leg_action_accel_proxy_peak"]


def exp15_leg_action_sign_flip_rate(env: ManagerBasedRlEnv):
  return _values(env)["leg_action_sign_flip_rate"]


def exp15_leg_joint_vel_rms(env: ManagerBasedRlEnv):
  return _values(env)["leg_joint_vel_rms"]


def exp15_leg_joint_vel_peak(env: ManagerBasedRlEnv):
  return _values(env)["leg_joint_vel_peak"]


def exp15_foot_action_delta_rms(env: ManagerBasedRlEnv):
  return _values(env)["foot_action_delta_rms"]


def exp15_foot_action_delta_peak(env: ManagerBasedRlEnv):
  return _values(env)["foot_action_delta_peak"]


def exp15_foot_action_accel_proxy_rms(env: ManagerBasedRlEnv):
  return _values(env)["foot_action_accel_proxy_rms"]


def exp15_foot_action_accel_proxy_peak(env: ManagerBasedRlEnv):
  return _values(env)["foot_action_accel_proxy_peak"]


def exp15_foot_action_sign_flip_rate(env: ManagerBasedRlEnv):
  return _values(env)["foot_action_sign_flip_rate"]


def exp15_foot_joint_vel_rms(env: ManagerBasedRlEnv):
  return _values(env)["foot_joint_vel_rms"]


def exp15_foot_joint_vel_peak(env: ManagerBasedRlEnv):
  return _values(env)["foot_joint_vel_peak"]
