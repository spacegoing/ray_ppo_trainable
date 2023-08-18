import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger.aim import AimLoggerCallback

import os
from datetime import datetime
import optuna
from myoptuna import OptunaSearch

from utils_ray import load_config, get_reporter

ppo_config = load_config(path='./ts_ppo_config.yaml')


def conditonal_search_space(trial):
  # no inter-step changes
  trial.suggest_categorical("num_envs_per_worker", list(range(1, 16)))

  # inter-step tuned:
  # make_env:
  trial.suggest_categorical("obs_norm", [True, False])
  # step: onpolicytrainer, rollout related
  trial.suggest_int("repeat_per_collect", 5, 20)
  trial.suggest_int("minibatch_size", 64, 2048, step=128)
  min_len = trial.params["minibatch_size"] * trial.params["repeat_per_collect"]
  trial.suggest_int(
      "step_per_collect", max(min_len, 1024), max(min_len, 30000), step=1000)
  # reset_config: policy related
  trial.suggest_float("lr", 1e-5, 1e-2, log=True)
  trial.suggest_float("gamma", 0.1, 0.9999)
  trial.suggest_float("gae_lambda", 0.1, 0.9999)
  trial.suggest_float("vf_coef", 0.01, 1.0)
  trial.suggest_float("ent_coef", 1e-8, 1e-1)
  trial.suggest_categorical("rew_norm", [True, False])
  trial.suggest_categorical("norm_adv", [True, False])


keys = [
    'num_envs_per_worker', 'repeat_per_collect', 'minibatch_size',
    'step_per_collect', 'lr', 'gamma', 'gae_lambda', 'vf_coef', 'ent_coef',
    'obs_norm', 'rew_norm', 'norm_adv'
]
for key in keys:
  if key in ppo_config:
    del ppo_config[key]


def main():
  ray.init(
      # so docker can listen to host port
      dashboard_host=ppo_config.dashboard_host,
      # redirect worker logs to driver stdout/err
      log_to_driver=True,
      logging_level=ppo_config.log_level)

  timelog = (
      str(datetime.date(datetime.now())) + "_" +
      str(datetime.time(datetime.now())))
  study_name = "Optuna_multi-v={}_group={}_{}".format(ppo_config.multivariate,
                                                      ppo_config.group, timelog)
  db_path = "./log/optuna_db/"
  os.makedirs(db_path, exist_ok=True)
  storage = "sqlite:///{}/{}.db".format(db_path, study_name)
  print("Inspecting with: optuna-dashboard", storage)
  import subprocess
  subprocess.Popen(
      "optuna-dashboard " + storage,
      shell=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL)

  # reporter
  reporter = get_reporter(
      parameter_columns={
          "num_envs_per_worker": "num_env",
          "step_per_collect": "steps/roll",
          "repeat_per_collect": "repeat/roll",
          "minibatch_size": "mbsz",
          "gae_lambda": "gae",
          "vf_coef": "vf",
          "ent_coef": "ent",
      },
      metric_columns={'reward_mean': 'reward'},
      max_report_frequency=ppo_config.max_report_frequency)

  # checkpoint config
  ckpt_config = air.CheckpointConfig(
      checkpoint_score_attribute='checkpoint_score',
      checkpoint_score_order='max',
      num_to_keep=ppo_config.saving_top_k,
      checkpoint_frequency=0,  # must be 0 for ray_ppo save top_k manually
      # For Trainable Subclass, Default is True in
      # tuner.fit()->...->tuner_internal.py:_get_tune_run_arguments()
      checkpoint_at_end=False,
      _checkpoint_keep_all_ranks=True,
  )

  # run
  from ppo.ray_ppo import Policy
  tuner = tune.Tuner(
      tune.with_resources(
          Policy,
          resources={
              'cpu': ppo_config.num_cpus,
              'gpu': ppo_config.num_gpus
          }),
      tune_config=tune.TuneConfig(
          metric="reward_mean",
          mode="max",
          search_alg=OptunaSearch(
              space=conditonal_search_space,
              metric="reward_mean",
              study_name=study_name,
              storage=storage,
              mode="max",
              sampler=optuna.samplers.TPESampler(
                  group=ppo_config.group,
                  multivariate=ppo_config.multivariate)),
          scheduler=ASHAScheduler(
              grace_period=ppo_config.asha_burn_in_time_attr,
              time_attr="training_iteration"),
          num_samples=-1,
          max_concurrent_trials=ppo_config.max_concurrent_trials,
      ),
      param_space=ppo_config,
      run_config=air.RunConfig(
          name=study_name,
          storage_path='~/ray_results',
          checkpoint_config=ckpt_config,
          callbacks=[AimLoggerCallback()],
          progress_reporter=reporter),
  )

  tuner.fit()
  ray.shutdown()


if __name__ == "__main__":
  main()
