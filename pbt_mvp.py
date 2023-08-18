import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.sample import Domain
from ray.tune.logger.aim import AimLoggerCallback

from datetime import datetime
import numpy as np
from utils_ray import load_config, get_reporter

ppo_config = load_config(path='./ts_ppo_config.yaml')

#* tune params
# at init, pbt only resample if params are missing from config,
# so have to submit to tuneconfig one more time
tune_dict = {
    "num_envs_per_worker": tune.choice(list(range(1, 10))),
    "step_per_collect": tune.choice(list(range(2048, 30000, 1000))),
    "repeat_per_collect": tune.choice(list(range(5, 20))),
    "minibatch_size": tune.choice(list(range(64, 2048, 128))),
    "lr": tune.loguniform(1e-5, 1e-2),
    "gamma": tune.uniform(0.1, 0.9999),
    "gae_lambda": tune.uniform(0.1, 0.9999),
    "vf_coef": tune.uniform(0.01, 1.0),
    "ent_coef": tune.uniform(1e-8, 1e-1),
}
ppo_config.update(tune_dict)

#* pbt params
# for int, pbt use list over tune.choice: pbt.py:_explore() will only
# multiply 1.2 or 0.8 but will use int() to truncate float to integers
# for small value like range(4,8), this makes no sense int(4*1.2)=4
pbt_dict = {
    "num_envs_per_worker": list(range(1, 10)),
    "step_per_collect": tune.choice(list(range(2048, 30000, 1000))),
    "repeat_per_collect": list(range(5, 20)),
    "minibatch_size": tune.choice(list(range(64, 2048, 128))),
    "lr": tune.loguniform(1e-5, 1e-2),
    "gamma": tune.uniform(0.1, 0.9999),
    "gae_lambda": tune.uniform(0.1, 0.9999),
    "vf_coef": tune.uniform(0.01, 1.0),
    "ent_coef": tune.uniform(1e-8, 1e-1),
    # can't pbt: within a max_epoch, previous collect = True, later = False doesn't makesense
    # "obs_norm": [True, False]
    # "rew_norm": [True, False],
    # "norm_adv": [True, False]
}


#* Postprocess the perturbed config to ensure it's still valid
def explore(config):
  # Postprocess the perturbed config to ensure it's still valid
  # ensure we collect enough timesteps to do sgd
  if config["step_per_collect"] < config["minibatch_size"] * config[
      "repeat_per_collect"]:
    config["step_per_collect"] = config["minibatch_size"] * config[
        "repeat_per_collect"]

  config['step_per_epoch'] = max(config['step_per_epoch'],
                                 config['step_per_collect'] * 2)

  config['buffer_size'] = config['step_per_collect']
  for k, v in pbt_dict.items():
    if isinstance(v, Domain):
      if hasattr(v, "lower") and hasattr(v, "upper"):
        config[k] = np.clip(config[k], v.lower, v.upper)

  return config


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

  pbt = PopulationBasedTraining(
      time_attr="training_iteration",
      perturbation_interval=ppo_config.perturbation_interval,
      resample_probability=ppo_config.resample_probability,
      quantile_fraction=ppo_config.quantile_fraction,
      # time_attr must appear in every Trainable.step()'s return
      require_attrs=True,
      # Specifies the mutations of these hyperparams
      hyperparam_mutations=pbt_dict,
      custom_explore_fn=explore,
  )

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
          scheduler=pbt,
          reuse_actors=True,
          num_samples=-1,
          max_concurrent_trials=ppo_config.max_concurrent_trials),
      param_space=ppo_config,
      run_config=air.RunConfig(
          name="pbt_{}".format(timelog),
          storage_path='~/ray_results',
          checkpoint_config=ckpt_config,
          callbacks=[AimLoggerCallback()],
          progress_reporter=reporter),
  )

  tuner.fit()
  ray.shutdown()


if __name__ == "__main__":
  main()
