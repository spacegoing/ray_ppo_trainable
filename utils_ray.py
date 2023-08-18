import yaml
from ray.tune import CLIReporter
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION, TIME_TOTAL_S


class ConfigDict(dict):

  def __init__(self, config_dict):
    flat_config = self.flat(config_dict)
    super(ConfigDict, self).__init__(flat_config)
    self.__dict__ = self

  def __setattr__(self, __name: str, __value) -> None:
    return super().__setattr__(__name, __value)

  def flat(self, config):
    flat_config = dict()
    for k in config:
      if type(config[k]) == dict:
        # drop first level keys, promote second level keys to first level
        for sk, v in config[k].items():
          flat_config[sk] = v
      else:  # keep first level kv-pair unchanged
        flat_config[k] = config[k]

    return flat_config


def load_config(path="ts_ppo_config.yaml"):

  with open(path, 'r') as f:
    em_config = yaml.safe_load(f)
    debug_config = em_config.pop('debug_config')

  config = ConfigDict(em_config)
  config = merge_debug(debug_config, config)
  return config


def merge_debug(debug_config, config):
  if debug_config['debug_flag']:
    config.update(debug_config)
  return config


def get_reporter(parameter_columns=None,
                 metric_columns=None,
                 max_report_frequency=30,
                 max_progress_rows=40,
                 sort_by_metric=True):

  default_metric = {TRAINING_ITERATION: "iter", TIMESTEPS_TOTAL: "ts"}
  metric_columns.update(default_metric)

  reporter = CLIReporter(
      parameter_columns=parameter_columns,
      metric_columns=metric_columns,
      max_report_frequency=max_report_frequency,
      max_progress_rows=max_progress_rows,
      sort_by_metric=sort_by_metric)
  return reporter
