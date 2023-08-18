import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ray.tune.result import DEFAULT_METRIC, TRAINING_ITERATION
from ray.tune.search.sample import (
    Categorical,
    Domain,
    Float,
    Integer,
    LogUniform,
    Quantized,
    Uniform,
)
from ray.tune.search import (
    UNRESOLVED_SEARCH_SPACE,
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
    Searcher,
)
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart

try:
  import optuna as ot
  from optuna.distributions import BaseDistribution as OptunaDistribution
  from optuna.samplers import BaseSampler
  from optuna.trial import TrialState as OptunaTrialState
  from optuna.trial import Trial as OptunaTrial
except ImportError:
  ot = None
  OptunaDistribution = None
  BaseSampler = None
  OptunaTrialState = None
  OptunaTrial = None

logger = logging.getLogger(__name__)

# print a warning if define by run function takes longer than this to execute
DEFINE_BY_RUN_WARN_THRESHOLD_S = 1  # 1 is arbitrary


class _OptunaTrialSuggestCaptor:
  """Utility to capture returned values from Optuna's suggest_ methods.

    This will wrap around the ``optuna.Trial` object and decorate all
    `suggest_` callables with a function capturing the returned value,
    which will be saved in the ``captured_values`` dict.
    """

  def __init__(self, ot_trial: OptunaTrial) -> None:
    self.ot_trial = ot_trial
    self.captured_values: Dict[str, Any] = {}

  def _get_wrapper(self, func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # name is always the first arg for suggest_ methods
      name = kwargs.get("name", args[0])
      ret = func(*args, **kwargs)
      self.captured_values[name] = ret
      return ret

    return wrapper

  def __getattr__(self, item_name: str) -> Any:
    item = getattr(self.ot_trial, item_name)
    if item_name.startswith("suggest_") and callable(item):
      return self._get_wrapper(item)
    return item


class OptunaSearch(Searcher):

  def __init__(self,
               space: Optional[Union[
                   Dict[str, "OptunaDistribution"],
                   List[Tuple],
                   Callable[["OptunaTrial"], Optional[Dict[str, Any]]],
               ]] = None,
               metric: Optional[Union[str, List[str]]] = None,
               mode: Optional[Union[str, List[str]]] = None,
               points_to_evaluate: Optional[List[Dict]] = None,
               sampler: Optional["BaseSampler"] = None,
               seed: Optional[int] = None,
               evaluated_rewards: Optional[List] = None,
               study_name='optuna',
               storage=None):
    assert ot is not None, "Optuna must be installed! Run `pip install optuna`."
    super().__init__(metric=metric, mode=mode)

    if isinstance(space, dict) and space:
      resolved_vars, domain_vars, grid_vars = parse_spec_vars(space)
      if domain_vars or grid_vars:
        logger.warning(
            UNRESOLVED_SEARCH_SPACE.format(
                par="space", cls=type(self).__name__))
        space = self.convert_search_space(space)
      else:
        # Flatten to support nested dicts
        space = flatten_dict(space, "/")

    self._space = space

    self._points_to_evaluate = points_to_evaluate or []
    self._evaluated_rewards = evaluated_rewards

    self._study_name = study_name  # Fixed study name for in-memory storage

    if sampler and seed:
      logger.warning(
          "You passed an initialized sampler to `MyOptuna`. The "
          "`seed` parameter has to be passed to the sampler directly "
          "and will be ignored.")
    elif sampler:
      assert isinstance(sampler,
                        BaseSampler), ("You can only pass an instance of "
                                       "`optuna.samplers.BaseSampler` "
                                       "as a sampler to `MyOptunaer`.")

    self._sampler = sampler
    self._seed = seed

    self._completed_trials = set()

    self._ot_trials = {}
    self._ot_study = None
    self.storage = storage
    if self._space:
      self._setup_study(mode)

  def _setup_study(self, mode: Union[str, list]):
    if self._metric is None and self._mode:
      if isinstance(self._mode, list):
        raise ValueError("If ``mode`` is a list (multi-objective optimization "
                         "case), ``metric`` must be defined.")
      # If only a mode was passed, use anonymous metric
      self._metric = DEFAULT_METRIC

    pruner = ot.pruners.NopPruner()
    storage = self.storage if self.storage else ot.storages.InMemoryStorage()

    if self._sampler:
      sampler = self._sampler
    elif isinstance(mode, list) and version.parse(
        ot.__version__) < version.parse("2.9.0"):
      # MOTPESampler deprecated in Optuna>=2.9.0
      sampler = ot.samplers.MOTPESampler(seed=self._seed)
    else:
      sampler = ot.samplers.TPESampler(seed=self._seed)

    if isinstance(mode, list):
      study_direction_args = dict(
          directions=["minimize" if m == "min" else "maximize" for m in mode],
      )
    else:
      study_direction_args = dict(
          direction="minimize" if mode == "min" else "maximize",
      )

    self._ot_study = ot.study.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=self._study_name,
        load_if_exists=True,
        **study_direction_args,
    )

    if self._points_to_evaluate:
      validate_warmstart(
          self._space,
          self._points_to_evaluate,
          self._evaluated_rewards,
          validate_point_name_lengths=not callable(self._space),
      )
      if self._evaluated_rewards:
        for point, reward in zip(self._points_to_evaluate,
                                 self._evaluated_rewards):
          self.add_evaluated_point(point, reward)
      else:
        for point in self._points_to_evaluate:
          self._ot_study.enqueue_trial(point)

  def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                            config: Dict, **spec) -> bool:
    if self._space:
      return False
    space = self.convert_search_space(config)
    self._space = space
    if metric:
      self._metric = metric
    if mode:
      self._mode = mode

    self._setup_study(self._mode)
    return True

  def _suggest_from_define_by_run_func(
      self,
      func: Callable[["OptunaTrial"], Optional[Dict[str, Any]]],
      ot_trial: "OptunaTrial",
  ) -> Dict:
    captor = _OptunaTrialSuggestCaptor(ot_trial)
    time_start = time.time()
    ret = func(captor)
    time_taken = time.time() - time_start
    if time_taken > DEFINE_BY_RUN_WARN_THRESHOLD_S:
      warnings.warn("Define-by-run function passed in the `space` argument "
                    f"took {time_taken} seconds to "
                    "run. Ensure that actual computation, training takes "
                    "place inside Tune's train functions or Trainables "
                    "passed to `tune.Tuner()`.")
    if ret is not None:
      if not isinstance(ret, dict):
        raise TypeError("The return value of the define-by-run function "
                        "passed in the `space` argument should be "
                        "either None or a `dict` with `str` keys. "
                        f"Got {type(ret)}.")
      if not all(isinstance(k, str) for k in ret.keys()):
        raise TypeError("At least one of the keys in the dict returned by the "
                        "define-by-run function passed in the `space` argument "
                        "was not a `str`.")
    return {**captor.captured_values, **ret} if ret else captor.captured_values

  def suggest(self, trial_id: str) -> Optional[Dict]:
    if not self._space:
      raise RuntimeError(
          UNDEFINED_SEARCH_SPACE.format(
              cls=self.__class__.__name__, space="space"))
    if not self._metric or not self._mode:
      raise RuntimeError(
          UNDEFINED_METRIC_MODE.format(
              cls=self.__class__.__name__, metric=self._metric,
              mode=self._mode))
    if callable(self._space):
      # Define-by-run case
      if trial_id not in self._ot_trials:
        self._ot_trials[trial_id] = self._ot_study.ask()

      ot_trial = self._ot_trials[trial_id]

      params = self._suggest_from_define_by_run_func(self._space, ot_trial)
    else:
      # Use Optuna ask interface (since version 2.6.0)
      if trial_id not in self._ot_trials:
        self._ot_trials[trial_id] = self._ot_study.ask(
            fixed_distributions=self._space)
      ot_trial = self._ot_trials[trial_id]
      params = ot_trial.params

    return unflatten_dict(params)

  def on_trial_result(self, trial_id: str, result: Dict):
    if isinstance(self.metric, list):
      # Optuna doesn't support incremental results
      # for multi-objective optimization
      return
    if trial_id in self._completed_trials:
      logger.warning(f"Received additional result for trial {trial_id}, but "
                     f"it already finished. Result: {result}")
      return
    metric = result[self.metric]
    step = result[TRAINING_ITERATION]
    ot_trial = self._ot_trials[trial_id]
    ot_trial.report(metric, step)

  def on_trial_complete(self,
                        trial_id: str,
                        result: Optional[Dict] = None,
                        error: bool = False):
    if trial_id in self._completed_trials:
      logger.warning(
          f"Received additional completion for trial {trial_id}, but "
          f"it already finished. Result: {result}")
      return

    ot_trial = self._ot_trials[trial_id]

    if result:
      if isinstance(self.metric, list):
        val = [result.get(metric, None) for metric in self.metric]
      else:
        val = result.get(self.metric, None)
    else:
      val = None
    ot_trial_state = OptunaTrialState.COMPLETE
    if val is None:
      if error:
        ot_trial_state = OptunaTrialState.FAIL
      else:
        ot_trial_state = OptunaTrialState.PRUNED
    try:
      self._ot_study.tell(ot_trial, val, state=ot_trial_state)
    except Exception as exc:
      logger.warning(exc)  # E.g. if NaN was reported

    self._completed_trials.add(trial_id)

  def add_evaluated_point(
      self,
      parameters: Dict,
      value: float,
      error: bool = False,
      pruned: bool = False,
      intermediate_values: Optional[List[float]] = None,
  ):
    if not self._space:
      raise RuntimeError(
          UNDEFINED_SEARCH_SPACE.format(
              cls=self.__class__.__name__, space="space"))
    if not self._metric or not self._mode:
      raise RuntimeError(
          UNDEFINED_METRIC_MODE.format(
              cls=self.__class__.__name__, metric=self._metric,
              mode=self._mode))
    if callable(self._space):
      raise TypeError(
          "Define-by-run function passed in `space` argument is not "
          "yet supported when using `evaluated_rewards`. Please provide "
          "an `OptunaDistribution` dict or pass a Ray Tune "
          "search space to `tune.Tuner()`.")

    ot_trial_state = OptunaTrialState.COMPLETE
    if error:
      ot_trial_state = OptunaTrialState.FAIL
    elif pruned:
      ot_trial_state = OptunaTrialState.PRUNED

    if intermediate_values:
      intermediate_values_dict = {
          i: value for i, value in enumerate(intermediate_values)
      }
    else:
      intermediate_values_dict = None

    trial = ot.trial.create_trial(
        state=ot_trial_state,
        value=value,
        params=parameters,
        distributions=self._space,
        intermediate_values=intermediate_values_dict,
    )

    self._ot_study.add_trial(trial)

  def save(self, checkpoint_path: str):
    save_object = self.__dict__.copy()
    with open(checkpoint_path, "wb") as outputFile:
      pickle.dump(save_object, outputFile)

  def restore(self, checkpoint_path: str):
    with open(checkpoint_path, "rb") as inputFile:
      save_object = pickle.load(inputFile)
    if isinstance(save_object, dict):
      self.__dict__.update(save_object)
    else:
      # Backwards compatibility
      (
          self._sampler,
          self._ot_trials,
          self._ot_study,
          self._points_to_evaluate,
          self._evaluated_rewards,
      ) = save_object

  @staticmethod
  def convert_search_space(spec: Dict) -> Dict[str, Any]:
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

    if not domain_vars and not grid_vars:
      return {}

    if grid_vars:
      raise ValueError(
          "Grid search parameters cannot be automatically converted "
          "to an Optuna search space.")

    # Flatten and resolve again after checking for grid search.
    spec = flatten_dict(spec, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

    def resolve_value(domain: Domain) -> ot.distributions.BaseDistribution:
      quantize = None

      sampler = domain.get_sampler()
      if isinstance(sampler, Quantized):
        quantize = sampler.q
        sampler = sampler.sampler
        if isinstance(sampler, LogUniform):
          logger.warning("Optuna does not handle quantization in loguniform "
                         "sampling. The parameter will be passed but it will "
                         "probably be ignored.")

      if isinstance(domain, Float):
        if isinstance(sampler, LogUniform):
          if quantize:
            logger.warning("Optuna does not support both quantization and "
                           "sampling from LogUniform. Dropped quantization.")
          return ot.distributions.LogUniformDistribution(
              domain.lower, domain.upper)

        elif isinstance(sampler, Uniform):
          if quantize:
            return ot.distributions.DiscreteUniformDistribution(
                domain.lower, domain.upper, quantize)
          return ot.distributions.UniformDistribution(domain.lower,
                                                      domain.upper)

      elif isinstance(domain, Integer):
        if isinstance(sampler, LogUniform):
          return ot.distributions.IntLogUniformDistribution(
              domain.lower, domain.upper - 1, step=quantize or 1)
        elif isinstance(sampler, Uniform):
          # Upper bound should be inclusive for quantization and
          # exclusive otherwise
          return ot.distributions.IntUniformDistribution(
              domain.lower,
              domain.upper - int(bool(not quantize)),
              step=quantize or 1,
          )
      elif isinstance(domain, Categorical):
        if isinstance(sampler, Uniform):
          return ot.distributions.CategoricalDistribution(domain.categories)

      raise ValueError("Optuna search does not support parameters of type "
                       "`{}` with samplers of type `{}`".format(
                           type(domain).__name__,
                           type(domain.sampler).__name__))

    # Parameter name is e.g. "a/b/c" for nested dicts
    values = {
        "/".join(path): resolve_value(domain) for path, domain in domain_vars
    }

    return values
