from typing import Any, Dict, Tuple, Union
import numpy as np
import tqdm
from tianshou.trainer import OnpolicyTrainer as Trainer
from tianshou.trainer.utils import gather_info
from tianshou.utils import (
    DummyTqdm,
    tqdm_config,
)


class OnpolicyTrainer(Trainer):

  def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    self.epoch += 1
    self.iter_num += 1

    if self.iter_num > 1:

      # iterator exhaustion check
      if self.epoch > self.max_epoch:
        raise StopIteration

      # exit flag 1, when stop_fn succeeds in train_step or test_step
      if self.stop_fn_flag:
        raise StopIteration

    # set policy in train mode
    self.policy.train()

    epoch_stat: Dict[str, Any] = dict()

    if self.show_progress:
      progress = tqdm.tqdm
    else:
      progress = DummyTqdm

    # SG: log episodic reward
    epirew_list = []

    # perform n step_per_epoch
    with progress(
        total=self.step_per_epoch, desc=f"Epoch #{self.epoch}",
        **tqdm_config) as t:
      while t.n < t.total and not self.stop_fn_flag:
        data: Dict[str, Any] = dict()
        result: Dict[str, Any] = dict()
        if self.train_collector is not None:
          data, result, self.stop_fn_flag = self.train_step()
          t.update(result["n/st"])
          if self.stop_fn_flag:
            t.set_postfix(**data)
            break
        else:
          assert self.buffer, "No train_collector or buffer specified"
          result["n/ep"] = len(self.buffer)
          result["n/st"] = int(self.gradient_step)
          t.update()

        self.policy_update_fn(data, result)
        t.set_postfix(**data)

        # SG: log episodic reward
        if result['n/ep'] > 0:
          epirew_list.append([result['rew'], result['rew_std']])

      if t.n <= t.total and not self.stop_fn_flag:
        t.update()

    # SG: train episodic reward mean
    self.this_epoch_eprews_mean = np.nan
    self.this_epoch_eprews_std = np.nan
    if epirew_list:
      # for 0 entries, it means in that collection there are none episodes ended
      epirew = np.mean(np.array(epirew_list), axis=0)
      self.this_epoch_eprews_mean = epirew[0]
      self.this_epoch_eprews_std = epirew[1]

    # for offline RL
    if self.train_collector is None:
      self.env_step = self.gradient_step * self.batch_size

    if not self.stop_fn_flag:
      self.logger.save_data(self.epoch, self.env_step, self.gradient_step,
                            self.save_checkpoint_fn)
      # test
      if self.test_collector is not None:
        test_stat, self.stop_fn_flag = self.test_step()
        if not self.is_run:
          epoch_stat.update(test_stat)

    if not self.is_run:
      epoch_stat.update({k: v.get() for k, v in self.stat.items()})
      epoch_stat["gradient_step"] = self.gradient_step
      epoch_stat.update({
          "env_step": self.env_step,
          "rew": self.last_rew,
          "len": int(self.last_len),
          "n/ep": int(result["n/ep"]),
          "n/st": int(result["n/st"]),
      })
      info = gather_info(self.start_time, self.train_collector,
                         self.test_collector, self.best_reward,
                         self.best_reward_std)
      return self.epoch, epoch_stat, info
    else:
      return None
