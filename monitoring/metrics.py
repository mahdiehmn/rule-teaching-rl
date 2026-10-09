"""
Shared metrics and run logging for comparable experiments.

TensorBoard is good for interactive curves, but later comparisons need
plain machine-readable files too. RunTracker writes two files into
each run directory:

- run_summary.json: one evolving/final summary for sweep tables.
- episodes.csv: one row per completed episode for learning curves.
"""

import csv
import json
import os
import platform
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone


def success_rate(outcomes):
    """
    Fraction of successful episodes in outcomes.
    """

    if not outcomes:
        return None
    return float(sum(outcomes) / len(outcomes))


def frames_to_threshold(curve, threshold):
    """
    First frame count where a success-rate curve crosses threshold.

    curve is an iterable of (frame_count, success_rate) pairs. Returns
    None if the threshold is never reached.
    """

    for frame_count, value in curve:
        if value is not None and value >= threshold:
            return int(frame_count)
    return None


def _mean(values):
    """
    Mean as a plain float, or None for an empty sequence.
    """

    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_json_value(value):
    """
    Convert common non-JSON scalar types into plain Python values.
    """

    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    return repr(value)


def _env_metadata(unwrapped_env):
    """
    Pull stable MiniGrid metadata from an unwrapped env when available.
    """

    if unwrapped_env is None:
        return {}

    spec = getattr(unwrapped_env, 'spec', None)
    return {
        'gym_id': getattr(spec, 'id', None),
        'env_class': unwrapped_env.__class__.__name__,
        'width': getattr(unwrapped_env, 'width', None),
        'height': getattr(unwrapped_env, 'height', None),
        'max_steps': getattr(unwrapped_env, 'max_steps', None),
        'agent_view_size': getattr(unwrapped_env, 'agent_view_size', None),
        'see_through_walls': getattr(
            unwrapped_env, 'see_through_walls', None
        ),
    }


class RunTracker:
    """
    Keep comparable run metadata, episode rows, and final summaries.
    """

    def __init__(
        self,
        run_dir,
        run_name,
        algo,
        args,
        obs_shape,
        action_space,
        unwrapped_env=None,
        device=None,
        extra=None,
    ):
        """
        Create the tracker and write the initial running summary.
        """

        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.run_name = run_name
        self.algo = algo
        self.args = {k: _safe_json_value(v) for k, v in vars(args).items()}
        self.obs_shape = list(obs_shape)
        self.action_space = repr(action_space)
        self.device = str(device) if device is not None else None
        self.env = _env_metadata(unwrapped_env)
        self.extra = _safe_json_value(extra or {})
        self.latest = None

        self.started_at = datetime.now(timezone.utc).isoformat()
        self.start_time = time.time()
        self.last_global_step = 0

        self.episode_count = 0
        self.success_count = 0
        self.returns = []
        self.lengths = []
        self.successes = []
        self.recent_returns = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        self.threshold_curve = []
        self.first_thresholds = {
            'frames_to_recent_100_success_0.5': None,
            'frames_to_recent_100_success_0.8': None,
            'frames_to_recent_100_success_0.9': None,
            'wall_time_to_recent_100_success_0.5_sec': None,
            'wall_time_to_recent_100_success_0.8_sec': None,
            'wall_time_to_recent_100_success_0.9_sec': None,
        }

        self.summary_path = os.path.join(run_dir, 'run_summary.json')
        self.episodes_path = os.path.join(run_dir, 'episodes.csv')
        self._episodes_file = open(
            self.episodes_path, 'w', newline='', encoding='utf-8'
        )
        self._episodes_writer = csv.DictWriter(
            self._episodes_file,
            fieldnames=[
                'episode_index',
                'global_step',
                'env_index',
                'return',
                'length',
                'success',
                'wall_time_sec',
                'sps',
            ],
        )
        self._episodes_writer.writeheader()
        self._episodes_file.flush()
        self.write_summary(status='running')

    def elapsed(self):
        """
        Wall-clock seconds since tracker creation.
        """

        return time.time() - self.start_time

    def sps(self, global_step=None):
        """
        Environment frames per wall-clock second.
        """

        step = self.last_global_step if global_step is None else global_step
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        return float(step / elapsed)

    def record_episode(self, global_step, env_index, ep_return, ep_length):
        """
        Append one completed episode to episodes.csv and aggregates.
        """

        self.last_global_step = int(global_step)
        ep_return = float(ep_return)
        ep_length = float(ep_length)
        success = 1.0 if ep_return > 0 else 0.0

        self.episode_count += 1
        self.success_count += int(success)
        self.returns.append(ep_return)
        self.lengths.append(ep_length)
        self.successes.append(success)
        self.recent_returns.append(ep_return)
        self.recent_lengths.append(ep_length)
        self.recent_successes.append(success)

        wall_time = self.elapsed()
        row = {
            'episode_index': self.episode_count,
            'global_step': int(global_step),
            'env_index': int(env_index),
            'return': ep_return,
            'length': ep_length,
            'success': success,
            'wall_time_sec': wall_time,
            'sps': self.sps(global_step),
        }
        self._episodes_writer.writerow(row)
        self._episodes_file.flush()

        if len(self.recent_successes) == 100:
            recent_rate = success_rate(self.recent_successes)
            self.threshold_curve.append((int(global_step), recent_rate))
            for threshold in (0.5, 0.8, 0.9):
                frame_key = f'frames_to_recent_100_success_{threshold}'
                time_key = (
                    f'wall_time_to_recent_100_success_{threshold}_sec'
                )
                if (
                    self.first_thresholds[frame_key] is None
                    and recent_rate >= threshold
                ):
                    self.first_thresholds[frame_key] = int(global_step)
                    self.first_thresholds[time_key] = wall_time

    def summary(self, status='running', global_step=None, extra=None):
        """
        Build the current summary dict.
        """

        if global_step is not None:
            self.last_global_step = int(global_step)
        if extra is not None:
            self.latest = _safe_json_value(extra)

        elapsed = self.elapsed()
        summary = {
            'status': status,
            'run_name': self.run_name,
            'algo': self.algo,
            'started_at_utc': self.started_at,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'wall_time_sec': elapsed,
            'global_step': int(self.last_global_step),
            'sps': self.sps(self.last_global_step),
            'total_episodes': int(self.episode_count),
            'total_successes': int(self.success_count),
            'success_rate_all': success_rate(self.successes),
            'mean_return_all': _mean(self.returns),
            'mean_length_all': _mean(self.lengths),
            'success_rate_recent_100': success_rate(
                self.recent_successes
            ),
            'mean_return_recent_100': _mean(self.recent_returns),
            'mean_length_recent_100': _mean(self.recent_lengths),
            'obs_shape': self.obs_shape,
            'action_space': self.action_space,
            'device': self.device,
            'env': self.env,
            'args': self.args,
            'extra': self.extra,
            'thresholds': self.first_thresholds,
        }
        if self.latest is not None:
            summary['latest'] = self.latest
        return summary

    def write_summary(self, status='running', global_step=None, extra=None,
                      required=False):
        """
        Write run_summary.json atomically enough for live monitoring.

        Set required=True when losing this write would lose the run's
        result; periodic snapshots leave it False.
        """

        data = self.summary(status=status, global_step=global_step, extra=extra)
        tmp_path = self.summary_path + '.tmp'

        # Retry the write-then-rename. On a shared parallel filesystem
        # (Lustre on the clusters) os.replace can fail transiently with
        # FileNotFoundError on the tmp file when the metadata server is
        # loaded -- observed on 3 of 655 concurrent jobs, at unrelated
        # iterations, with a single-threaded writer. Backing off and
        # retrying clears it.
        last_error = None
        for attempt in range(5):
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write('\n')
                os.replace(tmp_path, self.summary_path)
                return data
            except OSError as error:
                last_error = error
                time.sleep(0.5 * (2 ** attempt))

        # Fall back to writing in place. This is not atomic, so a reader
        # can catch a half-written file -- but a torn snapshot is
        # recoverable and a missing result is not.
        try:
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write('\n')
            return data
        except OSError as error:
            last_error = error

        # A periodic snapshot is monitoring, not results. Killing a job
        # that is hours into training because one of them failed costs
        # far more than the snapshot is worth, so warn and continue --
        # but never swallow a failure of the FINAL write, which would
        # leave a finished run indistinguishable from a crashed one.
        if required:
            raise last_error
        print(
            f'WARNING: could not write {self.summary_path}: {last_error}',
            file=sys.stderr,
            flush=True,
        )
        return data

    def close(self, status='completed', global_step=None, extra=None):
        """
        Write the final summary and close open files.
        """

        data = self.write_summary(
            status=status, global_step=global_step, extra=extra,
            required=True
        )
        self._episodes_file.close()
        return data


def runtime_metadata():
    """
    Stable host/runtime details useful when runs move to a cluster.
    """

    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
