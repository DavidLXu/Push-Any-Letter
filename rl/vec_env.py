"""200 live environments in 50 independent CPU processes (4 per process)."""
import multiprocessing as mp
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from .worker import group_worker


class GroupedVecEnv(VecEnv):
    def __init__(self, num_envs=200, workers=50, **kwargs):
        if num_envs % workers:
            raise ValueError("num_envs must be divisible by workers")
        self.group_size = num_envs // workers
        self.waiting = self.closed = False
        context = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for _ in range(workers):
            remote, child = context.Pipe()
            process = context.Process(target=group_worker, args=(child, remote, kwargs, self.group_size), daemon=True)
            process.start()
            child.close()
            self.remotes.append(remote)
            self.processes.append(process)
        spaces = [self._recv(r) for r in self.remotes]
        super().__init__(num_envs, *spaces[0])

    def _recv(self, remote):
        if not remote.poll(180):
            raise TimeoutError("CPU environment worker did not respond within 180 seconds")
        ok, data = remote.recv()
        if not ok:
            raise RuntimeError("Environment worker failed:\n" + data)
        return data

    def reset(self):
        for i, remote in enumerate(self.remotes):
            start = i * self.group_size
            remote.send(("reset", list(zip(self._seeds[start:start+self.group_size], self._options[start:start+self.group_size]))))
        results = [item for r in self.remotes for item in self._recv(r)]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs)

    def step_async(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i*self.group_size:(i+1)*self.group_size]))
        self.waiting = True

    def step_wait(self):
        results = [item for r in self.remotes for item in self._recv(r)]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, np.float32), np.array(dones), list(infos)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    self._recv(remote)
                except (EOFError, RuntimeError, TimeoutError):
                    pass
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # Only our own worker, never other jobs.
                process.join(timeout=5)
        for remote in self.remotes:
            remote.close()
        self.closed = True

    def _dispatch(self, cmd, args, indices):
        indices = list(self._get_indices(indices))
        wanted = set(indices)
        selected = {}
        for group, remote in enumerate(self.remotes):
            local = [i for i in range(self.group_size) if group*self.group_size+i in wanted]
            remote.send((cmd, (*args, local)))
        for group, remote in enumerate(self.remotes):
            for i, value in self._recv(remote):
                selected[group*self.group_size+i] = value
        return [selected[i] for i in indices]

    def get_attr(self, attr_name, indices=None):
        return self._dispatch("get_attr", (attr_name,), indices)

    def set_attr(self, attr_name, value, indices=None):
        self._dispatch("set_attr", (attr_name, value), indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._dispatch("env_method", (method_name, method_args, method_kwargs), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
