"""CPU-only process entry point; deliberately does not import PyTorch."""
import traceback
from .env import OnlineLetterEnv


def group_worker(remote, parent, kwargs, count):
    parent.close()
    envs = []
    try:
        envs = [OnlineLetterEnv(**kwargs) for _ in range(count)]
        remote.send((True, (envs[0].observation_space, envs[0].action_space)))
        while True:
            cmd, payload = remote.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                result = [env.reset(seed=seed, options=options) for env, (seed, options) in zip(envs, payload)]
            elif cmd == "step":
                result = []
                for env, action in zip(envs, payload):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                        info["reset_info"] = reset_info
                    result.append((obs, reward, done, info))
            elif cmd == "get_attr":
                name, indices = payload
                result = [(i, getattr(envs[i], name)) for i in indices]
            elif cmd == "set_attr":
                name, value, indices = payload
                result = [(i, setattr(envs[i], name, value)) for i in indices]
            elif cmd == "env_method":
                name, args, kwargs, indices = payload
                result = [(i, getattr(envs[i], name)(*args, **kwargs)) for i in indices]
            else:
                raise ValueError(cmd)
            remote.send((True, result))
    except EOFError:
        pass
    except BaseException:
        remote.send((False, traceback.format_exc()))
    finally:
        for env in envs:
            env.close()
        remote.close()
