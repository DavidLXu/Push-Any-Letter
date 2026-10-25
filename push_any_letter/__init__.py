"""Push arbitrary font glyphs with a circular end effector."""
from gymnasium.envs.registration import register
from .env import PushLetterEnv

register(id="PushAnyLetter-v0", entry_point="push_any_letter.env:PushLetterEnv")
__all__ = ["PushLetterEnv"]
