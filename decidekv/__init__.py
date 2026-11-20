"""decidekv: fast decisions over many options with a local LLM.

    from decidekv import KV, Choice, Score, Noul
    kv = KV("qwen3:8b", embed="qwen3-embedding:4b")
    idx = kv.index(["option one", "option two", ...])     # up to flat_max options: the list lives in the KV cache
    idx.choose("some state").choice                       # each call computes only the tokens of the state
"""
from .core import KV, Index, iter_answers, render_state
from .ollama import Ollama, OllamaError
from .primitives import Choice, ChoiceAnswer, Noul, NoulAnswer, Response, Score, ScoreAnswer, Usage

__version__ = "1.3.0"
__all__ = ["KV", "Index", "Choice", "Score", "Noul", "ChoiceAnswer", "ScoreAnswer", "NoulAnswer",
           "Response", "Usage", "Ollama", "OllamaError", "iter_answers", "render_state", "__version__"]
