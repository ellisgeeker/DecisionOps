"""Decision model provider adapters."""

from decisionops.adapters.base import DecisionProvider, StaticFileProvider
from decisionops.adapters.http import HttpProvider

__all__ = [
    "DecisionProvider",
    "HttpProvider",
    "JevProvider",
    "LayaProvider",
    "StaticFileProvider",
]


def __getattr__(name: str):
    # Lazy: keep torch and the TypeSafe SDK out of the default import path.
    if name == "LayaProvider":
        from decisionops.adapters.laya import LayaProvider

        return LayaProvider
    if name == "JevProvider":
        from decisionops.adapters.jev import JevProvider

        return JevProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
