"""Decision model provider adapters."""

from decisionops.adapters.base import DecisionProvider, StaticFileProvider

__all__ = ["DecisionProvider", "LayaProvider", "StaticFileProvider"]


def __getattr__(name: str):
    if name == "LayaProvider":
        from decisionops.adapters.laya import LayaProvider

        return LayaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
