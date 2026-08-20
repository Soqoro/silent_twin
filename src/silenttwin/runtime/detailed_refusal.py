"""Detailed decision-correlated refusal baseline."""

from silenttwin.runtime.base import Runtime


class DetailedRefusalRuntime(Runtime):
    name = "detailed_refusal"


DetailedRefusal = DetailedRefusalRuntime

__all__ = ["DetailedRefusal", "DetailedRefusalRuntime"]
