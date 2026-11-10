"""Experiment framework: describe (spec), run in parallel (runner), analyse (analysis), draw (plots)."""
from mascov.experiments.spec import Experiment, Job, ModelSpec, PlotSpec, expand

__all__ = ["Experiment", "Job", "ModelSpec", "PlotSpec", "expand"]
