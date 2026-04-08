"""Pipeline registry for Occursus-Claude."""

from __future__ import annotations

from pipelines.base import BasePipeline
from pipelines.single import SinglePipeline

_REGISTRY: dict[str, BasePipeline] = {}


def register_pipeline(pipeline: BasePipeline) -> None:
    _REGISTRY[pipeline.spec().id] = pipeline


def get_pipeline(pipeline_id: str) -> BasePipeline | None:
    return _REGISTRY.get(pipeline_id)


def list_pipelines() -> list[BasePipeline]:
    return list(_REGISTRY.values())


def _register_defaults() -> None:
    register_pipeline(SinglePipeline())

    try:
        from pipelines.best_of_n import BestOf3Pipeline, SampleAndVotePipeline
        register_pipeline(BestOf3Pipeline())
        register_pipeline(SampleAndVotePipeline())
    except ImportError:
        pass

    try:
        from pipelines.merge import (
            MergeFullPipeline, CritiqueThenMergePipeline, RankedMergePipeline
        )
        register_pipeline(MergeFullPipeline())
        register_pipeline(CritiqueThenMergePipeline())
        register_pipeline(RankedMergePipeline())
    except ImportError:
        pass

    try:
        from pipelines.debate import (
            Debate2WayPipeline, DissentThenMergePipeline, RedTeamBlueTeamPipeline
        )
        register_pipeline(Debate2WayPipeline())
        register_pipeline(DissentThenMergePipeline())
        register_pipeline(RedTeamBlueTeamPipeline())
    except ImportError:
        pass

    try:
        from pipelines.deep import (
            ChainOfVerificationPipeline, IterativeRefinementPipeline,
            MixtureOfAgentsPipeline
        )
        register_pipeline(ChainOfVerificationPipeline())
        register_pipeline(IterativeRefinementPipeline())
        register_pipeline(MixtureOfAgentsPipeline())
    except ImportError:
        pass

    try:
        from pipelines.experimental import (
            PersonaCouncilPipeline, AdversarialDecompositionPipeline,
            ReverseEngineerPipeline, TournamentPipeline
        )
        register_pipeline(PersonaCouncilPipeline())
        register_pipeline(AdversarialDecompositionPipeline())
        register_pipeline(ReverseEngineerPipeline())
        register_pipeline(TournamentPipeline())
    except ImportError:
        pass

    try:
        from pipelines.routing import ExpertRoutingPipeline, ConstraintCheckerPipeline
        register_pipeline(ExpertRoutingPipeline())
        register_pipeline(ConstraintCheckerPipeline())
    except ImportError:
        pass


_register_defaults()
