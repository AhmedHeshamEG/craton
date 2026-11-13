"""Field 2 as a CratonBench method: ``craton bench run field2`` (and ``field2-born``).

``field2`` is the heart trained on the world (``configs/field2/heart.yaml``); ``field2-born`` is
the same network at step 0, which is ETAS exactly, run to show it. Both answer CB-32 through
``unseen_forecaster`` (the heart trained again without the region) and CB-29 through
``world_forecaster().npp_factor``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from craton.bench.baselines import register_method
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field2.world import Field2World


def _quiet(_: str) -> None:
    return None


@dataclass
class Field2Method:
    name: str = "field2"
    kind: str = "model"
    config_name: str = "heart"
    born: bool = False
    log: Callable[[str], None] = _quiet
    _worlds: dict[str, Field2World] = field(default_factory=dict)

    def _world(self, holdout: str) -> Field2World:
        if holdout not in self._worlds:
            self._worlds[holdout] = Field2World(
                self.config_name, log=self.log, holdout=holdout, born=self.born
            )
        return self._worlds[holdout]

    def world_forecaster(self) -> Any:
        return self._world("")

    def unseen_forecaster(self, region: str) -> Any:
        return self._world(region)

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return self._world("").forecast(spec, ctx, seed)


FIELD2 = register_method(Field2Method())
FIELD2_BORN = register_method(Field2Method(name="field2-born", born=True))
