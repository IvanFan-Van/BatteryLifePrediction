"""Read the observation schedule for the task-indexed Tongji dataset."""

import re
from dataclasses import dataclass

import yaml


def read_percent(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:[1-9][0-9]?|100)%", value):
        raise ValueError(f"Expected a whole percent from '1%' to '100%', got {value!r}")
    return int(value[:-1])


@dataclass(frozen=True)
class ObservationPhase:
    start_epoch: int
    cycle_percent: int
    segment_percent: int
    value_keep: int | str


@dataclass(frozen=True)
class ObservationSchedule:
    max_epochs: int
    phases: tuple[ObservationPhase, ...]

    @classmethod
    def read(cls, stream, *, feature_count):
        return cls.from_config(yaml.safe_load(stream), feature_count=feature_count)

    @classmethod
    def from_config(cls, config, *, feature_count):
        max_epochs = config["trainer"]["max_epochs"]
        if type(max_epochs) is not int or max_epochs < 1:
            raise ValueError("trainer.max_epochs must be a positive integer")

        items = config["observation_schedule"]
        if not isinstance(items, list) or not items:
            raise ValueError("observation_schedule must be a nonempty list")

        phases = []
        previous_start = -1
        for item in items:
            start = item["start_epoch"]
            if type(start) is not int or not previous_start < start < max_epochs:
                raise ValueError("start_epoch must increase and be within max_epochs")
            keep = item["value_keep"]
            if type(keep) is int:
                if not 1 <= keep <= feature_count:
                    raise ValueError(f"value_keep must be between 1 and {feature_count}")
            else:
                read_percent(keep)

            phases.append(ObservationPhase(
                start, read_percent(item["cycle_fraction"]),
                read_percent(item["segment_fraction"]), keep,
            ))
            previous_start = start

        if phases[0].start_epoch != 0:
            raise ValueError("The first observation phase must start at epoch 0")
        return cls(max_epochs, tuple(phases))

    def for_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < self.max_epochs:
            raise ValueError(f"Epoch out of range: {epoch}")
        return next(phase for phase in reversed(self.phases) if phase.start_epoch <= epoch)
