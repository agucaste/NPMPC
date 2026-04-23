from __future__ import annotations

import atexit
import csv
import math
import os
from collections import deque
from typing import Any, Callable, TextIO

import numpy as np
from rich.console import Console
from rich.table import Table

from core.config import Config

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class Logger:
    """Tabular logger for non-expert experiments."""

    def __init__(
        self,
        output_dir: str,
        exp_name: str,
        output_fname: str = "progress.csv",
        use_wandb: bool = False,
        config: Config | None = None,
        wandb_name: str | None = None,
    ) -> None:
        self._console = Console()
        self._log_dir = output_dir
        self._exp_name = exp_name
        os.makedirs(self._log_dir, exist_ok=True)

        self._output_file: TextIO = open(
            os.path.join(self._log_dir, output_fname),
            mode="w",
            encoding="utf-8",
            newline="",
        )
        atexit.register(self._output_file.close)
        self._csv_writer = csv.writer(self._output_file)

        self._epoch = 0
        self._first_row = True
        self._data: dict[str, deque[float] | list[float]] = {}
        self._headers_windows: dict[str, int | None] = {}
        self._headers_minmax: dict[str, bool] = {}
        self._headers_delta: dict[str, bool] = {}
        self._display_formats: dict[str, str | None] = {}
        self._display_formatters: dict[str, Callable[[float], str] | None] = {}
        self._current_row: dict[str, float] = {}

        self._use_wandb = bool(use_wandb)
        self._wandb_enabled = False

        self.log(f"Logging data to {self._output_file.name}", "cyan", bold=True)

        if config is not None:
            self.save_config(config)

        if self._use_wandb:
            if wandb is None:
                self.log("wandb is not installed; disabling wandb logging.", "yellow", bold=True)
            else:
                project = getattr(config, "wandb_project", "non-expert") if config is not None else "non-expert"
                wandb.init(
                    project=project,
                    name=wandb_name or exp_name,
                    dir=self._log_dir,
                    config=config.todict() if config is not None else None,
                )
                self._wandb_enabled = True

    def log(self, msg: str, color: str = "green", bold: bool = False) -> None:
        style = " ".join([color, "bold" if bold else ""]).strip()
        self._console.print(msg, style=style)

    def save_config(self, config: Config) -> None:
        self.log("Saving config to config.json", "yellow", bold=True)
        with open(os.path.join(self._log_dir, "config.json"), mode="w", encoding="utf-8") as f:
            f.write(config.tojson())

    def register_key(
        self,
        key: str,
        window_length: int | None = None,
        min_and_max: bool = False,
        delta: bool = False,
        display_format: str | None = None,
        formatter: Callable[[float], str] | None = None,
    ) -> None:
        assert key not in self._current_row, f"Key {key} has been registered"
        self._current_row[key] = 0.0
        self._display_formats[key] = display_format
        self._display_formatters[key] = formatter

        if min_and_max:
            self._current_row[f"{key}/Min"] = 0.0
            self._current_row[f"{key}/Max"] = 0.0
            self._current_row[f"{key}/Std"] = 0.0
            self._headers_minmax[key] = True
            self._headers_minmax[f"{key}/Min"] = False
            self._headers_minmax[f"{key}/Max"] = False
            self._headers_minmax[f"{key}/Std"] = False
            self._display_formats[f"{key}/Min"] = display_format
            self._display_formats[f"{key}/Max"] = display_format
            self._display_formats[f"{key}/Std"] = display_format
            self._display_formatters[f"{key}/Min"] = formatter
            self._display_formatters[f"{key}/Max"] = formatter
            self._display_formatters[f"{key}/Std"] = formatter
        else:
            self._headers_minmax[key] = False

        if delta:
            self._current_row[f"{key}/Delta"] = 0.0
            self._headers_delta[key] = True
            self._headers_delta[f"{key}/Delta"] = False
            self._headers_minmax[f"{key}/Delta"] = False
            self._display_formats[f"{key}/Delta"] = display_format
            self._display_formatters[f"{key}/Delta"] = formatter
        else:
            self._headers_delta[key] = False

        if window_length is not None:
            self._data[key] = deque(maxlen=window_length)
            self._headers_windows[key] = window_length
        else:
            self._data[key] = []
            self._headers_windows[key] = None

    def store(
        self,
        data: dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        if data is not None:
            kwargs.update(data)

        for key, val in kwargs.items():
            assert key in self._current_row, f"Key {key} has not been registered"
            if isinstance(val, (int, float, np.integer, np.floating)):
                self._data[key].append(float(val))
            elif isinstance(val, np.ndarray):
                self._data[key].append(float(np.mean(val)))
            else:
                raise ValueError(f"Unsupported type {type(val)} for key {key}")

    def dump_tabular(self) -> None:
        self._update_current_row()
        table = Table("Metric", "Value")

        for key, val in self._current_row.items():
            display_key = f"{key}/Mean" if self._headers_minmax.get(key, False) else key
            table.add_row(display_key, self._format_value(key, val))

        if self._first_row:
            self._csv_writer.writerow(self._current_row.keys())
            self._first_row = False
        self._csv_writer.writerow(self._current_row.values())
        self._output_file.flush()

        if self._wandb_enabled:
            wandb.log(self._current_row, step=self._epoch)

        self._console.print(table)
        self._epoch += 1

    def _update_current_row(self) -> None:
        for key in self._data:
            old_data = self._current_row[key]
            if self._headers_minmax[key]:
                mean, min_val, max_val, std = self.get_stats(key, min_and_max=True)
                self._current_row[key] = mean
                self._current_row[f"{key}/Min"] = min_val
                self._current_row[f"{key}/Max"] = max_val
                self._current_row[f"{key}/Std"] = std
            else:
                mean = self.get_stats(key)[0]
                self._current_row[key] = mean

            if self._headers_delta[key]:
                self._current_row[f"{key}/Delta"] = mean - old_data

            if self._headers_windows[key] is None:
                self._data[key] = []

    def get_stats(self, key: str, min_and_max: bool = False) -> tuple[float, ...]:
        assert key in self._current_row, f"Key {key} has not been registered"
        vals = self._data[key]
        if isinstance(vals, deque):
            vals = list(vals)
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            if min_and_max:
                return (math.nan, math.nan, math.nan, math.nan)
            return (math.nan,)

        if min_and_max:
            return (
                float(np.mean(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.std(arr)),
            )
        return (float(np.mean(arr)),)

    def _format_value(self, key: str, value: float) -> str:
        if isinstance(value, (float, np.floating)) and math.isnan(value):
            return "n/a"
        formatter = self._display_formatters.get(key)
        if formatter is not None:
            return formatter(float(value))
        display_format = self._display_formats.get(key)
        if display_format is not None:
            return format(float(value), display_format)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6g}"

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def close(self) -> None:
        if self._wandb_enabled:
            wandb.finish()
        self._output_file.close()
