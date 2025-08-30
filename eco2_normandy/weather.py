from dataclasses import dataclass
from typing import Callable
from random import uniform, choice

from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from eco2_normandy.logger import Logger


# --- Classe Weather ---
@dataclass
class CurrentData:
    angle: int
    speed: int


@dataclass
class WeatherReport(object):
    wind: int
    wave: int
    current: CurrentData


class WeatherStation(object):
    def __init__(
        self,
        values: list,
        num_period: int,
        weather_probability: dict | None = None,
        generator: Callable | None = None,
        logger=None,
    ) -> None:
        self.logger = logger or Logger()
        self.weather: list[WeatherReport] = []
        if generator:
            self.weather = generator()
        else:
            self.weather: list[WeatherReport] = self._generate_weather(weather_probability, num_period, values)

    def get_weather(self, period: int) -> WeatherReport:
        """
        Returns the weather report for a period

        Args:
            period (int): period to get the weather from

        Returns:
            WeatherReport: the weather report for the period
        """
        return self.weather[period]

    def get_weather_period(self, start: int, end: int) -> list[WeatherReport]:
        """
        Returns the weather reports for multiple periods

        Args:
            start (int): start of the period
            end (int): end of the period

        Returns:
            list[WeatherReport]: list of weather reports
        """
        start_idx = int(max(0, start))
        end_idx = int(min(len(self.weather), end + 1))
        return self.weather[start_idx:end_idx]

    def _generate_weather(self, weather_probability: dict, num_period: int, values: dict) -> list:
        wind_p = weather_probability.get("wind", 0)
        wave_p = weather_probability.get("waves", 0)
        current_p = weather_probability.get("current", 0)

        wind_w = 1
        wave_w = 1
        current_w = 1
        sticky_factor = 0.2

        weathers = []
        for _ in range(num_period):
            wind = 0
            wave = 0
            current = 0

            if uniform(0, 1) <= wind_p:
                if uniform(0, 1) <= wind_w:
                    wind = choice(values) if wind_w == 1 else weathers[-1].wind
                    wind_w -= sticky_factor
                else:
                    wind_w = 1

            if uniform(0, 1) <= wave_p:
                if uniform(0, 1) <= wave_w:
                    wave = choice(values) if wave_w == 1 else weathers[-1].wave
                    wave_w -= sticky_factor
                else:
                    wave_w = 1

            if uniform(0, 1) <= current_p:
                if uniform(0, 1) <= current_w:
                    current = choice(values) if current_w == 1 else weathers[-1].current
                    current_w -= sticky_factor
                else:
                    current_w = 1

            weathers.append(WeatherReport(wind, wave, current))
        return weathers
