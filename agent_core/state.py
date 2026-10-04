"""Survey state: the target catalogue, learned sky-quality scale, and per-target
progress. Built once from `initialize`, then updated from every `decision_request`'s
messages and `last_result`. Holds no hidden data -- only what the public protocol
hands us, plus what we infer from our own hits (never from a file).

Mirrors the structure-of-parallel-arrays design (ids/ra/dec/flux/weight/required/...,
all indexed by the same integer i) used by this project's companion TypeScript example,
so both examples solve the same problem the same way and can be compared directly.
"""
from __future__ import annotations

import bisect
import math
import statistics
from collections import deque
from typing import NamedTuple, Optional

from .geometry import FiberGrid, max_hour_angle_deg, parse_utc, wrap180
from .scoring import ScoringModel

ALT_MARGIN_DEG = 0.6
SKY_MEMORY_HOURS = 2.0
RECENT_SAMPLES = 60
EARLIER_SAMPLES = 60
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


class PendingPrediction(NamedTuple):
    model: float          # lunar/airmass quality model used at planning time
    band_model: float      # independent weather/geometry prior, never throughput feedback
    alt: float
    az: float
    clean: bool            # true when no all-sky notice / directional block applied at plan time


class FaultEvidence(NamedTuple):
    recent_median: float
    earlier_median: float
    drop: float
    recent_samples: int
    recent_nights: int
    earlier_samples: int
    dark_checks: int
    dark_matched: int


class BlockedPatch(NamedTuple):
    az: float
    alt: float
    expires_hours: float


class ExposureRecord(NamedTuple):
    action_index: int
    target_id: str
    score: float
    lower: float
    upper: float
    estimate: float
    start_utc: str
    end_utc: str


def _mod(a: float, n: float) -> float:
    m = a % n
    return m + n if m < 0 else m


class SurveyState:
    def __init__(self, init_payload: dict):
        site = init_payload["site"]
        survey = init_payload["survey"]
        instrument = init_payload["instrument"]
        limits = init_payload.get("limits", {})

        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site.get("minimum_altitude_deg", 30.0))
        self.sun_altitude_limit_deg = float(site.get("sun_altitude_limit_deg", -18.0))

        self.survey_start = parse_utc(survey["start_utc"])
        self.survey_end = parse_utc(survey["end_utc"])
        self.slot_seconds = int(survey.get("slot_seconds", 900))
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"]))
                       for n in survey.get("nights", [])]
        from datetime import timedelta
        local_offset = float(site.get("utc_offset_hours", self.lon / 15))
        self.night_dates = [n.get("night_date", (start + timedelta(hours=local_offset - 12)).date().isoformat())
                            for n, (start, _) in zip(survey.get("nights", []), self.nights)]

        self.fiber_grid = FiberGrid(instrument)
        exposure = instrument.get("exposure", {})
        self.min_exposure = int(exposure.get("min_duration_seconds", 60))
        self.max_exposure = int(exposure.get("max_duration_seconds", 3600))

        self.scoring = ScoringModel(init_payload.get("scoring", {}), site)

        reporting = init_payload.get("scoring", {}).get("reporting", {})
        self.max_consecutive_reports = int(reporting.get("max_consecutive_reports", limits.get("max_consecutive_reports", 32)))
        self.false_report_free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.false_reports_since_correct = 0
        self.response_max_bytes = int(limits.get("response_max_bytes", 524288))

        # Parallel arrays, one slot per target, in catalogue order.
        self.ids: list[str] = []
        self.ra: list[float] = []
        self.dec: list[float] = []
        self.flux: list[float] = []
        self.weight: list[float] = []
        self.required: list[bool] = []
        self.index_of: dict[str, int] = {}

        columns = init_payload.get("targets", {}).get("columns", [])
        col = {name: idx for idx, name in enumerate(columns)}
        for row in init_payload.get("targets", {}).get("rows", []):
            target_id = str(row[col["target_id"]])
            self.index_of[target_id] = len(self.ids)
            self.ids.append(target_id)
            self.ra.append(float(row[col["ra_deg"]]))
            self.dec.append(float(row[col["dec_deg"]]))
            self.flux.append(float(row[col["feature_flux"]]))
            self.weight.append(float(row[col["science_weight"]]))
            self.required.append(bool(row[col["required"]]))

        n = len(self.ids)
        self.hmax = [max_hour_angle_deg(self.dec[i], self.lat, self.min_alt + ALT_MARGIN_DEG) for i in range(n)]
        self.factor = [0.0] * n
        self.factor_upper = [0.0] * n
        self.factor_estimate = [0.0] * n
        self.best_score = [0.0] * n
        self.misses = [0] * n
        self.attempts = [0] * n
        self.active = [i for i in range(n) if self.hmax[i] > 0.0]

        self._cells: dict[int, list[tuple[float, int]]] = {}
        self._build_index()
        self.first_night, self.last_night = self._build_windows()

        self.scale = 1.0
        self.prior_scale = 1.0
        self._samples: deque = deque(maxlen=24)           # (hours, ratio)
        self._all_ratios: deque = deque(maxlen=400)        # ratio
        self.clean_history: list[tuple[float, int, float]] = []  # (hours, night, ratio)
        self.clean_intervals = []  # (hours, night, lower ratio, upper ratio)
        self.censored_exposures = deque(maxlen=24)
        self.pending_night = -1
        self._band_checks: deque = deque(maxlen=60)        # (program, matched, model)
        self.force_program: Optional[str] = None
        self.pending: dict[str, PendingPrediction] = {}
        self.pending_program = "BACKUP"
        self.pending_duration = 0
        self.blocked: list[BlockedPatch] = []  # Temporary evidence from mixed positive/zero hits.
        self.notices: set[str] = set()                      # "kind|direction"
        self.terrain: set[str] = set()
        self.extra_avoid: set[str] = set()
        self.duration_scale = 1.0
        self.fast_level = 0

        # Per-observe-action ledger: (observe_action_index, target_id, factor), mirroring
        # the backend's own BestLedger so a Hard-mode state_resync can be answered exactly
        # (see _resync) instead of only from the resync message's best_scores.
        self.ledger: list[ExposureRecord] = []
        self.pending_action_index: Optional[int] = None
        self.pending_start = None

    # -- spatial index -------------------------------------------------------

    def _build_index(self) -> None:
        for i in self.active:
            key = math.floor(self.dec[i])
            self._cells.setdefault(key, []).append((self.ra[i], i))
        for band in self._cells.values():
            band.sort(key=lambda pair: pair[0])

    def neighbours(self, ra: float, dec: float, radius: float):
        """Indices within `radius` degrees of (ra, dec), using the 1-degree declination-band index."""
        found: list[int] = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = 180.0 if abs(dec) + radius >= 90 else min(180.0, radius / cos_dec)
        lo_key, hi_key = math.floor(dec - radius), math.floor(dec + radius)
        for key in range(lo_key, hi_key + 1):
            band = self._cells.get(key)
            if not band:
                continue
            spans: list[tuple[float, float]]
            lo, hi = ra - width, ra + width
            if width >= 180:
                spans = [(0.0, 360.0)]
            elif lo < 0:
                spans = [(0.0, hi), (lo + 360.0, 360.0)]
            elif hi >= 360:
                spans = [(lo, 360.0), (0.0, hi - 360.0)]
            else:
                spans = [(lo, hi)]
            keys = [r for r, _ in band]
            for low, high in spans:
                start = bisect.bisect_left(keys, low)
                end = bisect.bisect_right(keys, high)
                for k in range(start, end):
                    found.append(band[k][1])
        return found

    def _build_windows(self):
        """First/last night index on which each target has >=20 minutes above the limit."""
        need = self.min_exposure * SIDEREAL_DEG_PER_SECOND
        spans = []
        for start, end in self.nights:
            from .geometry import local_sidereal_deg
            l0 = local_sidereal_deg(start, self.lon)
            span = (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND
            spans.append((l0, span))
        n = len(self.ra)
        first_night = [len(self.nights)] * n
        last_night = [-1] * n
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = _mod(self.ra[i] - h - l0, 360.0)
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if first_night[i] > k:
                        first_night[i] = k
                    last_night[i] = k
        return first_night, last_night

    # -- messages and results -------------------------------------------------

    def on_messages(self, messages: list[dict], latest_bulletin: Optional[dict]) -> None:
        for message in messages:
            if message.get("record_type") == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        self.terrain.add(notice.get("direction"))
            elif message.get("record_type") == "state_resync":
                self._resync(message)
        notices = (latest_bulletin or {}).get("notices", [])
        current = {f"{n.get('event_kind')}|{n.get('direction')}" for n in notices
                   if n.get("event_kind") != "terrain_obstruction"}
        if current != self.notices:
            self.blocked.clear()  # An old local weather inference cannot outlive its source conditions.
        self.notices = current

    def factor_bounds(self, i, score, program=None):
        """All factors consistent with a rounded score and legal multipliers."""
        if self.weight[i] <= 0:
            return 0.0, 1.0
        if score < 0:
            return 0.0, 0.0
        multipliers = ({self.scoring.program_multipliers[program], self.scoring.mismatch_multiplier}
                      if program in self.scoring.program_multipliers else
                      set(self.scoring.program_multipliers.values()) | {self.scoring.mismatch_multiplier})
        score_lo, score_hi = max(0.0, score - .0000005), score + .0000005
        candidates = [(score_lo / (self.weight[i] * m), score_hi / (self.weight[i] * m)) for m in multipliers
                      if m > 0 and score_lo <= self.weight[i] * m]
        if not candidates:
            return 0.0, 1.0
        return min(1.0, min(lo for lo, hi in candidates)), min(1.0, max(hi for lo, hi in candidates))

    def _resync(self, message):
        window = message.get("invalidated_window") or {}
        start, end = window.get("action_index_start"), window.get("action_index_end_exclusive")
        self.ledger = [r for r in self.ledger if not (start <= r.action_index < end)] if start is not None and end is not None else []
        n = len(self.ids)
        self.factor, self.factor_upper, self.factor_estimate, self.best_score = ([0.0] * n for _ in range(4))
        for r in self.ledger:
            i = self.index_of[r.target_id]
            self.factor[i] = max(self.factor[i], r.lower)
            self.factor_upper[i] = max(self.factor_upper[i], r.upper)
            self.factor_estimate[i] = max(self.factor_estimate[i], r.estimate)
            self.best_score[i] = max(self.best_score[i], r.score)
        rows = message.get("best_scores") or []
        scores = ({r["target_id"]: float(r["best_score"]) for r in rows} if rows and isinstance(rows[0], dict)
                  else dict(zip(message.get("observed_target_ids", []), map(float, rows))))
        for target_id, i in self.index_of.items():
            score = scores.get(target_id, 0.0)
            if score > self.best_score[i] + 0.000001:
                lo, hi = self.factor_bounds(i, score)
                self.factor[i] = max(self.factor[i], lo)
                self.factor_upper[i] = max(self.factor_upper[i], hi)
                self.factor_estimate[i] = max(self.factor_estimate[i], (lo + hi) / 2)
            self.best_score[i] = score
        self.active = [i for i in range(n) if self.hmax[i] > 0.0]
        self.misses = [0] * n
        self.attempts = [0] * n
        self.forget_quality_history()

    def site_closed(self) -> bool:
        for key in self.notices:
            kind, _, direction = key.partition("|")
            if kind in ("rain", "storm") and direction == "ALL":
                return True
        return False

    def all_sky_notice(self) -> bool:
        return any(key.partition("|")[2] == "ALL" for key in self.notices)

    def on_result(self, last_result: Optional[dict], hours: float) -> None:
        self.blocked = [patch for patch in self.blocked if patch.expires_hours > hours]
        action_index = self.pending_action_index
        self.pending_action_index = None
        if last_result and last_result.get("action") == "report":
            if last_result.get("correct") is True:
                self.false_reports_since_correct = 0
                self.forget_quality_history()
            elif last_result.get("correct") is False:
                self.false_reports_since_correct += 1
        if not last_result or last_result.get("action") != "observe" or not self.pending:
            self.pending.clear()
            return
        hits = {h.get("target_id"): float(h.get("score", 0.0)) for h in last_result.get("hits", [])}
        any_positive = any(score > 0 for score in hits.values())
        scoring = self.scoring
        multipliers = scoring.program_multipliers
        mismatch = scoring.mismatch_multiplier
        declared_multiplier = multipliers.get(self.pending_program, 1.0)
        f0t0 = scoring.f0t0
        censored_clean = 0
        if self.pending_start is not None:
            self.pending_duration = min(self.pending_duration, max(1.0, hours * 3600 - (self.pending_start - self.survey_start).total_seconds()))

        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            if i is None:
                continue
            if target_id not in hits:
                self.misses[i] += 1
                continue
            score = hits[target_id]
            if score <= 0.0:
                lower, upper = self.factor_bounds(i, score, self.pending_program)
                self.factor_upper[i] = max(self.factor_upper[i], upper)
                if action_index is not None:
                    self._record_hit(action_index, target_id, score, lower, upper, 0, hours)
                if any_positive:
                    self.blocked.append(BlockedPatch(prediction.az, prediction.alt, hours + SKY_MEMORY_HOURS))
                    self.blocked = self.blocked[-40:]
                continue
            weight = self.weight[i] if self.weight[i] > 0 else 1e-9
            multiplier_seen = score / weight
            if prediction.clean:
                if abs(multiplier_seen - declared_multiplier) < 2e-4:
                    self._band_checks.append((self.pending_program, True, prediction.model))
                elif abs(multiplier_seen - mismatch) < 2e-4:
                    self._band_checks.append((self.pending_program, False, prediction.model))
            factor_if_match = score / (weight * declared_multiplier) if declared_multiplier > 0 else 0.0
            factor_if_miss = score / (weight * mismatch) if mismatch > 0 else 0.0
            band = scoring.program_band(prediction.band_model)
            # A score above the mismatch ceiling proves a bonus; otherwise
            # matching remains uncertain and uses the independent prior only.
            matched = score > weight * mismatch + 1e-9 or band == self.pending_program
            factor = factor_if_match if matched else factor_if_miss
            factor = min(1.0, factor)
            lower, upper = self.factor_bounds(i, score, self.pending_program)
            if prediction.clean and upper >= .97:
                censored_clean += 1
            self.factor[i] = max(self.factor[i], lower)
            self.factor_upper[i] = max(self.factor_upper[i], upper)
            self.factor_estimate[i] = max(self.factor_estimate[i], factor)
            self.best_score[i] = max(self.best_score[i], score)
            if action_index is not None:
                self._record_hit(action_index, target_id, score, lower, upper, factor, hours)
            if prediction.clean and upper < .97 and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                denominator = self.flux[i] * self.pending_duration * prediction.model / f0t0
                self.clean_intervals.append((hours, self.pending_night, lower / denominator, upper / denominator))
            if self.required[i] and self.factor[i] < scoring.required_threshold:
                self.attempts[i] += 1
            # If any legal interpretation is saturated, throughput has only a
            # lower bound. A guessed partial interpretation is not an exact
            # quality sample and must not pull the learned scale downward.
            if upper < 0.97 and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                ratio = (factor * f0t0) / (self.flux[i] * self.pending_duration * prediction.model)
                self._samples.append((hours, ratio))
                self._all_ratios.append(ratio)
                if prediction.clean:
                    self.clean_history.append((hours, self.pending_night, ratio))
        if censored_clean >= 3:
            self.censored_exposures.append((hours, self.pending_night))
        self.pending.clear()
        self.update_scale(hours)

    def _record_hit(self, action_index, target_id, score, lower, upper, factor, hours):
        from datetime import timedelta
        from .geometry import format_utc
        end = self.survey_start + timedelta(hours=hours)
        start = self.pending_start or end - timedelta(seconds=self.pending_duration)
        self.ledger.append(ExposureRecord(action_index, target_id, score, lower, upper, factor,
                                          format_utc(start), format_utc(end)))

    def has_recent_sample(self, hours: float) -> bool:
        return any(when >= hours - SKY_MEMORY_HOURS for when, _ in self._samples)

    def update_scale(self, hours: float) -> None:
        if len(self._all_ratios) >= 8:
            ordered = sorted(self._all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self._samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if len(recent) >= 4 else self.prior_scale

    # -- fault diagnostics ------------------------------------------------------

    def calibration_due(self, hours, night_index):
        """Only own feedback can establish that throughput is poorly sampled."""
        fresh = {hour for hour, night, _lower, _upper in self.clean_intervals
                 if night == night_index and hour >= hours - 2}
        censored = {hour for hour, night in self.censored_exposures
                    if night == night_index and hour >= hours - 2}
        return len(fresh) < 2 and len(censored) >= 4

    def fault_evidence(self, recent_night_count=2, min_night_exposures=4) -> Optional[FaultEvidence]:
        # Targets from the same exposure share weather and time. Aggregate them
        # first, then require a drop on two separate nights. A fixed tail of 60
        # targets ceases to span nights when a scheduler takes shorter exposures.
        groups = {}
        for hour, night, lower, upper in self.clean_intervals:
            groups.setdefault((hour, night), []).append((lower, upper))
        nights = sorted({night for _, night in groups})
        if len(nights) < recent_night_count + 1:
            return None
        recent_nights = set(nights[-recent_night_count:])
        recent = [(hour, night, statistics.median(p[1] for p in values))
                  for (hour, night), values in groups.items() if night in recent_nights]
        earlier = [statistics.median(p[0] for p in values)
                   for (_, night), values in groups.items() if night not in recent_nights]
        if len(earlier) < 8 or any(sum(night == n for _, night, _ in recent) < min_night_exposures for n in recent_nights):
            return None
        # Use the upper completion interpretation in each recent night against
        # the earlier lower interpretation. Program ambiguity alone cannot
        # manufacture a fall in efficiency. Both recent nights must be low.
        recent_median = max(statistics.median(value for _, night, value in recent if night == n) for n in recent_nights)
        earlier_median = statistics.median(earlier)
        # A saturated DARK bonus proves a matching band directly. Filtering
        # these hits by a guessed earlier sky scale can discard real evidence.
        dark = [c for c in list(self._band_checks)[-16:]
                if c[0] == "DARK"]
        return FaultEvidence(
            recent_median=round(recent_median, 3),
            earlier_median=round(earlier_median, 3),
            drop=round(recent_median / max(1e-9, earlier_median), 3),
            recent_samples=len(recent),
            recent_nights=len(recent_nights),
            earlier_samples=len(earlier),
            dark_checks=len(dark),
            dark_matched=sum(1 for c in dark if c[1]),
        )

    def forget_quality_history(self) -> None:
        self.blocked.clear()
        self.clean_history = []
        self.clean_intervals = []
        self.censored_exposures.clear()
        self._band_checks.clear()
        self._samples.clear()
        self._all_ratios.clear()
        self.prior_scale = 1.0

    # -- night lookup -------------------------------------------------------------

    def current_night(self, now):
        for index, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return index, start, end
        return None

    def next_night_start(self, now):
        for start, _end in self.nights:
            if start > now:
                return start
        return None
