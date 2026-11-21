"""Same-process NAVSIM PDM scorer used as the GRPO reward.

Role in the pipeline
--------------------
During the NAVSIM RFT stage (``train_navsim/DriveIK_NAVSIM_rft.sh`` ->
``drivevla/training/grpo.py``) the IK diffusion head samples G candidate
trajectories per scene. ``GRPOLatentActionWMTrainer._maybe_apply_inline_pdm``
hands those G trajectories (meters, ego frame) and the scene token to
:meth:`InlinePDMScorer.score_batch`, which runs the NAVSIM devkit's PDM
simulation + scoring against the pre-computed NAVSIM *metric cache*
(``PDM_CACHE``) and returns one sub-score dict per trajectory.
:meth:`InlinePDMScorer.aggregate_pdm_score` then collapses each dict into the
scalar reward that drives the GRPO preference loss.

Because the navsim and nuplan devkits are importable from the training
process (installed, or added to ``PYTHONPATH`` via ``NAVSIM_DEVKIT_ROOT``), the
scorer is imported and called directly in-process -- no separate scoring
server. The G samples of a group are scored serially in the calling thread.

Performance: score_batch() loads the scene_token's MetricCache **once** and
shares it with all G samples, so lzma decompress + pickle.load runs once
instead of G times, and a small LRU (maxsize=8) sits in front of
get_from_token(), avoiding a disk hit when the same scene is scored again
shortly afterwards.

Latency (single trajectory, on a B200): ~50ms.

Usage:
    scorer = InlinePDMScorer(metric_cache_dir)
    rewards = scorer.score_batch(samples_m=(G,T,2), scene_token=...)
"""

from __future__ import annotations
import functools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class InlinePDMScorer:
    """In-process PDM scorer.

    Wraps the NAVSIM devkit pieces needed to score one ego-frame trajectory
    against a scene's MetricCache: a ``MetricCacheLoader`` (scene token ->
    lzma-pickled MetricCache), a ``PDMSimulator`` (LQR + kinematic bicycle
    rollout of the proposal), ``LogReplayTrafficAgents`` (non-reactive
    background traffic replayed from the log) and a ``PDMScorer``.

    Lazy init: the navsim/nuplan imports are deferred until the first call
    that needs them (``is_ready`` / ``has_token`` / ``score_batch``). The class
    itself can therefore be imported even when navsim is not installed, which
    keeps it loosely coupled to the training code. If initialization fails,
    the error is remembered and every later call reports "not ready" instead
    of retrying.
    """

    def __init__(
        self,
        metric_cache_dir: str,
        proposal_num_poses: int = 40,
        proposal_interval: float = 0.1,
    ):
        """Store the configuration only; no devkit import happens here.

        Args:
            metric_cache_dir:
                NAVSIM metric cache directory (``PDM_CACHE``). Must contain a
                ``metadata/*.csv`` listing the per-token ``metric_cache.pkl``
                files, as read by ``navsim.common.dataloader.MetricCacheLoader``.
            proposal_num_poses, proposal_interval:
                PDM proposal sampling (number of poses, seconds between
                poses). Defaults 40 x 0.1 s = a 4 s horizon, matching
                NAVSIM's PDM scoring. Model trajectories are resampled onto
                this grid in ``_score_one``.
        """
        # Plain configuration, normalized to the expected Python types.
        self.metric_cache_dir = str(metric_cache_dir)
        self.proposal_num_poses = int(proposal_num_poses)
        self.proposal_interval = float(proposal_interval)

        # Devkit objects, created by _lazy_init() on first use.
        # _initialized / _init_error record the outcome of that one attempt.
        self._loader = None
        self._scorer = None
        self._simulator = None
        self._traffic_agents = None
        self._proposal_sampling = None
        self._initialized = False
        self._init_error: Optional[str] = None

    # ── Lazy initialization ──────────────────────────────────────────────────

    def _lazy_init(self) -> bool:
        """Import the devkit and build loader / simulator / scorer (once).

        Returns:
            True if the scorer is usable, False if initialization failed now
            or on an earlier call (the failure is cached in ``_init_error``;
            later calls that see it return False without retrying).

        Side effects:
            Reads the metric-cache metadata CSV and prints an "init OK" /
            "init FAILED" line.
        """
        # Already initialized, or a previous attempt failed.
        if self._initialized:
            return True
        if self._init_error is not None:
            return False
        try:
            t0 = time.time()
            # Deferred devkit imports (navsim + nuplan).
            from navsim.common.dataloader import MetricCacheLoader
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
                PDMScorer,
                PDMScorerConfig,
            )
            from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
                PDMSimulator,
            )
            from navsim.traffic_agents_policies.log_replay_traffic_agents import (
                LogReplayTrafficAgents,
            )
            from nuplan.planning.simulation.trajectory.trajectory_sampling import (
                TrajectorySampling,
            )

            # Common time grid for proposals, simulation and traffic
            # replay (default 40 poses x 0.1 s = 4 s).
            self._proposal_sampling = TrajectorySampling(
                num_poses=self.proposal_num_poses,
                interval_length=self.proposal_interval,
            )

            # PDMScorerConfig defaults from yaml (the NAVSIM devkit's
            # pdm_scorer.yaml): EPDMS weights 5/5/2/2/2 for EP/TTC/LK/HC/EC
            # plus the metric thresholds. human_penalty_filter=True makes
            # the devkit reset a sub-metric to 1 when the logged human
            # trajectory also fails it (only for original, non-synthetic
            # scenes).
            scorer_cfg_kwargs = dict(
                progress_weight=5.0,
                ttc_weight=5.0,
                lane_keeping_weight=2.0,
                history_comfort_weight=2.0,
                two_frame_extended_comfort_weight=2.0,
                driving_direction_horizon=1.0,
                driving_direction_compliance_threshold=2.0,
                driving_direction_violation_threshold=6.0,
                stopped_speed_threshold=5e-03,
                future_collision_horizon_window=1.0,
                progress_distance_threshold=5.0,
                lane_keeping_deviation_limit=0.5,
                lane_keeping_horizon_window=2.0,
                human_penalty_filter=True,
            )

            # Devkit objects shared by every scoring call:
            #   scorer    - computes the PDM sub-metrics of simulated states
            #   simulator - LQR tracker + kinematic bicycle rollout
            #   traffic   - non-reactive (log-replay) background agents
            #   loader    - token -> metric_cache.pkl path index (reads only
            #               the metadata CSV here; caches load on demand)
            self._scorer = PDMScorer(
                proposal_sampling=self._proposal_sampling,
                config=PDMScorerConfig(**scorer_cfg_kwargs),
            )
            self._simulator = PDMSimulator(proposal_sampling=self._proposal_sampling)
            self._traffic_agents = LogReplayTrafficAgents(
                future_trajectory_sampling=self._proposal_sampling
            )
            self._loader = MetricCacheLoader(Path(self.metric_cache_dir))

            # Log success (token count = number of scenes in the cache).
            init_dt = time.time() - t0
            print(
                f"[PDM-inline] init OK  tokens={len(self._loader.tokens)}  "
                f"init_time={init_dt:.2f}s"
            )
            self._initialized = True
            return True
        except Exception as ex:
            # Any failure (devkit missing, bad cache dir, ...) disables
            # the scorer permanently; callers then see is_ready() == False.
            self._init_error = f"{type(ex).__name__}: {ex}"
            print(f"[PDM-inline] init FAILED: {self._init_error}")
            return False

    def is_ready(self) -> bool:
        """Return True once the devkit objects exist (triggers lazy init)."""
        return self._lazy_init()

    def has_token(self, scene_token: str) -> bool:
        """Return True if ``scene_token`` has an entry in the metric cache.

        Args:
            scene_token: NAVSIM scene token (16-char hex, without any
                ``_wm``-style suffix).

        Returns:
            False if the scorer is not ready or the token is unknown. Note that
            ``MetricCacheLoader.tokens`` builds a fresh list of all cached
            tokens on every access, so this is a linear membership test.
        """
        if not self.is_ready():
            return False
        return scene_token in self._loader.tokens

    # ── Cache loading (with LRU) ─────────────────────────────────────────────
    # MetricCacheLoader.get_from_token() performs an lzma decompress + pickle
    # load on every call (~10-20ms / call on a B200 node's SSD). The G samples of
    # a group share one scene_token, so score_batch decompresses it only once
    # and shares the object. A short LRU is added as well, so revisiting the
    # same scene within a few steps does not hit the disk again.

    def _get_cache_uncached(self, scene_token: str):
        """Raw lzma+pickle load of one scene's MetricCache (no caching).

        Only called through ``_cached_loader_inner`` (i.e. via ``get_cache``).
        Raises ``KeyError`` for a token missing from the cache index.
        """
        return self._loader.get_from_token(scene_token)

    @functools.lru_cache(maxsize=8)
    def _cached_loader_inner(self, scene_token: str):
        """LRU on (self, scene_token).

        ``functools.lru_cache`` includes ``self`` in the key, so entries of
        different instances never collide. The decorator wraps the class
        function, however, so the 8 slots are shared by all instances and the
        cache keeps strong references to ``self``. maxsize=8 assumes a cache of
        roughly ~30MB, which absorbs same-scene revisits from the previous
        steps without a large memory cost. Exceptions are not cached.
        """
        return self._get_cache_uncached(scene_token)

    def get_cache(self, scene_token: str):
        """LRU-cached MetricCache fetch.

        Called once per ``score_batch``; the returned object is then shared by
        all G samples. The object is treated as read-only by this
        module.
        """
        return self._cached_loader_inner(scene_token)

    # ── Single trajectory scoring ────────────────────────────────────────────

    def _score_one(
        self,
        scene_token: str,
        traj_xy_m: np.ndarray,  # (T, 2) ego-frame
        cache: Optional[Any] = None,  # pre-loaded MetricCache (shared by score_batch)
    ) -> Optional[Dict[str, float]]:
        """PDM-score a single trajectory and return its sub-score dict.

        The heading is derived from the positions with a plain ``np.gradient``
        finite difference (on NAVSIM expert GT this scores dac≈0.95, versus
        ≈0.48 with a Savitzky-Golay tangent). If ``cache`` is given, the lzma
        decompress is skipped (score_batch path).

        Args:
            scene_token: NAVSIM scene token; must exist in the metric cache.
            traj_xy_m: float array (T, 2), meters, in the ego frame NAVSIM expects for ``Trajectory.poses`` (nuPlan
                convention: x forward, y left). The T poses are assumed to be
                evenly spaced over a 4 s horizon, i.e. pose k at (k+1)*4/T s.
            cache: pre-loaded MetricCache for ``scene_token`` or None to load it.

        Returns:
            Dict with keys ``nc`` (no at-fault collision), ``dac`` (drivable
            area compliance), ``ddc`` (driving direction compliance), ``tlc``
            (traffic light compliance), ``ep`` (ego progress), ``ttc`` (time to
            collision within bound), ``lk`` (lane keeping), ``hc`` (history
            comfort), ``ec`` (two-frame extended comfort) and ``pdm_score``
            (the devkit's own aggregate), all floats in [0, 1] (``pdm_score``
            may be NaN). None if the scorer is not ready, the token is not
            cached, or anything raised during scoring (the error is printed).
        """
        # Not usable / unknown scene -> no score (callers treat None as
        # "invalid sample").
        if not self.is_ready():
            return None
        if not self.has_token(scene_token):
            return None
        try:
            from navsim.common.dataclasses import Trajectory
            from navsim.evaluate.pdm_score import (
                pdm_score_from_interpolated_trajectory,
                transform_trajectory,
            )

            # Standalone call: fetch (LRU-cached) MetricCache ourselves.
            if cache is None:
                cache = self.get_cache(scene_token)

            # The training model's traj_xy_m is (T_model, 2) — usually
            # 8 waypoints at 0.5 s = 4 s. PDM needs 40 poses at 0.1 s = 4 s, so
            # upsample by linear interpolation in time. No origin pose (t=0) is
            # prepended, so np.interp holds the first waypoint constant for
            # PDM times earlier than the first model time (0.1-0.4 s for 8 wp).
            T_model = int(traj_xy_m.shape[0])
            T_pdm = self.proposal_num_poses
            interval = self.proposal_interval
            # Timestamps [s] of the model waypoints and of the PDM poses.
            t_model = np.arange(1, T_model + 1) * (4.0 / T_model)  # assume 4s horizon
            t_pdm = np.arange(1, T_pdm + 1) * interval

            # Resample x / y onto the PDM time grid.
            x_pdm = np.interp(t_pdm, t_model, traj_xy_m[:, 0])
            y_pdm = np.interp(t_pdm, t_model, traj_xy_m[:, 1])
            # Heading: finite difference of the resampled positions.
            dxs = np.gradient(x_pdm)
            dys = np.gradient(y_pdm)
            heading = np.arctan2(dys, dxs)

            # Build the NAVSIM Trajectory (T_pdm, 3) = (x, y, heading) in the
            # ego frame, then convert it to a global-frame nuPlan
            # InterpolatedTrajectory anchored at the cached initial ego state
            # (transform_trajectory prepends that state at t=0).
            poses = np.stack([x_pdm, y_pdm, heading], axis=-1).astype(np.float64)
            traj = Trajectory(poses=poses, trajectory_sampling=self._proposal_sampling)
            traj_interp = transform_trajectory(traj, cache.ego_state)

            # Simulate (LQR + bicycle) and score the proposal against the
            # cached map / agents / PDM reference trajectory.
            result = pdm_score_from_interpolated_trajectory(
                metric_cache=cache,
                pred_trajectory=traj_interp,
                future_sampling=self._proposal_sampling,
                simulator=self._simulator,
                scorer=self._scorer,
                traffic_agents_policy=self._traffic_agents,
            )
            # The devkit checked here returns (results_df, simulated_states);
            # the isinstance check also accepts a bare results DataFrame.
            df = result[0] if isinstance(result, tuple) else result

            # Extract the one-row DataFrame into plain floats. Missing columns
            # fall back to a neutral default (1.0 for pass/fail metrics, 0.0
            # for progress, NaN for pdm_score). In the NAVSIM devkit checked
            # for this release the single-frame result has no
            # "two_frame_extended_comfort" column (EC is computed by the
            # devkit's scene aggregator), so "ec" takes the 1.0 fallback.
            return {
                "nc": float(df.get("no_at_fault_collisions", [1.0])[0]),
                "dac": float(df.get("drivable_area_compliance", [1.0])[0]),
                "ddc": float(df.get("driving_direction_compliance", [1.0])[0]),
                "tlc": float(df.get("traffic_light_compliance", [1.0])[0]),
                "ep": float(df.get("ego_progress", [0.0])[0]),
                "ttc": float(df.get("time_to_collision_within_bound", [1.0])[0]),
                "lk": float(df.get("lane_keeping", [1.0])[0]),
                "hc": float(df.get("history_comfort", [1.0])[0]),
                "ec": float(df.get("two_frame_extended_comfort", [1.0])[0]),
                "pdm_score": float(df.get("pdm_score", [float("nan")])[0]),
            }
        except Exception as ex:
            # Scoring errors are non-fatal: report and mark this sample invalid.
            print(f"[PDM-inline] score_one failed: {type(ex).__name__}: {ex}")
            return None

    # ── Batch scoring ────────────────────────────────────────────────────────

    def score_batch(
        self,
        samples_m: np.ndarray,  # (G, T, 2)
        scene_token: str,
    ) -> List[Optional[Dict[str, float]]]:
        """PDM-score the G samples of one scene, one after the other.

        The scene_token's MetricCache is lzma-decompressed only once per batch
        and then shared by the G samples.

        Args:
            samples_m: float array (G, T, 2), meters, ego frame; all G samples
                belong to the same scene.
            scene_token: NAVSIM scene token of that scene.

        Returns:
            List of length G, in sample order. Each entry is the sub-score dict
            from ``_score_one`` or None for a sample that could not be scored.
            All entries are None if the scorer is not ready or the cache
            could not be loaded.
        """
        # Scorer unusable -> every sample is invalid.
        if not self.is_ready():
            return [None] * int(samples_m.shape[0])
        G = int(samples_m.shape[0])

        # Load the cache only once per batch (saves G x lzma decompress).
        try:
            cache = self.get_cache(scene_token)
        except Exception as ex:
            print(
                f"[PDM-inline] cache load failed for {scene_token!r}: " f"{type(ex).__name__}: {ex}"
            )
            return [None] * G

        return [self._score_one(scene_token, samples_m[g], cache=cache) for g in range(G)]

    # ── Reward aggregation ───────────────────────────────────────────────────

    @staticmethod
    def aggregate_pdm_score(sub: Dict[str, float]) -> float:
        """Collapse a sub-score dict into the scalar GRPO reward; NaN if None.

        Uses the same weighted formula as the official EPDMS (NAVSIM v2):

        EPDMS = NC × DAC × DDC × TLC × (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC) / 16

        This is recomputed from the sub-scores; the devkit's own ``pdm_score``
        entry of the dict is not used. With single-frame scoring ``ec`` is the
        1.0 fallback (see ``_score_one``), so the EC term contributes a
        constant 2/16 to the weighted part.

        Args:
            sub: dict returned by ``_score_one`` (keys nc, dac, ddc, tlc, ep,
                ttc, lk, hc, ec), or None.

        Returns:
            Float in [0, 1], or NaN when ``sub`` is None.
        """
        if sub is None:
            return float("nan")
        # Multiplicative gate: any failed hard constraint (0) zeroes the reward
        # (NC can also be 0.5 for collisions with static objects).
        mult = sub["nc"] * sub["dac"] * sub["ddc"] * sub["tlc"]
        # Weighted average of the soft metrics (weights sum to 16).
        weighted = (
            5 * sub["ep"] + 5 * sub["ttc"] + 2 * sub["lk"] + 2 * sub["hc"] + 2 * sub["ec"]
        ) / 16.0
        return mult * weighted
