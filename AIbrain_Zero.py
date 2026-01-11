from numpy import random as np_random
import random
import numpy as np
import copy
import time
from zipfile import BadZipFile
from constants import MAX_SPEED as ENGINE_MAX_SPEED


class AIbrain_Zero:
    generation_stats = {
        "best_score": float("-inf"),
        "best_checkpoint": 0,
        "mutation_count": 0,
        "record_count": 0,
        "init_count": 0,
        "resize_count": 0,
        "last_summary_t": 0.0,
    }

    SUMMARY_INTERVAL_S = 2.5
    LOG_EVERY_N_MUTATIONS = 25
    MAX_TILES_REASONABLE = 10_000
    TILE_PX_GUESS = 40.0
    MAX_RAY_DIST = 5.0
    MAX_SPEED = float(ENGINE_MAX_SPEED) / 2.5
    CLIP_Z2 = 60.0
    WEIGHT_CLIP = 5.0
    RAY_SCALE_AUTO = None

    FRONT_BRAKE = 0.24
    FRONT_COAST = 0.36
    TARGET_SPEED_STRAIGHT = 0.55
    TARGET_SPEED_TURN = 0.35
    TURN_BALANCE_TRIG = 0.18
    
    SAFE_ZONE_MIN = 0.25
    SAFE_ZONE_MAX = 0.35
    BOUNDARY_DANGER_DIST = 0.3
    BOUNDARY_CRITICAL_DIST = 0.23

    def __init__(self):
        self.score = 0.0
        self.id = ''.join(random.choices("0123456789", k=3))
        self.speed = 0.0
        self.x = 0.0
        self.y = 0.0
        self.initialized = False
        self.input_size = None
        self.hidden_size = 24
        self.hidden_size2 = 16
        self.output_size = 4
        self.w1, self.b1 = None, None
        self.w2, self.b2 = None, None
        self.w3, self.b3 = None, None
        self.use_layer_norm = False
        self.last_mutation_tag = "INIT"
        self.pg_enabled = True
        self.pg_lr = 0.002
        self.pg_baseline = 0.0
        self.pg_baseline_beta = 0.02
        self._pg_prev_distance = None
        self._pg_prev_time = None
        self._pg_last_step = None
        self.init_param()

    def init_param(self):
        self.NAME = f"Bot_{self.id}"
        self.store()

    def _print_summary_if_needed(self):
        now = time.perf_counter()
        last = AIbrain_Zero.generation_stats["last_summary_t"]
        if (now - last) < self.SUMMARY_INTERVAL_S:
            return
        AIbrain_Zero.generation_stats["last_summary_t"] = now
        bs = AIbrain_Zero.generation_stats["best_score"]
        bc = AIbrain_Zero.generation_stats["best_checkpoint"]
        mc = AIbrain_Zero.generation_stats["mutation_count"]
        rc = AIbrain_Zero.generation_stats["record_count"]
        ic = AIbrain_Zero.generation_stats["init_count"]
        rz = AIbrain_Zero.generation_stats["resize_count"]
        print(f"BestScore: {bs:.2f} | Tiles: {bc} | Records: {rc} | Mut: {mc} | Inits: {ic} | Resizes: {rz}")

    def initialize_network(self, input_size: int, reason: str = "INIT"):
        input_size = int(input_size)
        self.w1 = np_random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / max(1, input_size))
        self.b1 = np.zeros(self.hidden_size, dtype=np.float64)
        self.w2 = np_random.randn(self.hidden_size, self.hidden_size2) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.hidden_size2, dtype=np.float64)
        self.w3 = np_random.randn(self.hidden_size2, self.output_size) * np.sqrt(2.0 / self.hidden_size2)
        self.b3 = np.array([-0.5, -1.6, -0.2, -0.2], dtype=np.float64)
        self.input_size = input_size
        self.initialized = True
        self.last_mutation_tag = reason
        AIbrain_Zero.generation_stats["init_count"] += 1
        self.store()

    def _resize_input_layer(self, new_input_size: int):
        old = int(self.w1.shape[0])
        new = int(new_input_size)
        if new == old:
            return
        if new > old:
            pad = new - old
            extra = np_random.randn(pad, self.hidden_size) * np.sqrt(2.0 / max(1, new))
            self.w1 = np.vstack([self.w1, extra])
        else:
            cut = old - new
            self.w1 = self.w1[:new, :]
        self.input_size = new
        AIbrain_Zero.generation_stats["resize_count"] += 1
    
    def _layer_norm(self, x, epsilon=1e-8):
        if not self.use_layer_norm:
            return x
        mean = np.mean(x)
        std = np.std(x) + epsilon
        return (x - mean) / std

    def _ensure_network(self, desired_input_size: int):
        desired_input_size = int(desired_input_size)
        if not self.initialized:
            self.initialize_network(desired_input_size, reason="AUTO_INIT")
            return
        if self.input_size is None:
            self.input_size = int(self.w1.shape[0])
        if self.input_size != desired_input_size:
            self._resize_input_layer(desired_input_size)
            self.store()

    def store(self):
        if not self.initialized:
            self.parameters = {
                "NAME": self.NAME,
                "initialized": False,
                "input_size": None,
                "hidden_size": self.hidden_size,
                "hidden_size2": self.hidden_size2,
                "output_size": self.output_size,
                "pg_enabled": bool(self.pg_enabled),
                "pg_lr": float(self.pg_lr),
                "pg_baseline": float(self.pg_baseline),
                "pg_baseline_beta": float(self.pg_baseline_beta),
            }
        else:
            self.parameters = {
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
                "w3": self.w3,
                "b3": self.b3,
                "NAME": self.NAME,
                "initialized": True,
                "input_size": int(self.input_size),
                "hidden_size": int(self.hidden_size),
                "hidden_size2": int(self.hidden_size2),
                "output_size": int(self.output_size),
                "pg_enabled": bool(self.pg_enabled),
                "pg_lr": float(self.pg_lr),
                "pg_baseline": float(self.pg_baseline),
                "pg_baseline_beta": float(self.pg_baseline_beta),
            }

    def _normalize_rays(self, data):
        rays_raw = np.asarray(data, dtype=np.float64)
        raw_max = float(np.max(rays_raw)) if rays_raw.size else 1.0
        if AIbrain_Zero.RAY_SCALE_AUTO is None:
            if raw_max <= 1.5:
                AIbrain_Zero.RAY_SCALE_AUTO = 1.0
            elif raw_max <= 10.0:
                AIbrain_Zero.RAY_SCALE_AUTO = 10.0
            elif raw_max <= 100.0:
                AIbrain_Zero.RAY_SCALE_AUTO = 100.0
            else:
                AIbrain_Zero.RAY_SCALE_AUTO = 1000.0
        scale = float(AIbrain_Zero.RAY_SCALE_AUTO)
        rays = np.clip(rays_raw, 0.0, scale) / scale
        return rays, raw_max

    def _forward(self, inputs: np.ndarray, *, return_cache: bool = False):
        z1 = inputs @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ self.w3 + self.b3
        z3 = np.clip(z3, -self.CLIP_Z2, self.CLIP_Z2)
        a3 = 1.0 / (1.0 + np.exp(-z3))
        if not return_cache:
            return a3
        cache = {
            "inputs": inputs,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "p": a3,
        }
        return a3, cache

    def _sample_actions(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        a = (np_random.rand(4) < p).astype(np.float64)
        if a[0] > 0.5 and a[1] > 0.5:
            if p[1] >= p[0]:
                a[0] = 0.0
            else:
                a[1] = 0.0
        if a[2] > 0.5 and a[3] > 0.5:
            if p[2] >= p[3]:
                a[3] = 0.0
            else:
                a[2] = 0.0
        return a

    def _pg_update(self, step_cache: dict, *, reward: float):
        if not self.pg_enabled:
            return
        if step_cache is None:
            return
        r = float(reward)
        self.pg_baseline = (1.0 - self.pg_baseline_beta) * self.pg_baseline + self.pg_baseline_beta * r
        adv = r - self.pg_baseline
        adv = float(np.clip(adv, -2.0, 2.0))
        if abs(adv) < 1e-9:
            return
        a = step_cache["a"]
        p = step_cache["p"]
        delta3 = (a - p) * adv
        a2 = step_cache["a2"]
        z2 = step_cache["z2"]
        a1 = step_cache["a1"]
        z1 = step_cache["z1"]
        x = step_cache["inputs"]
        lr = float(self.pg_lr)
        gw3 = np.outer(a2, delta3)
        gb3 = delta3
        delta2 = (self.w3 @ delta3) * (z2 > 0.0)
        gw2 = np.outer(a1, delta2)
        gb2 = delta2
        delta1 = (self.w2 @ delta2) * (z1 > 0.0)
        gw1 = np.outer(x, delta1)
        gb1 = delta1
        self.w3 += lr * gw3
        self.b3 += lr * gb3
        self.w2 += lr * gw2
        self.b2 += lr * gb2
        self.w1 += lr * gw1
        self.b1 += lr * gb1
        np.clip(self.w1, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w1)
        np.clip(self.b1, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b1)
        np.clip(self.w2, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w2)
        np.clip(self.b2, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b2)
        np.clip(self.w3, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w3)
        np.clip(self.b3, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b3)

    def _extract_situation(self, rays: np.ndarray) -> dict:
        n = len(rays)
        c = n // 2
        front_ids = [max(0, c - 1), c, min(n - 1, c + 1)]
        front_min = float(np.min(rays[front_ids])) if n else 0.0
        third = max(1, n // 3) if n else 1
        left_mean = float(np.mean(rays[:third])) if n else 0.0
        right_mean = float(np.mean(rays[n - third:])) if n else 0.0
        balance = left_mean - right_mean
        is_turn = abs(balance) > self.TURN_BALANCE_TRIG
        target_speed = self.TARGET_SPEED_TURN if is_turn else self.TARGET_SPEED_STRAIGHT
        return {
            "front_min": front_min,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "balance": balance,
            "is_turn": is_turn,
            "target_speed": target_speed,
        }

    def _apply_safety_throttle_brake(
        self,
        *,
        front_min: float,
        speed_norm: float,
        target_speed: float,
        a4: np.ndarray,
    ) -> tuple[float, float]:
        brake = 1.0 if (front_min < self.FRONT_BRAKE and speed_norm > 0.15) else 0.0
        if front_min < self.FRONT_BRAKE * 0.7:
            brake = 1.0
        if brake > 0.5:
            gas = 0.0
        else:
            gas = float(a4[0])
            if front_min >= self.FRONT_COAST and speed_norm < target_speed:
                gas = max(gas, 0.80)
            if front_min >= self.FRONT_COAST and speed_norm < 0.05:
                gas = max(gas, 0.95)
            if front_min < self.FRONT_COAST:
                gas = min(gas, 0.55)
        if speed_norm >= target_speed:
            gas = 0.0
            if speed_norm > (target_speed + 0.15):
                brake = 1.0
        return gas, brake

    def _steering_from_net(self, a4: np.ndarray) -> tuple[float, float]:
        left = float(a4[2])
        right = float(a4[3])
        if left > 0.5 and right > 0.5:
            if left >= right:
                right = 0.0
            else:
                left = 0.0
        return left, right

    def _apply_boundary_avoidance(
        self,
        *,
        left_mean: float,
        right_mean: float,
        balance: float,
        gas: float,
        brake: float,
        left: float,
        right: float,
    ) -> tuple[float, float, float, float]:
        boundary_danger = AIbrain_Zero.BOUNDARY_DANGER_DIST
        boundary_critical = AIbrain_Zero.BOUNDARY_CRITICAL_DIST
        if left_mean < boundary_danger:
            if left_mean < boundary_critical:
                right = 1.0
                left = 0.0
                if brake < 0.5:
                    gas = gas * 0.5
            else:
                if right < 0.3:
                    left = 0.7
                right = 0.0
        if right_mean < boundary_danger:
            if right_mean < boundary_critical:
                left = 1.0
                right = 0.0
                if brake < 0.5:
                    gas = gas * 0.5
            else:
                if left < 0.3:
                    right = 0.7
                left = 0.0
        if left <= 0.5 and right <= 0.5:
            if balance > 0.18:
                left = 1.0
            elif balance < -0.18:
                right = 1.0
        return gas, brake, left, right

    def decide(self, data):
        desired_input_size = len(data) + 1
        self._ensure_network(desired_input_size)
        rays, _raw_max = self._normalize_rays(data)
        speed_norm = float(np.clip(self.speed / self.MAX_SPEED, -1.0, 1.0))
        inputs = np.concatenate([rays, np.array([speed_norm], dtype=np.float64)], axis=0)
        p, cache = self._forward(inputs, return_cache=True)
        sit = self._extract_situation(rays)
        a = self._sample_actions(p)
        cache["a"] = a
        gas, brake = self._apply_safety_throttle_brake(
            front_min=sit["front_min"],
            speed_norm=speed_norm,
            target_speed=sit["target_speed"],
            a4=p,
        )
        left, right = self._steering_from_net(p)
        gas, brake, left, right = self._apply_boundary_avoidance(
            left_mean=sit["left_mean"],
            right_mean=sit["right_mean"],
            balance=sit["balance"],
            gas=gas,
            brake=brake,
            left=left,
            right=right,
        )
        out = np.array([gas, brake, left, right], dtype=np.float64)
        out = (out > 0.5).astype(np.float64)
        if out[0] > 0.5 and out[1] > 0.5:
            out[0] = 0.0
        if out[2] > 0.5 and out[3] > 0.5:
            out[3] = 0.0
        cache["p"] = p
        cache["sit"] = sit
        self._pg_last_step = cache
        return out.tolist()


    def mutate(self):
        if not self.initialized:
            return
        AIbrain_Zero.generation_stats["mutation_count"] += 1
        mcount = AIbrain_Zero.generation_stats["mutation_count"]
        best_checkpoint = AIbrain_Zero.generation_stats.get("best_checkpoint", 0)
        if best_checkpoint > 50:
            luck = random.random()
            if luck < 0.20:
                mutation_rate, mutation_strength = 0.001, 0.001
                mutation_type = "ELITE"
            elif luck < 0.70:
                mutation_rate, mutation_strength = 0.10, 0.12
                mutation_type = "FINE"
            else:
                mutation_rate, mutation_strength = 0.20, 0.25
                mutation_type = "RAND"
        else:
            luck = random.random()
            if luck < 0.10:
                mutation_rate, mutation_strength = 0.001, 0.001
                mutation_type = "ELITE"
            elif luck < 0.60:
                mutation_rate, mutation_strength = 0.10, 0.12
                mutation_type = "FINE"
            else:
                mutation_rate, mutation_strength = 0.25, 0.30
                mutation_type = "RAND"
        self.last_mutation_tag = mutation_type
        for param in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3):
            mask = np_random.rand(*param.shape) < mutation_rate
            noise = np_random.normal(0.0, mutation_strength, param.shape)
            param += mask * noise
        np.clip(self.w1, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w1)
        np.clip(self.b1, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b1)
        np.clip(self.w2, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w2)
        np.clip(self.b2, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b2)
        np.clip(self.w3, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w3)
        np.clip(self.b3, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.b3)
        self.store()

    def calculate_score(self, distance, time_s, no):
        dist = float(distance) if distance is not None else 0.0
        t = float(time_s) if time_s is not None else 0.0
        safe_no = None
        try:
            n = float(no)
            if np.isfinite(n) and 0.0 <= n <= float(self.MAX_TILES_REASONABLE):
                safe_no = n
        except Exception:
            safe_no = None
        if safe_no is None:
            safe_no = dist / self.TILE_PX_GUESS
        avg_speed = dist / max(t, 1e-6)
        if self.pg_enabled:
            if self._pg_prev_distance is None or self._pg_prev_time is None:
                self._pg_prev_distance = float(dist)
                self._pg_prev_time = float(t)
            else:
                dd = float(dist) - float(self._pg_prev_distance)
                dt = float(t) - float(self._pg_prev_time)
                self._pg_prev_distance = float(dist)
                self._pg_prev_time = float(t)
                r = 2.0 * dd
                if self._pg_last_step is not None:
                    sit = self._pg_last_step.get("sit", {})
                    front_min = float(sit.get("front_min", 1.0))
                    left_mean = float(sit.get("left_mean", 1.0))
                    right_mean = float(sit.get("right_mean", 1.0))
                    if front_min < self.FRONT_BRAKE:
                        r -= 0.10
                    if left_mean < self.BOUNDARY_DANGER_DIST or right_mean < self.BOUNDARY_DANGER_DIST:
                        r -= 0.10
                    if left_mean < self.BOUNDARY_CRITICAL_DIST or right_mean < self.BOUNDARY_CRITICAL_DIST:
                        r -= 0.20
                    r += 0.01
                self._pg_update(self._pg_last_step, reward=r)
        progress_score = safe_no * 2000.0
        speed_bonus = avg_speed * 30.0
        self.score = progress_score + speed_bonus
        gs = AIbrain_Zero.generation_stats
        current_best = gs["best_score"]
        checkpoint = int(safe_no)
        improved_score = (not np.isfinite(current_best)) or (self.score > current_best)
        improved_checkpoint = checkpoint > int(gs["best_checkpoint"])
        if improved_score:
            gs["best_score"] = self.score
        if improved_checkpoint:
            gs["best_checkpoint"] = checkpoint
            gs["record_count"] += 1
        self._print_summary_if_needed()

    def passcardata(self, x, y, speed):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)

    def getscore(self):
        return float(self.score)

    def get_parameters(self):
        p = self.parameters
        if not p.get("initialized", False):
            return copy.deepcopy(p)
        input_size = p.get("input_size")
        if input_size is None and "w1" in p and p["w1"] is not None:
            input_size = int(p["w1"].shape[0]) if len(p["w1"].shape) > 0 else None
        return {
            "w1": np.array(p["w1"], copy=True),
            "b1": np.array(p["b1"], copy=True),
            "w2": np.array(p["w2"], copy=True),
            "b2": np.array(p["b2"], copy=True),
            "w3": np.array(p["w3"], copy=True),
            "b3": np.array(p["b3"], copy=True),
            "NAME": str(p["NAME"]),
            "initialized": True,
            "input_size": int(input_size) if input_size is not None else int(p.get("input_size", 0)),
            "hidden_size": int(p.get("hidden_size", self.hidden_size)),
            "hidden_size2": int(p.get("hidden_size2", self.hidden_size2)),
            "output_size": int(p.get("output_size", self.output_size)),
        }

    def set_parameters(self, p):
        if isinstance(p, np.lib.npyio.NpzFile):
            d = {}
            for k in p.files:
                try:
                    d[k] = p[k]
                except (BadZipFile, IOError, OSError, ValueError, KeyError, Exception) as e:
                    print(f"Warning: Failed to load '{k}' from .npz file (file may be corrupted): {e}")
                    d[k] = None
        else:
            d = copy.deepcopy(p)
        self.NAME = str(d.get("NAME", self.NAME))
        initialized = bool(d.get("initialized", False))
        if not initialized:
            self.initialized = False
            self.input_size = None
            self.w1 = self.b1 = self.w2 = self.b2 = self.w3 = self.b3 = None
            self.store()
            return
        if d.get("w1") is not None:
            self.w1 = np.array(d["w1"], copy=True)
        if d.get("b1") is not None:
            self.b1 = np.array(d["b1"], copy=True)
        if d.get("w2") is not None:
            self.w2 = np.array(d["w2"], copy=True)
        if d.get("b2") is not None:
            self.b2 = np.array(d["b2"], copy=True)
        if d.get("w3") is not None:
            self.w3 = np.array(d["w3"], copy=True)
        if d.get("b3") is not None:
            self.b3 = np.array(d["b3"], copy=True)
        if hasattr(self, 'w1') and self.w1 is not None and len(self.w1.shape) > 1:
            self.hidden_size = int(d.get("hidden_size", self.w1.shape[1]))
            self.input_size = int(d.get("input_size", self.w1.shape[0]))
        else:
            self.hidden_size = int(d.get("hidden_size", self.hidden_size))
            self.input_size = int(d.get("input_size", self.input_size))
        if hasattr(self, 'w2') and self.w2 is not None and len(self.w2.shape) > 1:
            self.hidden_size2 = int(d.get("hidden_size2", self.w2.shape[1]))
        else:
            self.hidden_size2 = int(d.get("hidden_size2", self.hidden_size2))
        if hasattr(self, 'w3') and self.w3 is not None and len(self.w3.shape) > 1:
            self.output_size = int(d.get("output_size", self.w3.shape[1]))
        else:
            self.output_size = int(d.get("output_size", self.output_size))
        self.initialized = True
        self.store()