"""BPF tuning-skills helpers (v4).

Lessons from v2 (CF=155 / 188.3) + BW=15 generalization (seed 20260911):
- PB-ok + SB-bad with peak in-band must NOT open with strong narrow.
- When SB meets but PB collapsed, widen (recover PB), never keep narrowing.
- Peak "near" margin must be capped (min(½·BW, 3 MHz)); uncapped ½·BW over-fires
  narrow on wide goals and stalls centering outside the true window.
- Peak just outside → slight center into window, not narrow.
- Peak inside but off-center → nudge toward mid before narrowing.
- Finish line: keep slight widen while SB has ≥1 dB margin; hold only when SB is thin.
"""
from __future__ import annotations

from pathlib import Path

from jobs.bpf5_agent.goals import BpfGoalSpec
from jobs.bpf5_agent.task import VARIABLES

DEFAULT_SKILLS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "bpf_tuning_skills_system.md"
)

_SERIES_L = ("L1", "L3", "L5")
_SERIES_C = ("C1", "C3", "C5")
_SHUNT_L = ("L2", "L4")
_SHUNT_C = ("C2", "C4")

# Cap far/near hysteresis so wide BW goals still force true outside → center.
_PEAK_MARGIN_CAP_HZ = 3.0e6


def load_skills_system_prompt(path: Path | None = None) -> str:
    p = path or DEFAULT_SKILLS_PATH
    return p.read_text(encoding="utf-8").strip()


def peak_margin_hz(band_hz: float) -> float:
    """BW-aware peak hysteresis margin (Hz)."""
    return min(0.5 * max(band_hz, 1.0), _PEAK_MARGIN_CAP_HZ)


def _token(direction: str, magnitude: str) -> str:
    if direction == "hold":
        return "hold"
    if magnitude == "strong":
        return f"{direction}_strong"
    if magnitude == "slight":
        return f"{direction}_slight"
    return direction


def _hold_all() -> dict[str, str]:
    return {v: "hold" for v in VARIABLES}


def _apply_narrow_bw(magnitude: str) -> dict[str, str]:
    intent = _hold_all()
    for k in _SERIES_L:
        intent[k] = _token("increase", magnitude)
    for k in _SERIES_C:
        intent[k] = _token("decrease", magnitude)
    for k in _SHUNT_L:
        intent[k] = _token("decrease", magnitude)
    for k in _SHUNT_C:
        intent[k] = _token("increase", magnitude)
    return intent


def _apply_widen_bw(magnitude: str) -> dict[str, str]:
    intent = _hold_all()
    for k in _SERIES_L:
        intent[k] = _token("decrease", magnitude)
    for k in _SERIES_C:
        intent[k] = _token("increase", magnitude)
    for k in _SHUNT_L:
        intent[k] = _token("increase", magnitude)
    for k in _SHUNT_C:
        intent[k] = _token("decrease", magnitude)
    return intent


def _apply_center_shift(direction: str, magnitude: str) -> dict[str, str]:
    intent = _hold_all()
    move = "decrease" if direction == "up" else "increase"
    for k in VARIABLES:
        intent[k] = _token(move, magnitude)
    return intent


def _gap_magnitude(gap_db: float) -> str:
    if gap_db > 12:
        return "strong"
    if gap_db > 5:
        return "normal"
    return "slight"


def _sb_magnitude_peak_aware(sb_gap_db: float, peak_side: str | None) -> str:
    """When peak is in/near window, never open with strong; prefer normal over crawl."""
    if peak_side in ("inside", "near", None):
        if sb_gap_db > 2:
            return "normal"
        return "slight"
    return _gap_magnitude(sb_gap_db)


def _peak_side(peak: float, f_lo: float, f_hi: float, margin: float) -> str:
    if peak < f_lo - margin:
        return "below"
    if peak > f_hi + margin:
        return "above"
    if f_lo <= peak <= f_hi:
        return "inside"
    return "near"


def _last_block(prev_thinking: str | None) -> str | None:
    if not prev_thinking:
        return None
    t = prev_thinking.lower()
    if "widen" in t:
        return "widen"
    if "narrow" in t:
        return "narrow"
    if "center up" in t:
        return "center_up"
    if "center down" in t:
        return "center_down"
    return None


def _intent_block_summary(intent: dict[str, str]) -> str:
    """Describe which Skill-B block the intent matches (no raw JSON dump)."""
    if all(v == "hold" for v in intent.values()):
        return "hold all variables"

    def _mag(token: str) -> str:
        if token.endswith("_strong"):
            return "strong"
        if token.endswith("_slight"):
            return "slight"
        if token == "hold":
            return "hold"
        return "normal"

    sample = next(v for v in intent.values() if v != "hold")
    mag = _mag(sample)
    # Classify block from relative directions of L1/C1/L2/C2.
    l1, c1, l2, c2 = intent["L1"], intent["C1"], intent["L2"], intent["C2"]
    if all(t.startswith("decrease") for t in intent.values()):
        return f"center-up block ({mag}): decrease every L and C"
    if all(t.startswith("increase") for t in intent.values()):
        return f"center-down block ({mag}): increase every L and C"
    if (
        l1.startswith("increase")
        and c1.startswith("decrease")
        and l2.startswith("decrease")
        and c2.startswith("increase")
    ):
        return (
            f"narrow-BW block ({mag}): series L↑ C↓, shunt L↓ C↑ "
            "(Skill B narrower-BW row)"
        )
    if (
        l1.startswith("decrease")
        and c1.startswith("increase")
        and l2.startswith("increase")
        and c2.startswith("decrease")
    ):
        return (
            f"widen-BW block ({mag}): series L↓ C↑, shunt L↑ C↓ "
            "(Skill B wider-BW row)"
        )
    changed = ", ".join(f"{k}={v}" for k, v in intent.items() if v != "hold")
    return f"sparse / mixed intents ({mag}): {changed}"


def format_skills_reasoning(
    cost: dict,
    goal: BpfGoalSpec,
    intent: dict[str, str],
    diag: str,
    iteration: int,
) -> str:
    """SFT-ready chain-of-thought for one skills-policy step.

    Goes into ``<reasoning>...</reasoning>`` only. Do not embed intent JSON here;
    ``format_agent_completion`` already writes a separate ``<intent>`` block.
    """
    pb = float(cost["passband_min_s21_db"])
    sb = float(cost["stopband_max_s21_db"])
    pb_gate = float(goal.passband_il_max_db)
    sb_gate = float(goal.stopband_atten_min_db)
    pb_gap = pb_gate - pb
    sb_gap = sb - sb_gate
    peak = cost.get("s21_peak_freq_hz")
    f_lo, f_hi = goal.f_low_hz, goal.f_high_hz
    band = max(f_hi - f_lo, 1.0)
    side = None
    peak_mhz = None
    if isinstance(peak, (int, float)):
        peak_mhz = float(peak) / 1e6
        side = _peak_side(float(peak), f_lo, f_hi, peak_margin_hz(band))

    peak_line = (
        f"- S21 peak ≈ {peak_mhz:.1f} MHz vs goal window "
        f"[{f_lo/1e6:.1f}, {f_hi/1e6:.1f}] MHz → peak_side={side}"
        if peak_mhz is not None
        else "- S21 peak frequency unavailable"
    )

    lines = [
        f"Iteration {iteration}: read the observation against the BPF skills card.",
        "",
        "1) Observe metrics vs gates",
        f"- Passband min S21 = {pb:.2f} dB (gate {pb_gate:.2f} dB, gap {pb_gap:+.2f} dB)",
        f"- Stopband max S21 = {sb:.2f} dB (gate {sb_gate:.2f} dB, gap {sb_gap:+.2f} dB)",
        peak_line,
        "",
        "2) Diagnose with Skill A priority",
        f"- {diag}",
        "",
        "3) Map diagnosis → Skill B block + Skill C magnitude",
        f"- {_intent_block_summary(intent)}",
        "- Emit only qualitative increase/decrease/hold tokens; no numeric L/C values.",
        "",
        "4) Why this move (for the next sim)",
    ]

    dlow = diag.lower()
    if "center up" in dlow:
        lines.append(
            "- Peak sits far below the goal window while PB is bad, so shift the "
            "whole ladder toward higher center frequency before polishing BW."
        )
    elif "center down" in dlow:
        lines.append(
            "- Peak sits far above the goal window while PB is bad, so shift the "
            "whole ladder toward lower center frequency before polishing BW."
        )
    elif "oscillation" in dlow:
        lines.append(
            "- Consecutive peaks flipped across the window (Skill G), so stop "
            "strong centering and switch to a BW block."
        )
    elif "widen" in dlow and "recover" in dlow:
        lines.append(
            "- SB already meets its gate but PB collapsed; narrowing further would "
            "hurt PB, so widen-BW to recover the passband (Skill A/E)."
        )
    elif "widen" in dlow:
        lines.append(
            "- SB meets (or has margin) while PB is still short of its gate, so "
            "apply a controlled widen-BW step (Skill F / finish-line)."
        )
    elif "narrow" in dlow:
        lines.append(
            "- PB is acceptable enough that the broken gate is stopband leak; "
            "narrow-BW raises selectivity. Magnitude follows Skill C "
            "(no _strong when peak is already in/near the window)."
        )
    elif "hold" in dlow:
        lines.append(
            "- Gaps are closed or further moves would chatter at the finish line; hold."
        )
    elif "polish" in dlow:
        lines.append(
            "- Both gaps are small; Skill D sparse-polishes only the middle arm."
        )
    else:
        lines.append(f"- Follow the diagnosis above: {diag}")

    return "\n".join(lines)


def choose_intent_from_skills(
    cost: dict,
    goal: BpfGoalSpec,
    iteration: int,
    prev_cost: dict | None = None,
    prev_thinking: str | None = None,
) -> tuple[dict[str, str], str]:
    if cost.get("goal_met"):
        return _hold_all(), "Skill G: goal already met — hold."

    pb = float(cost["passband_min_s21_db"])
    sb = float(cost["stopband_max_s21_db"])
    pb_gap = float(goal.passband_il_max_db) - pb
    sb_gap = sb - float(goal.stopband_atten_min_db)
    peak = cost.get("s21_peak_freq_hz")
    f_lo, f_hi = goal.f_low_hz, goal.f_high_hz
    band = max(f_hi - f_lo, 1.0)
    margin = peak_margin_hz(band)
    pb_ok = pb_gap <= 1.0
    sb_ok = sb_gap <= 0.0
    pb_meets = pb_gap <= 0.0
    prev_block = _last_block(prev_thinking)

    side = None
    if isinstance(peak, (int, float)):
        side = _peak_side(float(peak), f_lo, f_hi, margin)

    # --- Near both gates ---
    if abs(pb_gap) <= 1.5 and abs(sb_gap) <= 1.5:
        if not sb_ok and pb_meets:
            return (
                _apply_narrow_bw("slight"),
                "Skill A finish: SB barely short, PB meets — slight narrow.",
            )
        if not sb_ok and not pb_meets:
            if sb_gap >= pb_gap:
                return (
                    _apply_narrow_bw("slight"),
                    "Skill A finish: both short — slight narrow (SB larger gap).",
                )
            return (
                _apply_widen_bw("slight"),
                "Skill A finish: both short — slight widen (PB larger gap).",
            )
        if sb_ok and not pb_meets:
            # Keep widening while SB has real margin; hold only if SB is thin.
            if sb_gap <= -1.0:
                return (
                    _apply_widen_bw("slight"),
                    "Skill F finish: SB has margin, PB short — keep slight widen.",
                )
            if prev_block == "widen":
                return (
                    _hold_all(),
                    "Skill G: finish-line after widen with thin SB — hold (protect SB).",
                )
            return (
                _apply_widen_bw("slight"),
                "Skill F finish: SB meets, PB short — slight widen once.",
            )
        return _hold_all(), "Skill G: finish-line both meet — hold."

    # --- Priority 1: SB ok, PB failing → widen (recover passband) ---
    if sb_ok and pb_gap > 0:
        mag = _gap_magnitude(pb_gap)
        if prev_block == "narrow" and pb_gap > 2.5:
            mag = "normal" if mag == "slight" else mag
        return (
            _apply_widen_bw(mag),
            f"Skill A/E: SB ok ({sb:.1f} dB) but PB bad ({pb:.1f} dB) — "
            f"widen-BW ({mag}) to recover passband.",
        )

    # --- Priority 2: PB ok, SB failing → narrow (capped if peak in/near) ---
    if pb_ok and sb_gap > 0:
        mag = _sb_magnitude_peak_aware(sb_gap, side)
        if (
            prev_cost is not None
            and prev_block == "narrow"
            and sb_gap > 5
            and float(prev_cost.get("stopband_max_s21_db", sb)) - sb < 1.0
        ):
            mag = "strong" if mag == "normal" else ("normal" if mag == "slight" else mag)
        return (
            _apply_narrow_bw(mag),
            f"Skill A/C: PB ok ({pb:.1f} dB) but SB short ({sb:.1f} vs "
            f"{goal.stopband_atten_min_db:.1f}) — narrow-BW ({mag}), peak={side}.",
        )

    # --- Center far outside + PB bad ---
    if side in ("below", "above") and not pb_ok:
        if prev_cost is not None and isinstance(
            prev_cost.get("s21_peak_freq_hz"), (int, float)
        ):
            prev_side = _peak_side(
                float(prev_cost["s21_peak_freq_hz"]), f_lo, f_hi, margin
            )
            if {prev_side, side} == {"below", "above"}:
                mag = _sb_magnitude_peak_aware(max(sb_gap, 8.0), "near")
                return (
                    _apply_narrow_bw(mag),
                    f"Skill G: center oscillation — switch to narrow-BW ({mag}).",
                )
        mag = _gap_magnitude(max(pb_gap, abs(sb_gap)))
        if iteration >= 18:
            mag = "slight"
        if side == "below":
            return (
                _apply_center_shift("up", mag),
                f"Skill A/B: peak {float(peak)/1e6:.1f} MHz far below "
                f"[{f_lo/1e6:.1f},{f_hi/1e6:.1f}] — center up ({mag}).",
            )
        return (
            _apply_center_shift("down", mag),
            f"Skill A/B: peak {float(peak)/1e6:.1f} MHz far above "
            f"[{f_lo/1e6:.1f},{f_hi/1e6:.1f}] — center down ({mag}).",
        )

    # --- Just outside (near) + PB bad → center into window (not narrow) ---
    if side == "near" and pb_gap > 1 and isinstance(peak, (int, float)):
        mag = "slight" if pb_gap <= 5 else "normal"
        if float(peak) < f_lo:
            return (
                _apply_center_shift("up", mag),
                f"Skill A/B: peak {float(peak)/1e6:.1f} MHz just below window "
                f"[{f_lo/1e6:.1f},{f_hi/1e6:.1f}] — center up ({mag}).",
            )
        return (
            _apply_center_shift("down", mag),
            f"Skill A/B: peak {float(peak)/1e6:.1f} MHz just above window "
            f"[{f_lo/1e6:.1f},{f_hi/1e6:.1f}] — center down ({mag}).",
        )

    # --- Peak inside but off-center while PB bad → nudge toward mid ---
    if (
        side == "inside"
        and pb_gap > 1
        and isinstance(peak, (int, float))
        and sb_gap > 0
    ):
        mid = 0.5 * (f_lo + f_hi)
        if abs(float(peak) - mid) > 0.2 * band:
            mag = "slight" if pb_gap <= 8 else "normal"
            if float(peak) < mid:
                return (
                    _apply_center_shift("up", mag),
                    f"Skill A/B: peak {float(peak)/1e6:.1f} MHz inside but low vs mid "
                    f"{mid/1e6:.1f} — center up ({mag}).",
                )
            return (
                _apply_center_shift("down", mag),
                f"Skill A/B: peak {float(peak)/1e6:.1f} MHz inside but high vs mid "
                f"{mid/1e6:.1f} — center down ({mag}).",
            )

    # --- Both struggling, peak centered inside ---
    if pb_gap > 1:
        mag = _gap_magnitude(pb_gap)
        if side in ("inside", "near") and mag == "strong":
            mag = "normal"
        return (
            _apply_narrow_bw(mag),
            f"Skill A/B/C: PB_gap={pb_gap:.1f} SB_gap={sb_gap:.1f} "
            f"(peak={side}) — narrow-BW ({mag}).",
        )

    if sb_gap > 0:
        mag = _sb_magnitude_peak_aware(sb_gap, side)
        return (
            _apply_narrow_bw(mag),
            f"Skill A/C: SB short ({sb:.1f} dB) — narrow-BW ({mag}).",
        )

    intent = _hold_all()
    if pb_gap > 0:
        intent["L3"] = _token("increase", "slight")
        intent["C3"] = _token("decrease", "slight")
        return intent, "Skill D: sparse polish on L3/C3 (slight)."

    return _hold_all(), "Skill G: gaps closed — hold."
