"""Paired program probes distinguish band quality from instrument efficiency."""
import statistics


def verify_pair(state, first, second_hits, second_predictions, duration, earlier_lower):
    program = first["program"]
    mismatch = state.scoring.mismatch_multiplier
    if mismatch <= 0:
        return None
    expected = state.scoring.program_multipliers[program] / mismatch
    if abs(expected - 1) <= .05:
        return None  # The bonuses cannot discriminate matching from mismatching bands.
    band_floor = state.scoring.program_bands[program]
    efficiencies = []
    for target, score1 in first["hits"].items():
        score2 = second_hits.get(target, 0)
        if score1 <= 0 or score2 <= 0 or target not in second_predictions:
            continue
        if .0000005 / score1 + .0000005 / score2 > .005:
            continue
        i = state.index_of[target]
        model1 = first["models"][target]
        model2 = second_predictions[target].model
        observed = score1 / score2 * duration / first["duration"] * model2 / max(model1, 1e-9)
        if abs(observed / expected - 1) > .025:
            continue
        factor = score1 / max(1e-9, state.weight[i] * state.scoring.program_multipliers[program])
        if factor >= .90:
            continue  # Saturated scores cannot reveal the efficiency ratio.
        q_eff = factor * state.scoring.f0t0 / max(1e-9, state.flux[i] * first["duration"])
        efficiencies.append(q_eff / band_floor)
    if len(efficiencies) < 3:
        return None
    upper = statistics.median(efficiencies)
    # The matched band constrains sky quality. Compare with the historical
    # clean reference; multiple fields and a persistent drop are required by
    # Planner before treating this evidence as an instrument problem. A five
    # percent gap exceeds the paired-ratio tolerance; demanding a much larger
    # gap misses faults when the earlier clean lower bound is itself conservative.
    return upper if upper < earlier_lower * .95 else None
