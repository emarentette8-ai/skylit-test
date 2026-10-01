"""ATM-band event state machine (spec section 4) on completed 1-minute bar closes.

For strike K and close S_t:  x_t = 100 (S_t - K) / S_t.
  * armed      : an observed close with |x| >= 2b (side = sign x), and >= COOLDOWN
                 minutes since the previous entry at this strike.
  * ENTRY (A)  : first close with |x| <= b while armed. Approach direction is frozen:
                 s = +1 if armed from below (x < 0), s = -1 if from above.
  * BAND JUMP  : while armed, a close on the opposite side with |x| > b and no close
                 inside the band -> classified separately, never an entry.
Only closes are used (no intrabar ordering is invented). State resets every session.
"""
import numpy as np


def run_state_machine(x, minutes, b, arm_mult=2.0, cooldown=5):
    """x: (T, K) array of signed distances, minutes: (T,) int minutes since open.

    Returns list of (t_index, k_index, kind, s) with kind in {"entry", "jump"}.
    """
    T, K = x.shape
    side = np.zeros(K)                 # +1/-1 = armed from that side, 0 = unarmed
    last_entry = np.full(K, -10**9)
    out = []
    for t in range(T):
        xt = x[t]
        valid = ~np.isnan(xt)
        ext = valid & (np.abs(xt) >= arm_mult * b)
        inside = valid & (np.abs(xt) <= b)
        armed = side != 0
        # entries
        ent = armed & inside
        for k in np.flatnonzero(ent):
            out.append((t, k, "entry", -side[k]))
        # jumps: armed, now strictly beyond the band on the opposite side
        jmp = armed & valid & ~inside & (np.sign(xt) == -side)
        for k in np.flatnonzero(jmp):
            out.append((t, k, "jump", -side[k]))
        side[ent | jmp] = 0
        last_entry[ent] = minutes[t]
        # (re)arming from an exterior observation, respecting cooldown
        can_arm = ext & (minutes[t] - last_entry >= cooldown)
        side[can_arm] = np.sign(xt[can_arm])
    return out
