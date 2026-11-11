"""Load the C split engines (split_exact.c) through ctypes.

Two engines live in that file, plus a third added in iteration 6a. The LIVE one is the sequential
decomposition (split_decomp_ev): the split is valued in the order the game
plays the hands, the hand being split maximising the total EV, the later
hands each maximising their own EV against the expected dealer
distribution. It is near-exact (the later hands' draws do not feed back
into the earlier hands' settlement -- about a hundredth at a full deck)
and fast enough to run at the table. The other is the joint recursion
(split_ev), which tracks both hands and the dealer together and is exact
-- it is the verification tool the decomposition is checked against, but
its state space explodes with resplits, so it is not the live engine.
Iteration 6a adds split_moments on the same joint recursion: the exact EV
and the exact VARIANCE of the split payoff, for Kelly sizing and risk of
ruin. Only the joint recursion can see the variance, because the two
hands share one dealer hand and their payoffs are correlated.

This module is the only bridge: it finds the shared library next to this
file, loads it once, and exposes both engines with the same meaning as the
old Python _split_odds (total EV of splitting a pair, per unit staked).

If the library is not built (run python build.py), available() is False
and both engines raise a RuntimeError that says so. They never silently
fall back to an approximation: a wrong split number is worse than no
split number. The slow pure-Python version of the decomposition is
archived, not imported -- see archive/split_sequential.py.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rules import Rules


def _library_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # Leading underscore: a split_exact.so would shadow this module itself
    # at import time (Python would try to load it as an extension module).
    name = "_split_exact.dylib" if sys.platform == "darwin" else "_split_exact.so"
    return os.path.join(here, name)


_lib: ctypes.CDLL | None | False = None


def _load() -> ctypes.CDLL | None:
    """Load the shared library once. None when it is not built."""
    global _lib
    if _lib is False:
        return None
    if _lib is None:
        try:
            lib = ctypes.CDLL(_library_path())
        except OSError:
            _lib = False
            return None
        _args = [
            ctypes.POINTER(ctypes.c_int),  # counts[10], by blackjack value
            ctypes.c_int,  # single_value: the pair's rank, blackjack value
            ctypes.c_int,  # single_is_ace: 0/1
            ctypes.c_int,  # up_value: dealer upcard, blackjack value
            ctypes.c_int,  # up_is_ace: 0/1
            ctypes.c_int,  # exclude_first: peek value ruled out, 0 = none
            ctypes.c_int,  # max_splits
            ctypes.c_int,  # double_after_split: 0/1
            ctypes.c_int,  # resplit_aces: 0/1
            ctypes.c_int,  # hit_split_aces: 0/1
            ctypes.c_int,  # dealer_hits_soft_17: 0/1
        ]
        lib.split_ev_exact.argtypes = _args
        lib.split_ev_exact.restype = ctypes.c_double
        lib.split_ev_decomp.argtypes = _args
        lib.split_ev_decomp.restype = ctypes.c_double
        lib.split_moments_exact.argtypes = _args + [
            ctypes.POINTER(ctypes.c_double),  # ev_out
            ctypes.POINTER(ctypes.c_double),  # var_out
        ]
        lib.split_moments_exact.restype = ctypes.c_int
        lib.split_moments_states.argtypes = []
        lib.split_moments_states.restype = ctypes.c_long
        lib.split_exact_joint_states.argtypes = []
        lib.split_exact_joint_states.restype = ctypes.c_long
        lib.split_exact_dealer_states.argtypes = []
        lib.split_exact_dealer_states.restype = ctypes.c_long
        lib.split_exact_dealer_lookups.argtypes = []
        lib.split_exact_dealer_lookups.restype = ctypes.c_long
        lib.split_exact_dealer_hits.argtypes = []
        lib.split_exact_dealer_hits.restype = ctypes.c_long
        lib.split_decomp_first_states.argtypes = []
        lib.split_decomp_first_states.restype = ctypes.c_long
        lib.split_decomp_tail_states.argtypes = []
        lib.split_decomp_tail_states.restype = ctypes.c_long
        lib.split_decomp_hand_states.argtypes = []
        lib.split_decomp_hand_states.restype = ctypes.c_long
        lib.split_decomp_states.argtypes = []
        lib.split_decomp_states.restype = ctypes.c_long
        _lib = lib
    return _lib  # type: ignore[return-value]


def available() -> bool:
    """True when the C helper is built and loadable."""
    return _load() is not None


def _call_engine(
    lib: ctypes.CDLL,
    fn_name: str,
    counts: tuple[int, ...],
    single_value: int,
    single_is_ace: bool,
    up_value: int,
    up_is_ace: bool,
    exclude_first: int | None,
    rules: Rules,
) -> float:
    """Call one C engine and fail loudly on a NaN (out of memory)."""
    arr = (ctypes.c_int * 10)(*counts)
    value = getattr(lib, fn_name)(
        arr,
        single_value,
        1 if single_is_ace else 0,
        up_value,
        1 if up_is_ace else 0,
        exclude_first or 0,
        rules.max_splits,
        1 if rules.double_after_split else 0,
        1 if rules.resplit_aces else 0,
        1 if rules.hit_split_aces else 0,
        1 if rules.dealer_hits_soft_17 else 0,
    )
    if math.isnan(value):
        raise RuntimeError(
            "the C split engine failed (out of memory?). Try again. If it "
            "persists, the machine cannot hold the state space."
        )
    return value


def _require_lib() -> ctypes.CDLL:
    lib = _load()
    if lib is None:
        raise RuntimeError(
            "split odds need the C helper, which is not built: "
            "run python build.py in the iteration_6a folder, then try again. "
            "The slow pure-Python version is archived at "
            "blackjack/archive/split_sequential.py and is not used at runtime."
        )
    return lib


def split_ev(
    counts: tuple[int, ...],
    single_value: int,
    single_is_ace: bool,
    up_value: int,
    up_is_ace: bool,
    exclude_first: int | None,
    rules: Rules,
) -> float:
    """The JOINT engine: the exact total EV of splitting a pair, per unit
    staked on the original hand. This is the verification tool, not the live
    engine -- it is exact, but its state space explodes with resplits.

    counts is the unseen composition (shoe plus the dealer's hole card),
    indexed by blackjack value - 1. exclude_first is the hole-card value
    the peek rules out (None when the upcard cannot make blackjack).
    Raises RuntimeError when the helper is not built.
    """
    return _call_engine(
        _require_lib(), "split_ev_exact", counts, single_value, single_is_ace,
        up_value, up_is_ace, exclude_first, rules,
    )


def split_decomp_ev(
    counts: tuple[int, ...],
    single_value: int,
    single_is_ace: bool,
    up_value: int,
    up_is_ace: bool,
    exclude_first: int | None,
    rules: Rules,
) -> float:
    """The DECOMPOSITION engine (the live one): the near-exact total EV of
    splitting a pair, per unit staked on the original hand. Same arguments
    and meaning as split_ev. The value agrees with the joint to about a
    hundredth at a full deck. Raises RuntimeError when the helper is not
    built.
    """
    return _call_engine(
        _require_lib(), "split_ev_decomp", counts, single_value,
        single_is_ace, up_value, up_is_ace, exclude_first, rules,
    )


def split_moments(
    counts: tuple[int, ...],
    single_value: int,
    single_is_ace: bool,
    up_value: int,
    up_is_ace: bool,
    exclude_first: int | None,
    rules: Rules,
) -> tuple[float, float]:
    """The JOINT engine with a second moment (iteration 6a): the exact total
    EV and the exact VARIANCE of splitting a pair, per unit staked on the
    original hand, under the EV-maximising policy. Same arguments and
    meaning as split_ev, plus the variance -- the number Kelly sizing
    needs, which only the joint recursion can see (the two hands share
    one dealer hand, so their payoffs are correlated). Raises RuntimeError
    when the helper is not built or the engine runs out of memory.
    """
    lib = _require_lib()
    arr = (ctypes.c_int * 10)(*counts)
    ev = ctypes.c_double()
    var = ctypes.c_double()
    ok = lib.split_moments_exact(
        arr,
        single_value,
        1 if single_is_ace else 0,
        up_value,
        1 if up_is_ace else 0,
        exclude_first or 0,
        rules.max_splits,
        1 if rules.double_after_split else 0,
        1 if rules.resplit_aces else 0,
        1 if rules.hit_split_aces else 0,
        1 if rules.dealer_hits_soft_17 else 0,
        ctypes.byref(ev),
        ctypes.byref(var),
    )
    if not ok or math.isnan(ev.value) or math.isnan(var.value):
        raise RuntimeError(
            "the C split moments engine failed (out of memory?). Try again. "
            "if it persists, the machine cannot hold the state space."
        )
    return ev.value, max(0.0, var.value)


def moments_states() -> int:
    """Distinct joint_moments states computed by the last split_moments() call."""
    lib = _load()
    return lib.split_moments_states() if lib is not None else 0


def joint_states() -> int:
    """Distinct joint states computed by the last split_ev() call."""
    lib = _load()
    return lib.split_exact_joint_states() if lib is not None else 0


def dealer_states() -> int:
    """Distinct dealer states computed by the last engine call (either one)."""
    lib = _load()
    return lib.split_exact_dealer_states() if lib is not None else 0


def dealer_cache_hit_rate() -> float:
    """Fraction of dealer cache lookups that hit, on the last engine call."""
    lib = _load()
    if lib is None:
        return 0.0
    lookups = lib.split_exact_dealer_lookups()
    if lookups == 0:
        return 0.0
    return lib.split_exact_dealer_hits() / lookups


def decomp_first_states() -> int:
    """Distinct decomp_first states from the last split_decomp_ev() call."""
    lib = _load()
    return lib.split_decomp_first_states() if lib is not None else 0


def decomp_tail_states() -> int:
    """Distinct decomp_tail states from the last split_decomp_ev() call."""
    lib = _load()
    return lib.split_decomp_tail_states() if lib is not None else 0


def decomp_hand_states() -> int:
    """Distinct decomp_hand_after states from the last split_decomp_ev() call."""
    lib = _load()
    return lib.split_decomp_hand_states() if lib is not None else 0


def decomp_states() -> int:
    """Total decomposition states from the last split_decomp_ev() call."""
    lib = _load()
    return lib.split_decomp_states() if lib is not None else 0
