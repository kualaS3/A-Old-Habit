/* split_exact.c -- the split EV, in C. Two engines live in this file.
 *
 * Why this file exists, in short: splitting is the one spot in the game
 * where the hands cannot be valued one at a time. Two split hands share one
 * shrinking shoe and one dealer hand, so the exact value has to track both
 * hands and the dealer together, in a single recursion. That recursion is
 * the honest formulation -- and in Python it is unusable: the state space
 * runs into the millions, and Python's dictionaries, tuples and function
 * calls make each state so expensive that a full-deck spot takes minutes and
 * eats gigabytes (the pure-Python version of the decomposition below is kept
 * at archive/split_sequential.py, to show the attempt). In C the same
 * recursion is plain integer arrays and hash tables, so it runs in seconds
 * and fits in memory. Same maths, same answers -- see the iteration 5 README
 * for the measured numbers.
 *
 * THE LIVE ENGINE is the sequential decomposition (split_ev_decomp):
 * the split is valued in the order the game actually plays the hands. The
 * hand being split is played to maximise the TOTAL ev of every hand. The
 * hands still to come are valued by decomp_tail, each to maximise its own
 * ev, returning the expected dealer distribution after it -- so the first
 * hand's decisions do account for what its cards do to the second hand and
 * to the dealer. The one thing the later hands do not account for is the
 * reverse: a later hand's draws also move the dealer distribution the
 * earlier hands are settled against, and its decisions are not re-optimised
 * for that. That effect is second-order (about a hundredth at a full deck),
 * so the decomposition is near-exact -- and, unlike the joint recursion,
 * its state space stays small enough to run live: it does not track all the
 * hands jointly, so each extra hand adds states instead of multiplying them.
 *
 * THE VERIFICATION TOOL is the joint recursion (split_ev_exact): both hands
 * and the dealer in one recursion, every hand maximising the TOTAL ev. It is
 * the exact value, and it is what the decomposition is checked against --
 * but its state space explodes with resplits (389k states at one split,
 * 27M at two, out of memory at three), so it cannot be the live engine.
 * That explosion is also why the table caps splits at one by default (see
 * Rules.max_splits): the exact value cannot be computed with resplits, so
 * the default is the case the joint can verify.

 * Iteration 6a adds a third entry point on the same joint recursion:
 * split_moments_exact returns the EV and the VARIANCE of the split
 * payoff, by carrying a second moment through the recursion (see
 * joint_moments). The variance is what Kelly sizing needs, and only the
 * joint recursion can see it: the two hands share one dealer hand, so
 * their payoffs are correlated, and the decomposition settles each hand
 * against an expected dealer distribution and never sees the cross term.
 *
 *
 * What both compute: the total EV of splitting a pair, per 1 unit bet on
 * the original hand, under optimal play of every split hand. The dealer
 * plays once at the end from whatever cards are left, and the split hands
 * are settled against that one dealer hand. Doubles after a split, resplits,
 * restricted split aces (one card only) and the S17/H17 dealer rule are all
 * in here, following the table in Rules. One difference between the two:
 * the joint follows the game and lets a restricted ace pair resplit. The
 * decomposition stands restricted pairs, like the archived Python version
 * it is ported from -- which only matters when the split cap is raised
 * above one, since the default cap allows no resplit at all.
 *
 * How it is used: Python loads the compiled shared library through ctypes
 * (see split_exact.py) and calls split_ev_decomp() for the live odds and
 * split_ev_exact() to check them. If the library is not built, Python
 * raises a clear error telling you to run build.py -- it never silently
 * falls back to a wrong number. The slow Python version in archive/ is not
 * imported anywhere.
 *
 * Build: python build.py (needs gcc or cc). That produces split_exact.so
 * (or .dylib on macOS) next to this file. The .so is NOT committed to the
 * repo: it is platform-specific, so each checkout builds its own.
 *
 * The code is deliberately plain C99 with no dependencies beyond libc.
 * Memory is allocated per call and freed before returning. There are no
 * globals that survive a call except the statistics counters, which the
 * tests read through split_exact_joint_states(),
 * split_exact_dealer_states() and split_decomp_states().
 *
 * One design point worth knowing: the dealer distribution is memoised in
 * a DIRECT-MAPPED cache (one slot per hash, overwritten on collision),
 * not a growing table like the joint memo. The dealer is called with a
 * different counts composition at almost every leaf of the recursion, so a
 * growing dealer table would hold tens of millions of entries and eat the
 * machine -- that is what killed the first version of this file at full
 * deck. The direct-mapped cache is bounded (a few hundred MB at most),
 * stays correct when an entry is evicted (a miss just recomputes), and
 * still hits well because the recursion visits nearby leaves with nearby
 * counts. The joint and decomposition memos, by contrast, are proper
 * growing tables: their state counts are in the thousands to millions,
 * which fits.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Rules: plain ints, 0/1 for the toggles. The caller (split_exact.py)
 * fills these in from the table's Rules object.                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int max_splits;         /* splits allowed per round */
    int double_after_split; /* 0/1 */
    int resplit_aces;       /* 0/1 */
    int hit_split_aces;     /* 0/1 */
    int dealer_hits_soft_17;/* 0/1 */
    int up_value;           /* dealer upcard, blackjack value 1..10 */
    int up_is_ace;          /* 0/1 */
    int exclude_first;      /* peek: hole-card value ruled out, 0 = none */
} Rules;

/* ------------------------------------------------------------------ */
/* Hands. A hand is fully described, for the future, by five numbers:
 * its hard total, whether it holds an ace counted softly, how many cards
 * it has (doubling and resplitting are two-card moves), whether it is
 * restricted (a split ace that takes one card only), and -- for a two-card
 * hand -- the rank of the pair, or 0 when the two cards differ. The exact
 * card values do not matter beyond those five numbers, which is what makes
 * the state small enough to memoise. A hand never holds more than 21 cards:
 * its hard total would pass 21 first.                                */
/* ------------------------------------------------------------------ */

#define MAX_HANDS 4   /* one split + three resplits */
#define MAX_CARDS 21

typedef struct {
    uint8_t hard;       /* hard total, aces counted as 1 */
    uint8_t ace;        /* 0/1: holds an ace that could count as 11 */
    uint8_t n;          /* cards in the hand */
    uint8_t restricted; /* 0/1: a split ace, one card only */
    uint8_t pair_rank;  /* the rank when the two cards are a pair, else 0 */
} HandState;

/* A finished hand: its best total and whether its stake was doubled. */
typedef struct {
    uint8_t total;  /* best total, 2..21 */
    uint8_t stake2; /* 0 = single stake, 1 = doubled */
} FinState;

/* ------------------------------------------------------------------ */
/* Small helpers.                                                      */
/* ------------------------------------------------------------------ */

static int best_total(int hard, int ace) {
    return (ace && hard + 10 <= 21) ? hard + 10 : hard;
}

static int sum_counts(const int *counts) {
    int n = 0;
    for (int i = 0; i < 10; i++) n += counts[i];
    return n;
}

/* The dealer must hit on this (best, soft) under the table's rule. */
static int dealer_must_hit(int best, int soft, const Rules *r) {
    return best < 17 || (best == 17 && soft && r->dealer_hits_soft_17);
}

/* ------------------------------------------------------------------ */
/* The dealer distribution. A distribution is six doubles: the chance
 * the dealer finishes on 17, 18, 19, 20, 21, and the chance he busts.
 * Nothing else is possible -- the dealer never stands below 17 -- so
 * six numbers are the whole distribution.                           */
/* ------------------------------------------------------------------ */

#define DIST_LEN 6  /* totals 17..21, then bust */

static double dist_at(const double *d, int total) {
    if (total < 17) return 0.0;
    if (total <= 21) return d[total - 17];
    return d[5]; /* bust */
}

/* Expected payoff, per unit staked, of standing on `total` against the
 * dealer distribution: win when the dealer busts or finishes below you,
 * lose when he finishes above you, push on a tie (worth 0).
 * The distribution is stored as d[0..4] = totals 17..21, d[5] = bust, so
 * a dealer total u lives at d[u - 17] -- and only for u >= 17. The lose
 * loop must start at 17, not at total + 1: for a low total (say 13) the
 * dealer totals 14..16 do not exist, and indexing them would read
 * before the array. That one-line mistake was a real bug in the first
 * version of this file: it read stack garbage as probabilities, which
 * is why the split values came out wrong (and, in one build, absurd --
 * 10^300) while the dealer distributions themselves all checked out. */
static double payoff_vs_dist(const double *d, int total) {
    double win = d[5]; /* dealer busts */
    for (int u = 17; u < total && u <= 21; u++) win += d[u - 17];
    double lose = 0.0;
    int lo = total + 1 > 17 ? total + 1 : 17; /* dealer totals start at 17 */
    for (int u = lo; u <= 21; u++) lose += d[u - 17];
    return win - lose;
}

/* ------------------------------------------------------------------ */
/* Hash tables. Two of them, with different jobs and different rules.
 * The joint memo is open-addressed with linear probing and grows by
 * doubling when it passes 70% full -- its state count is in the
 * millions, which fits. The dealer cache is DIRECT-MAPPED (one slot per
 * hash, overwritten on collision) and never grows: the dealer is called
 * with a different counts composition at almost every leaf, so a growing
 * dealer table would hold tens of millions of entries and eat the
 * machine. Eviction stays correct -- a miss just recomputes. Both are
 * allocated per split_ev_exact() call and freed before it returns.    */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t w[3];
} JointKey;

typedef struct {
    JointKey key;
    double value;
} JointSlot;

typedef struct {
    JointSlot *slots;
    uint8_t *used;
    size_t cap; /* always a power of two */
    size_t len;
} JointTable;

typedef struct {
    uint64_t counts; /* counts packed, 6 bits each */
    uint32_t ha;     /* dealer hard (8 bits) and ace (1 bit) */
} DealerKey;

typedef struct {
    DealerKey key;
    double dist[DIST_LEN];
} DealerSlot;

/* Direct-mapped: cap is fixed, slots are overwritten on collision, and
 * there is no `used` array and no growth. `len` counts distinct keys
 * currently resident (approximately: a collision can evict one). */
typedef struct {
    DealerSlot *slots;
    size_t cap; /* always a power of two */
    size_t len;
} DealerTable;

/* A decomposition memo value: per-hand EVs (up to four hands) plus the
 * expected payoff of a finished hand standing on each total 2..21 (index
 * total - 2). decomp_hand_after fills evs[0] with its own EV and n = 1.
 * decomp_tail fills evs[0..n-1] with the per-hand EVs and n = the hand
 * count. The payoffs are what an earlier finished hand needs to settle
 * against a dealer it has not seen yet -- a payoff is linear in the dealer
 * distribution, so the expected payoff against the expected distribution is
 * the expected payoff. The value is wide (25 doubles) because the payoffs
 * travel with every state. See the README for the measured state counts. */
typedef struct {
    double evs[MAX_HANDS];
    double payoffs[20]; /* totals 2..21, index total - 2 */
    int n;              /* hands in evs */
} DecompVal;

typedef struct {
    uint64_t w[3];
} DecompKey;

typedef struct {
    DecompKey key;
    DecompVal value;
} DecompSlot;

/* Growing, open-addressed, like the joint table: the decomposition's
 * state counts are in the thousands to millions, which fits. */
typedef struct {
    DecompSlot *slots;
    uint8_t *used;
    size_t cap; /* always a power of two */
    size_t len;
} DecompTable;

/* A moments memo value: the expected value and the expected square of the
 * total payoff from a joint state, under the EV-maximising policy. The
 * variance is e2 - ev * ev. Two doubles, because the second moment is
 * what the first moment cannot see: two split hands share one dealer
 * hand, so their payoffs are correlated, and the variance of their sum
 * needs the cross term -- which only a joint recursion can carry. */
typedef struct {
    double ev;
    double e2;
} MomentsVal;

typedef struct {
    JointKey key;
    MomentsVal value;
} MomentsSlot;

/* Growing, open-addressed, like the joint table. */
typedef struct {
    MomentsSlot *slots;
    uint8_t *used;
    size_t cap; /* always a power of two */
    size_t len;
} MomentsTable;

/* Pack the unseen counts (each 0..32, so 6 bits) into one word. */
static uint64_t pack_counts(const int *counts) {
    uint64_t w = 0;
    for (int i = 0; i < 10; i++) w |= (uint64_t)(counts[i] & 63) << (6 * i);
    return w;
}

/* Pack one hand into 16 bits: hard 5, ace 1, n 5, restricted 1, pair 4. */
static uint16_t pack_hand(const HandState *h) {
    return (uint16_t)(
        (h->hard & 31) | ((h->ace & 1) << 5) | ((h->n & 31) << 6) |
        ((h->restricted & 1) << 11) | ((h->pair_rank & 15) << 12));
}

/* Pack the whole joint state into 24 bytes. Pending hands keep their
 * order (hand 0 is the one being played). Finished hands are a multiset,
 * so the caller keeps them sorted by (total, stake) and the key is
 * canonical. */
static JointKey pack_joint(const int *counts, const HandState *pending,
                           int n_pending, const FinState *finished,
                           int n_finished, int splits_used) {
    JointKey k;
    k.w[0] = pack_counts(counts);
    uint64_t p = 0;
    for (int i = 0; i < MAX_HANDS; i++) {
        uint16_t h = (i < n_pending) ? pack_hand(&pending[i]) : 0;
        p |= (uint64_t)h << (16 * i);
    }
    k.w[1] = p;
    uint64_t m = 0;
    for (int i = 0; i < MAX_HANDS; i++) {
        uint32_t f = 0;
        if (i < n_finished)
            f = (finished[i].total & 31) | ((finished[i].stake2 & 1) << 5);
        m |= (uint64_t)f << (6 * i);
    }
    m |= (uint64_t)(n_pending & 7) << 24;
    m |= (uint64_t)(n_finished & 7) << 27;
    m |= (uint64_t)(splits_used & 3) << 30;
    k.w[2] = m;
    return k;
}

static int joint_key_equal(const JointKey *a, const JointKey *b) {
    return a->w[0] == b->w[0] && a->w[1] == b->w[1] && a->w[2] == b->w[2];
}

/* Pack a decomposition state into 24 bytes: the unseen counts, the hands
 * in order (hand 0 is the one being played), how many hands there are, and
 * how many splits have been used. For decomp_hand_after the state is a
 * single hand, packed in the first slot, with n_hands = 1 and
 * splits_used = 0 -- a hand is fully described by (hard, ace, n,
 * restricted, pair_rank), the same five numbers as in the joint, so one
 * packed hand is the whole key. */
static DecompKey pack_decomp(const int *counts, const HandState *hands,
                             int n_hands, int splits_used) {
    DecompKey k;
    k.w[0] = pack_counts(counts);
    uint64_t p = 0;
    for (int i = 0; i < MAX_HANDS; i++) {
        uint16_t h = (i < n_hands) ? pack_hand(&hands[i]) : 0;
        p |= (uint64_t)h << (16 * i);
    }
    k.w[1] = p;
    k.w[2] = (uint64_t)(n_hands & 7) | ((uint64_t)(splits_used & 3) << 3);
    return k;
}

static int decomp_key_equal(const DecompKey *a, const DecompKey *b) {
    return a->w[0] == b->w[0] && a->w[1] == b->w[1] && a->w[2] == b->w[2];
}

/* FNV-1a over the key bytes: simple, fast enough, well distributed. */
static uint64_t hash_bytes(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int table_init_joint(JointTable *t, size_t cap) {
    t->slots = (JointSlot *)calloc(cap, sizeof(JointSlot));
    t->used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!t->slots || !t->used) {
        free(t->slots);
        free(t->used);
        return 0;
    }
    t->cap = cap;
    t->len = 0;
    return 1;
}

static void table_free_joint(JointTable *t) {
    free(t->slots);
    free(t->used);
    t->slots = NULL;
    t->used = NULL;
    t->cap = t->len = 0;
}

static int table_grow_joint(JointTable *t) {
    size_t ncap = t->cap ? t->cap * 2 : 65536;
    JointSlot *nslots = (JointSlot *)calloc(ncap, sizeof(JointSlot));
    uint8_t *nused = (uint8_t *)calloc(ncap, sizeof(uint8_t));
    if (!nslots || !nused) {
        free(nslots);
        free(nused);
        return 0;
    }
    for (size_t i = 0; i < t->cap; i++) {
        if (!t->used[i]) continue;
        size_t j = (size_t)(hash_bytes(&t->slots[i].key, sizeof(JointKey)) &
                            (ncap - 1));
        while (nused[j]) j = (j + 1) & (ncap - 1);
        nslots[j] = t->slots[i];
        nused[j] = 1;
    }
    free(t->slots);
    free(t->used);
    t->slots = nslots;
    t->used = nused;
    t->cap = ncap;
    return 1;
}

/* Returns 1 and fills *out when the key is present, else 0. */
static int table_get_joint(JointTable *t, const JointKey *key, double *out) {
    if (t->cap == 0) return 0;
    size_t i = (size_t)(hash_bytes(key, sizeof(JointKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (joint_key_equal(&t->slots[i].key, key)) {
            *out = t->slots[i].value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    return 0;
}

/* Stores the key/value pair. Returns 0 on allocation failure. */
static int table_put_joint(JointTable *t, const JointKey *key, double value) {
    if (t->cap == 0 || (t->len + 1) * 10 > t->cap * 7) {
        if (!table_grow_joint(t)) return 0;
    }
    size_t i = (size_t)(hash_bytes(key, sizeof(JointKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (joint_key_equal(&t->slots[i].key, key)) {
            t->slots[i].value = value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    t->slots[i].key = *key;
    t->slots[i].value = value;
    t->used[i] = 1;
    t->len++;
    return 1;
}

static int table_init_dealer(DealerTable *t, size_t cap) {
    t->slots = (DealerSlot *)calloc(cap, sizeof(DealerSlot));
    if (!t->slots) return 0;
    /* A zeroed key (counts == 0, ha == 0) can never be a real dealer
     * state -- the dealer always holds at least the upcard -- so a slot
     * whose key is all zero doubles as the "empty" marker. */
    t->cap = cap;
    t->len = 0;
    return 1;
}

static void table_free_dealer(DealerTable *t) {
    free(t->slots);
    t->slots = NULL;
    t->cap = t->len = 0;
}

/* One slot, by hash. A hit needs the full key to match. Anything else is
 * a miss, and a put simply overwrites whatever is there. */
/* Hash a dealer key over its meaningful bytes only. DealerKey has four
 * padding bytes after ha, and they are uninitialised stack garbage: the
 * old code hashed sizeof(DealerKey), so put and get could hash the same
 * logical key to different slots and the cache would mostly miss -- slow,
 * not wrong (a hit still needed the full key to match). Hashing only
 * counts (8 bytes) and ha (4 bytes) makes the hash deterministic. The key
 * is also built zeroed (see dealer_dist), so the padding never matters. */
static uint64_t hash_dealer_key(const DealerKey *k) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)&k->counts;
    for (int i = 0; i < 8; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    p = (const uint8_t *)&k->ha;
    for (int i = 0; i < 4; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int table_get_dealer(DealerTable *t, const DealerKey *key,
                            double *out) {
    if (t->cap == 0) return 0;
    size_t i = (size_t)(hash_dealer_key(key) & (t->cap - 1));
    if (t->slots[i].key.counts == key->counts &&
        t->slots[i].key.ha == key->ha &&
        (key->counts != 0 || key->ha != 0)) {
        memcpy(out, t->slots[i].dist, sizeof(double) * DIST_LEN);
        return 1;
    }
    return 0;
}

static void table_put_dealer(DealerTable *t, const DealerKey *key,
                             const double *dist) {
    if (t->cap == 0) return;
    size_t i = (size_t)(hash_dealer_key(key) & (t->cap - 1));
    if (t->slots[i].key.counts != key->counts ||
        t->slots[i].key.ha != key->ha)
        t->len++;
    t->slots[i].key = *key;
    memcpy(t->slots[i].dist, dist, sizeof(double) * DIST_LEN);
}

/* The decomposition's memo tables: same shape as the joint table (open-
 * addressed, linear probing, doubling past 70% full), with a wider value.
 * Two instances are used: one for decomp_hand_after, one for decomp_tail.
 * decomp_first reuses the joint table (its value is a single double, and
 * its key is pack_joint with no finished hands). */

static int table_init_decomp(DecompTable *t, size_t cap) {
    t->slots = (DecompSlot *)calloc(cap, sizeof(DecompSlot));
    t->used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!t->slots || !t->used) {
        free(t->slots);
        free(t->used);
        return 0;
    }
    t->cap = cap;
    t->len = 0;
    return 1;
}

static void table_free_decomp(DecompTable *t) {
    free(t->slots);
    free(t->used);
    t->slots = NULL;
    t->used = NULL;
    t->cap = t->len = 0;
}

static int table_grow_decomp(DecompTable *t) {
    size_t ncap = t->cap ? t->cap * 2 : 65536;
    DecompSlot *nslots = (DecompSlot *)calloc(ncap, sizeof(DecompSlot));
    uint8_t *nused = (uint8_t *)calloc(ncap, sizeof(uint8_t));
    if (!nslots || !nused) {
        free(nslots);
        free(nused);
        return 0;
    }
    for (size_t i = 0; i < t->cap; i++) {
        if (!t->used[i]) continue;
        size_t j = (size_t)(hash_bytes(&t->slots[i].key, sizeof(DecompKey)) &
                            (ncap - 1));
        while (nused[j]) j = (j + 1) & (ncap - 1);
        nslots[j] = t->slots[i];
        nused[j] = 1;
    }
    free(t->slots);
    free(t->used);
    t->slots = nslots;
    t->used = nused;
    t->cap = ncap;
    return 1;
}

/* Returns 1 and fills *out when the key is present, else 0. */
static int table_get_decomp(DecompTable *t, const DecompKey *key,
                            DecompVal *out) {
    if (t->cap == 0) return 0;
    size_t i = (size_t)(hash_bytes(key, sizeof(DecompKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (decomp_key_equal(&t->slots[i].key, key)) {
            *out = t->slots[i].value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    return 0;
}

/* Stores the key/value pair. Returns 0 on allocation failure. */
static int table_put_decomp(DecompTable *t, const DecompKey *key,
                            const DecompVal *value) {
    if (t->cap == 0 || (t->len + 1) * 10 > t->cap * 7) {
        if (!table_grow_decomp(t)) return 0;
    }
    size_t i = (size_t)(hash_bytes(key, sizeof(DecompKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (decomp_key_equal(&t->slots[i].key, key)) {
            t->slots[i].value = *value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    t->slots[i].key = *key;
    t->slots[i].value = *value;
    t->used[i] = 1;
    t->len++;
    return 1;
}

/* The moments memo tables: same shape as the joint table (open-addressed,
 * linear probing, doubling past 70% full), with a two-double value. One
 * instance is used, by joint_moments (iteration 6a). */

static int table_init_moments(MomentsTable *t, size_t cap) {
    t->slots = (MomentsSlot *)calloc(cap, sizeof(MomentsSlot));
    t->used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!t->slots || !t->used) {
        free(t->slots);
        free(t->used);
        return 0;
    }
    t->cap = cap;
    t->len = 0;
    return 1;
}

static void table_free_moments(MomentsTable *t) {
    free(t->slots);
    free(t->used);
    t->slots = NULL;
    t->used = NULL;
    t->cap = t->len = 0;
}

static int table_grow_moments(MomentsTable *t) {
    size_t ncap = t->cap ? t->cap * 2 : 65536;
    MomentsSlot *nslots = (MomentsSlot *)calloc(ncap, sizeof(MomentsSlot));
    uint8_t *nused = (uint8_t *)calloc(ncap, sizeof(uint8_t));
    if (!nslots || !nused) {
        free(nslots);
        free(nused);
        return 0;
    }
    for (size_t i = 0; i < t->cap; i++) {
        if (!t->used[i]) continue;
        size_t j = (size_t)(hash_bytes(&t->slots[i].key, sizeof(JointKey)) &
                            (ncap - 1));
        while (nused[j]) j = (j + 1) & (ncap - 1);
        nslots[j] = t->slots[i];
        nused[j] = 1;
    }
    free(t->slots);
    free(t->used);
    t->slots = nslots;
    t->used = nused;
    t->cap = ncap;
    return 1;
}

/* Returns 1 and fills *out when the key is present, else 0. */
static int table_get_moments(MomentsTable *t, const JointKey *key,
                             MomentsVal *out) {
    if (t->cap == 0) return 0;
    size_t i = (size_t)(hash_bytes(key, sizeof(JointKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (joint_key_equal(&t->slots[i].key, key)) {
            *out = t->slots[i].value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    return 0;
}

/* Stores the key/value pair. Returns 0 on allocation failure. */
static int table_put_moments(MomentsTable *t, const JointKey *key,
                             const MomentsVal *value) {
    if (t->cap == 0 || (t->len + 1) * 10 > t->cap * 7) {
        if (!table_grow_moments(t)) return 0;
    }
    size_t i = (size_t)(hash_bytes(key, sizeof(JointKey)) & (t->cap - 1));
    while (t->used[i]) {
        if (joint_key_equal(&t->slots[i].key, key)) {
            t->slots[i].value = *value;
            return 1;
        }
        i = (i + 1) & (t->cap - 1);
    }
    t->slots[i].key = *key;
    t->slots[i].value = *value;
    t->used[i] = 1;
    t->len++;
    return 1;
}

/* ------------------------------------------------------------------ */
/* The engine: the rules, both tables, and the two statistics counters
 * the tests read back.                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    Rules rules;
    JointTable joint;
    DealerTable dealer;
    DecompTable hand; /* decomp_hand_after memo */
    DecompTable tail; /* decomp_tail memo */
    MomentsTable moments; /* joint_moments memo (iteration 6a) */
    long joint_states;  /* distinct joint states computed */
    long dealer_states; /* distinct dealer states computed */
    long dealer_lookups; /* dealer cache lookups */
    long dealer_hits;    /* dealer cache hits */
    long decomp_first_states; /* distinct decomp_first states computed */
    long decomp_tail_states;  /* distinct decomp_tail states computed */
    long decomp_hand_states;  /* distinct decomp_hand_after states computed */
    long moments_states;  /* distinct joint_moments states computed */
    int failed;         /* set when an allocation fails */
} Engine;

/* The dealer distribution from here, memoised. The hole card has already
 * been drawn at this point (exclude_first only ever applies to the
 * outermost call, which uses dealer_dist_excl below), so this version is
 * unconditioned. */
static void dealer_dist(Engine *e, const int *counts, int hard, int ace,
                        double *out) {
    for (int i = 0; i < DIST_LEN; i++) out[i] = 0.0;
    if (hard > 21) {
        out[5] = 1.0; /* bust */
        return;
    }
    int best = best_total(hard, ace);
    int soft = (best != hard);
    int n = sum_counts(counts);
    if (n == 0 || !dealer_must_hit(best, soft, &e->rules)) {
        out[best - 17] = 1.0; /* stands: best is 17..21 here */
        return;
    }
    DealerKey key;
    memset(&key, 0, sizeof(key)); /* zero the padding: never hash it */
    key.counts = pack_counts(counts);
    key.ha = (uint32_t)((hard & 255) | ((ace & 1) << 8));
    e->dealer_lookups++;
    if (table_get_dealer(&e->dealer, &key, out)) {
        e->dealer_hits++;
        return;
    }
    double acc[DIST_LEN];
    for (int i = 0; i < DIST_LEN; i++) acc[i] = 0.0;
    for (int v = 1; v <= 10; v++) {
        int c = counts[v - 1];
        if (c == 0) continue;
        int nxt[10];
        memcpy(nxt, counts, sizeof(int) * 10);
        nxt[v - 1]--;
        double sub[DIST_LEN];
        dealer_dist(e, nxt, hard + v, ace || v == 1, sub);
        double p = (double)c / (double)n;
        for (int i = 0; i < DIST_LEN; i++) acc[i] += p * sub[i];
    }
    memcpy(out, acc, sizeof(double) * DIST_LEN);
    /* Direct-mapped: this may evict another entry, which is fine -- a
     * later miss just recomputes. The store cannot fail. */
    table_put_dealer(&e->dealer, &key, out);
    e->dealer_states++;
}

/* The outermost dealer call: the first card drawn is the hole card, and
 * the peek has already ruled out one value for it (exclude_first), so the
 * draw is renormalised without that value. Deeper draws are unconditioned
 * and go through the memoised dealer_dist above. */
static void dealer_dist_excl(Engine *e, const int *counts, int hard, int ace,
                            int excl, double *out) {
    for (int i = 0; i < DIST_LEN; i++) out[i] = 0.0;
    if (hard > 21) {
        out[5] = 1.0;
        return;
    }
    int best = best_total(hard, ace);
    int soft = (best != hard);
    int n = sum_counts(counts);
    if (n == 0 || !dealer_must_hit(best, soft, &e->rules)) {
        out[best - 17] = 1.0;
        return;
    }
    if (excl && n - counts[excl - 1] == 0) excl = 0; /* nothing to condition on */
    int denom = excl ? n - counts[excl - 1] : n;
    for (int v = 1; v <= 10; v++) {
        if (v == excl) continue;
        int c = counts[v - 1];
        if (c == 0) continue;
        int nxt[10];
        memcpy(nxt, counts, sizeof(int) * 10);
        nxt[v - 1]--;
        double sub[DIST_LEN];
        dealer_dist(e, nxt, hard + v, ace || v == 1, sub);
        double p = (double)c / (double)denom;
        for (int i = 0; i < DIST_LEN; i++) out[i] += p * sub[i];
    }
}

/* ------------------------------------------------------------------ */
/* The joint recursion.                                                */
/*                                                                     */
/* joint(counts, pending, finished, splits_used) is the total EV of
 * every hand -- the ones still to play and the ones already finished --
 * per unit staked on the original hand. Hand 0 of pending is the one
 * being played. It chooses the action that maximises the TOTAL ev, not
 * just its own, because one player owns all the hands. Finished hands
 * are settled against the dealer distribution once every hand has
 * played. A hand that busts loses its stake at once and leaves the
 * table, so its loss is added by its parent, not carried in the state.
 *                                                                     */
/* The one rule worth a comment: a restricted hand (a split ace that took
 * its one card) cannot hit or double -- but it CAN resplit, when its one
 * card was another ace and the table allows resplitting aces. That is
 * the game as played (see Round.can_split). The archived Python version
 * stood restricted pairs instead, which is why its ace-split value is a
 * touch lower.                                                       */
/* ------------------------------------------------------------------ */

static void sort_finished(FinState *f, int n) {
    for (int i = 1; i < n; i++) {
        FinState x = f[i];
        int j = i - 1;
        while (j >= 0 &&
               (f[j].total > x.total ||
                (f[j].total == x.total && f[j].stake2 > x.stake2))) {
            f[j + 1] = f[j];
            j--;
        }
        f[j + 1] = x;
    }
}

static double joint(Engine *e, const int *counts, const HandState *pending,
                    int n_pending, const FinState *finished, int n_finished,
                    int splits_used) {
    JointKey key = pack_joint(counts, pending, n_pending, finished, n_finished,
                              splits_used);
    double cached;
    if (table_get_joint(&e->joint, &key, &cached)) return cached;

    double result;
    if (n_pending == 0) {
        /* Every hand has played. If nothing finished -- every hand bust --
         * the dealer never plays and the total is 0 here: the bust losses
         * were already added by the parents. */
        if (n_finished == 0) {
            result = 0.0;
        } else {
            double dist[DIST_LEN];
            dealer_dist_excl(e, counts, e->rules.up_value, e->rules.up_is_ace,
                             e->rules.exclude_first, dist);
            result = 0.0;
            for (int i = 0; i < n_finished; i++) {
                double stake = finished[i].stake2 ? 2.0 : 1.0;
                result += stake * payoff_vs_dist(dist, finished[i].total);
            }
        }
        if (!e->failed) {
            if (!table_put_joint(&e->joint, &key, result)) e->failed = 1;
            else e->joint_states++;
        }
        return result;
    }

    const Rules *r = &e->rules;
    HandState h0 = pending[0];
    int best = best_total(h0.hard, h0.ace);
    int n = sum_counts(counts);

    /* STAND: hand 0 finishes on its best total, then the rest play. */
    FinState fin[MAX_HANDS];
    if (n_finished > 0)
        memcpy(fin, finished, sizeof(FinState) * (size_t)n_finished);
    fin[n_finished].total = (uint8_t)best;
    fin[n_finished].stake2 = 0;
    sort_finished(fin, n_finished + 1);
    double best_ev = joint(e, counts, pending + 1, n_pending - 1, fin,
                           n_finished + 1, splits_used);

    if (!h0.restricted && best < 21 && n > 0) {
        /* HIT: draw one card and keep playing this hand. */
        double ev_hit = 0.0;
        for (int v = 1; v <= 10; v++) {
            int c = counts[v - 1];
            if (c == 0) continue;
            int nxt[10];
            memcpy(nxt, counts, sizeof(int) * 10);
            nxt[v - 1]--;
            HandState nh = h0;
            nh.hard = (uint8_t)(h0.hard + v);
            nh.ace = (uint8_t)(h0.ace || v == 1);
            nh.n = (uint8_t)(h0.n + 1);
            nh.pair_rank = 0;
            double sub;
            if (nh.hard > 21) {
                sub = -1.0 + joint(e, nxt, pending + 1, n_pending - 1,
                                   finished, n_finished, splits_used);
            } else {
                HandState rest[MAX_HANDS];
                rest[0] = nh;
                memcpy(rest + 1, pending + 1,
                       sizeof(HandState) * (size_t)(n_pending - 1));
                sub = joint(e, nxt, rest, n_pending, finished, n_finished,
                            splits_used);
            }
            ev_hit += ((double)c / (double)n) * sub;
        }
        if (ev_hit > best_ev) best_ev = ev_hit;

        /* DOUBLE: exactly one card, on a doubled stake -- only on two
         * cards, and only when the table allows doubling after a split. */
        if (h0.n == 2 && r->double_after_split) {
            double ev_dbl = 0.0;
            for (int v = 1; v <= 10; v++) {
                int c = counts[v - 1];
                if (c == 0) continue;
                int nxt[10];
                memcpy(nxt, counts, sizeof(int) * 10);
                nxt[v - 1]--;
                int new_hard = h0.hard + v;
                double sub;
                if (new_hard > 21) {
                    sub = -2.0 + joint(e, nxt, pending + 1, n_pending - 1,
                                       finished, n_finished, splits_used);
                } else {
                    FinState f2[MAX_HANDS];
                    if (n_finished > 0)
                        memcpy(f2, finished,
                               sizeof(FinState) * (size_t)n_finished);
                    f2[n_finished].total =
                        (uint8_t)best_total(new_hard, h0.ace || v == 1);
                    f2[n_finished].stake2 = 1;
                    sort_finished(f2, n_finished + 1);
                    sub = joint(e, nxt, pending + 1, n_pending - 1, f2,
                                n_finished + 1, splits_used);
                }
                ev_dbl += ((double)c / (double)n) * sub;
            }
            if (ev_dbl > best_ev) best_ev = ev_dbl;
        }
    }

    /* RESPLIT: hand 0 is a pair under the cap. Aces need the resplit_aces
     * toggle. This branch is NOT gated on restricted: a restricted ace
     * pair is exactly the hand that resplits in the real game. The two new
     * hands are dealt their second cards first (casino order), as an
     * unordered pair -- the hands are symmetric, so (v1, v2) and (v2, v1)
     * are one outcome with twice the probability. */
    if (h0.n == 2 && h0.pair_rank != 0 && splits_used < r->max_splits &&
        !(h0.pair_rank == 1 && !r->resplit_aces) && n >= 2) {
        int single = h0.pair_rank;
        int new_restricted = (single == 1 && !r->hit_split_aces);
        double ev_resplit = 0.0;
        for (int v1 = 1; v1 <= 10; v1++) {
            int c1 = counts[v1 - 1];
            if (c1 == 0) continue;
            int mid[10];
            memcpy(mid, counts, sizeof(int) * 10);
            mid[v1 - 1]--;
            for (int v2 = v1; v2 <= 10; v2++) {
                int c2 = mid[v2 - 1];
                if (c2 == 0) continue;
                int nxt[10];
                memcpy(nxt, mid, sizeof(int) * 10);
                nxt[v2 - 1]--;
                HandState h1, h2;
                h1.hard = (uint8_t)(single + v1);
                h1.ace = (uint8_t)(single == 1 || v1 == 1);
                h1.n = 2;
                h1.restricted = (uint8_t)new_restricted;
                h1.pair_rank = (uint8_t)(v1 == single ? single : 0);
                h2.hard = (uint8_t)(single + v2);
                h2.ace = (uint8_t)(single == 1 || v2 == 1);
                h2.n = 2;
                h2.restricted = (uint8_t)new_restricted;
                h2.pair_rank = (uint8_t)(v2 == single ? single : 0);
                HandState rest[MAX_HANDS];
                rest[0] = h1;
                rest[1] = h2;
                memcpy(rest + 2, pending + 1,
                       sizeof(HandState) * (size_t)(n_pending - 1));
                double p = ((double)c1 / (double)n) *
                           ((double)c2 / (double)(n - 1));
                if (v2 != v1) p *= 2.0;
                ev_resplit += p * joint(e, nxt, rest, n_pending + 1, finished,
                                        n_finished, splits_used + 1);
            }
        }
        if (ev_resplit > best_ev) best_ev = ev_resplit;
    }

    result = best_ev;
    if (!e->failed) {
        if (!table_put_joint(&e->joint, &key, result)) e->failed = 1;
        else e->joint_states++;
    }
    return result;
}

/* ------------------------------------------------------------------ */
/* The joint moments recursion (iteration 6a).                         */
/*                                                                     */
/* joint_moments is joint() carrying a second moment: at every state   */
/* it returns (ev, e2), the expected value and the expected square of  */
/* the total payoff of every hand from here on, per unit staked on     */
/* the original hand, under the EV-maximising policy. The action at    */
/* each state is the one that maximises EV -- the same choice joint()  */
/* makes -- so the ev here agrees with joint() and the e2 is the       */
/* second moment under that policy.                                    */
/*                                                                     */
/* Why the second moment needs the joint recursion: the two split      */
/* hands are settled against ONE dealer hand, so their payoffs are     */
/* correlated and Var(X1 + X2) = Var(X1) + Var(X2) + 2 Cov(X1, X2).   */
/* The decomposition cannot see the covariance -- it settles each      */
/* hand against an expected dealer distribution -- but the joint leaf  */
/* can: it holds the whole dealer distribution, so it computes the     */
/* payoff of every hand against each dealer outcome and squares the    */
/* sum. A bust is a constant added to the continuation, and a constant */
/* shifts the second moment by E[(c + Y)^2] = E[Y^2] + 2c E[Y] + c^2. */
/* A mixture over drawn cards averages both moments, since the branch  */
/* is one outcome path.                                                */
/* ------------------------------------------------------------------ */

static void joint_moments(Engine *e, const int *counts, const HandState *pending,
                    int n_pending, const FinState *finished, int n_finished,
                    int splits_used, double *ev_out, double *e2_out) {
    if (e->failed) {
        *ev_out = 0.0;
        *e2_out = 0.0;
        return;
    }
    JointKey key = pack_joint(counts, pending, n_pending, finished, n_finished,
                              splits_used);
    MomentsVal cached;
    if (table_get_moments(&e->moments, &key, &cached)) {
        *ev_out = cached.ev;
        *e2_out = cached.e2;
        return;
    }

    double ev, e2;
    if (n_pending == 0) {
        /* Every hand has played. The dealer plays once, from these counts,
         * and every finished hand is settled against the one dealer hand he
         * actually gets -- so the squared total is the square of the sum,
         * cross term included, not the sum of the squares. */
        if (n_finished == 0) {
            ev = 0.0;
            e2 = 0.0;
        } else {
            double dist[DIST_LEN];
            dealer_dist_excl(e, counts, e->rules.up_value, e->rules.up_is_ace,
                             e->rules.exclude_first, dist);
            ev = 0.0;
            e2 = 0.0;
            for (int d = 0; d < DIST_LEN; d++) {
                double p = dist[d];
                if (p == 0.0) continue;
                double x = 0.0;
                for (int i = 0; i < n_finished; i++) {
                    double stake = finished[i].stake2 ? 2.0 : 1.0;
                    int t = finished[i].total;
                    double y;
                    if (d == 5) y = 1.0;
                    else if (17 + d < t) y = 1.0;
                    else if (17 + d == t) y = 0.0;
                    else y = -1.0;
                    x += stake * y;
                }
                ev += p * x;
                e2 += p * x * x;
            }
        }
        if (!e->failed) {
            MomentsVal val;
            val.ev = ev;
            val.e2 = e2;
            if (!table_put_moments(&e->moments, &key, &val)) e->failed = 1;
            else e->moments_states++;
        }
        *ev_out = ev;
        *e2_out = e2;
        return;
    }

    const Rules *r = &e->rules;
    HandState h0 = pending[0];
    int best = best_total(h0.hard, h0.ace);
    int n = sum_counts(counts);

    /* STAND: hand 0 finishes on its best total, then the rest play. */
    FinState fin[MAX_HANDS];
    if (n_finished > 0)
        memcpy(fin, finished, sizeof(FinState) * (size_t)n_finished);
    fin[n_finished].total = (uint8_t)best;
    fin[n_finished].stake2 = 0;
    sort_finished(fin, n_finished + 1);
    double best_ev, best_e2;
    joint_moments(e, counts, pending + 1, n_pending - 1, fin,
                  n_finished + 1, splits_used, &best_ev, &best_e2);

    if (!h0.restricted && best < 21 && n > 0) {
        /* HIT: draw one card and keep playing this hand. A bust is a
         * constant -1 added to the rest of the round, so the second
         * moment shifts by E[(Y - 1)^2] = E[Y^2] - 2 E[Y] + 1. */
        double ev_hit = 0.0, e2_hit = 0.0;
        for (int v = 1; v <= 10; v++) {
            int c = counts[v - 1];
            if (c == 0) continue;
            int nxt[10];
            memcpy(nxt, counts, sizeof(int) * 10);
            nxt[v - 1]--;
            HandState nh = h0;
            nh.hard = (uint8_t)(h0.hard + v);
            nh.ace = (uint8_t)(h0.ace || v == 1);
            nh.n = (uint8_t)(h0.n + 1);
            nh.pair_rank = 0;
            double sub_ev, sub_e2;
            if (nh.hard > 21) {
                double ev_r, e2_r;
                joint_moments(e, nxt, pending + 1, n_pending - 1,
                              finished, n_finished, splits_used, &ev_r, &e2_r);
                sub_ev = -1.0 + ev_r;
                sub_e2 = e2_r - 2.0 * ev_r + 1.0;
            } else {
                HandState rest[MAX_HANDS];
                rest[0] = nh;
                memcpy(rest + 1, pending + 1,
                       sizeof(HandState) * (size_t)(n_pending - 1));
                joint_moments(e, nxt, rest, n_pending, finished, n_finished,
                              splits_used, &sub_ev, &sub_e2);
            }
            double p = (double)c / (double)n;
            ev_hit += p * sub_ev;
            e2_hit += p * sub_e2;
        }
        if (ev_hit > best_ev) {
            best_ev = ev_hit;
            best_e2 = e2_hit;
        }

        /* DOUBLE: exactly one card, on a doubled stake. A bust is a
         * constant -2: E[(Y - 2)^2] = E[Y^2] - 4 E[Y] + 4. */
        if (h0.n == 2 && r->double_after_split) {
            double ev_dbl = 0.0, e2_dbl = 0.0;
            for (int v = 1; v <= 10; v++) {
                int c = counts[v - 1];
                if (c == 0) continue;
                int nxt[10];
                memcpy(nxt, counts, sizeof(int) * 10);
                nxt[v - 1]--;
                int new_hard = h0.hard + v;
                double sub_ev, sub_e2;
                if (new_hard > 21) {
                    double ev_r, e2_r;
                    joint_moments(e, nxt, pending + 1, n_pending - 1,
                                  finished, n_finished, splits_used, &ev_r, &e2_r);
                    sub_ev = -2.0 + ev_r;
                    sub_e2 = e2_r - 4.0 * ev_r + 4.0;
                } else {
                    FinState f2[MAX_HANDS];
                    if (n_finished > 0)
                        memcpy(f2, finished,
                               sizeof(FinState) * (size_t)n_finished);
                    f2[n_finished].total =
                        (uint8_t)best_total(new_hard, h0.ace || v == 1);
                    f2[n_finished].stake2 = 1;
                    sort_finished(f2, n_finished + 1);
                    joint_moments(e, nxt, pending + 1, n_pending - 1, f2,
                                  n_finished + 1, splits_used, &sub_ev, &sub_e2);
                }
                double p = (double)c / (double)n;
                ev_dbl += p * sub_ev;
                e2_dbl += p * sub_e2;
            }
            if (ev_dbl > best_ev) {
                best_ev = ev_dbl;
                best_e2 = e2_dbl;
            }
        }
    }

    /* RESPLIT: mirror joint() -- same conditions, same unordered-pair
     * dealing, the two new hands valued as one mixture. */
    if (h0.n == 2 && h0.pair_rank != 0 && splits_used < r->max_splits &&
        !(h0.pair_rank == 1 && !r->resplit_aces) && n >= 2) {
        int single = h0.pair_rank;
        int new_restricted = (single == 1 && !r->hit_split_aces);
        double ev_resplit = 0.0, e2_resplit = 0.0;
        for (int v1 = 1; v1 <= 10; v1++) {
            int c1 = counts[v1 - 1];
            if (c1 == 0) continue;
            int mid[10];
            memcpy(mid, counts, sizeof(int) * 10);
            mid[v1 - 1]--;
            for (int v2 = v1; v2 <= 10; v2++) {
                int c2 = mid[v2 - 1];
                if (c2 == 0) continue;
                int nxt[10];
                memcpy(nxt, mid, sizeof(int) * 10);
                nxt[v2 - 1]--;
                HandState h1, h2;
                h1.hard = (uint8_t)(single + v1);
                h1.ace = (uint8_t)(single == 1 || v1 == 1);
                h1.n = 2;
                h1.restricted = (uint8_t)new_restricted;
                h1.pair_rank = (uint8_t)(v1 == single ? single : 0);
                h2.hard = (uint8_t)(single + v2);
                h2.ace = (uint8_t)(single == 1 || v2 == 1);
                h2.n = 2;
                h2.restricted = (uint8_t)new_restricted;
                h2.pair_rank = (uint8_t)(v2 == single ? single : 0);
                HandState rest[MAX_HANDS];
                rest[0] = h1;
                rest[1] = h2;
                memcpy(rest + 2, pending + 1,
                       sizeof(HandState) * (size_t)(n_pending - 1));
                double p = ((double)c1 / (double)n) *
                           ((double)c2 / (double)(n - 1));
                if (v2 != v1) p *= 2.0;
                double sub_ev, sub_e2;
                joint_moments(e, nxt, rest, n_pending + 1, finished,
                              n_finished, splits_used + 1, &sub_ev, &sub_e2);
                ev_resplit += p * sub_ev;
                e2_resplit += p * sub_e2;
            }
        }
        if (ev_resplit > best_ev) {
            best_ev = ev_resplit;
            best_e2 = e2_resplit;
        }
    }

    ev = best_ev;
    e2 = best_e2;
    if (!e->failed) {
        MomentsVal val;
        val.ev = ev;
        val.e2 = e2;
        if (!table_put_moments(&e->moments, &key, &val)) e->failed = 1;
        else e->moments_states++;
    }
    *ev_out = ev;
    *e2_out = e2;
}

/* ------------------------------------------------------------------ */
/* The sequential decomposition -- the LIVE engine.                    */
/*                                                                     */
/* A straight port of archive/split_sequential.py (the pure-Python      */
/* version), which is why the structure below mirrors that file's       */
/* functions one for one. The split is valued in the order the game     */
/* actually plays the hands:                                           */
/* - decomp_first plays the hand being split to maximise the TOTAL ev  */
/*   of every hand, calling decomp_tail for the hands still to come.    */
/* - decomp_hand_after values ONE hand played after others have        */
/*   finished: its own ev, plus the expected payoff of a finished hand  */
/*   standing on each total 2..21 against the dealer distribution      */
/*   after it. It is keyed by the hand alone -- no other hand in the   */
/*   key -- so its states are shared across every hand that is played  */
/*   first, which is what keeps the whole thing fast enough to run.    */
/* - decomp_tail values a list of hands still to come (only ever       */
/*   needed after a resplit), each to maximise its own ev, returning    */
/*   (per-hand EVs, expected payoffs). It delegates to                 */
/*   decomp_hand_after for a single hand.                              */
/* A payoff is linear in the dealer distribution, so the expected      */
/* payoff against the expected distribution is the expected payoff --   */
/* that is the step that lets a finished hand be settled without        */
/* knowing which cards the later hands will draw.                      */
/*                                                                     */
/* The one thing the later hands do not account for is the reverse: a  */
/* later hand's draws also move the dealer distribution the earlier    */
/* hands are settled against, and its decisions are not re-optimised   */
/* for that. That effect is second-order (about a hundredth at a full  */
/* deck), so this engine is near-exact, not exact -- the joint above   */
/* is the exact check. And like the Python version it stands a         */
/* restricted pair of split aces rather than resplitting it, although  */
/* the game allows the resplit -- the joint resplits them, which is    */
/* why its ace-split value is a touch higher. With the default split   */
/* cap of one that never arises: no resplit is possible at all.        */
/*                                                                     */
/* Why this runs live and the joint does not: its state is one hand    */
/* at a time, not all of them at once, so each extra hand adds states  */
/* instead of multiplying them. The cost is the second-order coupling  */
/* above and a wide memo value (the payoffs travel with every state).  */
/* ------------------------------------------------------------------ */

static double decomp_first(Engine *e, const int *counts,
                           const HandState *hands, int n_hands,
                           int splits_used);
static int decomp_tail(Engine *e, const int *counts, const HandState *hands,
                       int n_hands, int splits_used, double *evs_out,
                       double *payoffs_out);

/* The expected payoff of standing on each total 2..21 (index total - 2),
 * per unit staked, against the dealer distribution from these counts. */
static void decomp_payoffs(Engine *e, const int *counts, double *payoffs) {
    double dist[DIST_LEN];
    dealer_dist_excl(e, counts, e->rules.up_value, e->rules.up_is_ace,
                     e->rules.exclude_first, dist);
    for (int total = 2; total <= 21; total++)
        payoffs[total - 2] = payoff_vs_dist(dist, total);
}

/* (Own EV, expected payoffs) for one hand played from here, optimally for
 * its own ev. The payoffs are the expected payoff of a finished hand
 * standing on each total 2..21 against the dealer distribution after this
 * hand has played -- which is what a hand that finished earlier needs, to
 * be settled against a dealer it has not seen yet. When the engine has
 * failed (out of memory) the outputs are zeroed. The caller discards them
 * because the entry point returns NaN. */
static void decomp_hand_after(Engine *e, const int *counts,
                              const HandState *h, double *own_out,
                              double *payoffs_out) {
    if (e->failed) {
        *own_out = 0.0;
        memset(payoffs_out, 0, sizeof(double) * 20);
        return;
    }
    DecompKey key = pack_decomp(counts, h, 1, 0);
    DecompVal cached;
    if (table_get_decomp(&e->hand, &key, &cached)) {
        *own_out = cached.evs[0];
        memcpy(payoffs_out, cached.payoffs, sizeof(double) * 20);
        return;
    }
    int best = best_total(h->hard, h->ace);
    double payoffs[20];
    decomp_payoffs(e, counts, payoffs);
    double best_own = payoffs[best - 2]; /* STAND */
    double best_pay[20];
    memcpy(best_pay, payoffs, sizeof(double) * 20);
    int n = sum_counts(counts);
    const Rules *r = &e->rules;
    if (!h->restricted && best < 21 && n > 0) {
        /* HIT: average over the card drawn, then continue optimally. */
        double own_ev = 0.0;
        double pay_acc[20] = {0};
        for (int v = 1; v <= 10; v++) {
            int c = counts[v - 1];
            if (c == 0) continue;
            int nxt[10];
            memcpy(nxt, counts, sizeof(int) * 10);
            nxt[v - 1]--;
            int new_hard = h->hard + v;
            double own, pay[20];
            if (new_hard > 21) {
                own = -1.0;
                decomp_payoffs(e, nxt, pay);
            } else {
                HandState nh = *h;
                nh.hard = (uint8_t)new_hard;
                nh.ace = (uint8_t)(h->ace || v == 1);
                nh.n = (uint8_t)(h->n + 1);
                nh.pair_rank = 0;
                decomp_hand_after(e, nxt, &nh, &own, pay);
            }
            double p = (double)c / (double)n;
            own_ev += p * own;
            for (int i = 0; i < 20; i++) pay_acc[i] += p * pay[i];
        }
        if (own_ev > best_own) {
            best_own = own_ev;
            memcpy(best_pay, pay_acc, sizeof(double) * 20);
        }
        /* DOUBLE: exactly one card, on a doubled stake -- only on two
         * cards, and only when the table allows doubling after a split. */
        if (h->n == 2 && r->double_after_split) {
            double own_ev2 = 0.0;
            double pay_acc2[20] = {0};
            for (int v = 1; v <= 10; v++) {
                int c = counts[v - 1];
                if (c == 0) continue;
                int nxt[10];
                memcpy(nxt, counts, sizeof(int) * 10);
                nxt[v - 1]--;
                int new_hard = h->hard + v;
                double pay[20];
                decomp_payoffs(e, nxt, pay);
                double own;
                if (new_hard > 21) {
                    own = -2.0;
                } else {
                    own = 2.0 * pay[best_total(new_hard, h->ace || v == 1) - 2];
                }
                double p = (double)c / (double)n;
                own_ev2 += p * own;
                for (int i = 0; i < 20; i++) pay_acc2[i] += p * pay[i];
            }
            if (own_ev2 > best_own) {
                best_own = own_ev2;
                memcpy(best_pay, pay_acc2, sizeof(double) * 20);
            }
        }
    }
    *own_out = best_own;
    memcpy(payoffs_out, best_pay, sizeof(double) * 20);
    if (!e->failed) {
        DecompVal val;
        val.evs[0] = best_own;
        for (int i = 1; i < MAX_HANDS; i++) val.evs[i] = 0.0;
        memcpy(val.payoffs, best_pay, sizeof(double) * 20);
        val.n = 1;
        if (!table_put_decomp(&e->hand, &key, &val)) e->failed = 1;
        else e->decomp_hand_states++;
    }
}

/* Draw one card to the first hand still to come: a hit (keep playing it)
 * or a double (finish it on a doubled stake). Fills *own_out with this
 * hand's ev, evs_out with the rest's per-hand EVs, and payoffs_out with
 * the expected payoffs. */
static void decomp_draw_tail(Engine *e, const int *counts,
                             const HandState *hands, int n_hands,
                             int splits_used, int dbl, double *own_out,
                             double *evs_out, double *payoffs_out) {
    int n = sum_counts(counts);
    HandState h0 = hands[0];
    int n_rest = n_hands - 1;
    double own_ev = 0.0;
    double evs_acc[MAX_HANDS] = {0};
    double pay_acc[20] = {0};
    for (int v = 1; v <= 10; v++) {
        int c = counts[v - 1];
        if (c == 0) continue;
        int nxt[10];
        memcpy(nxt, counts, sizeof(int) * 10);
        nxt[v - 1]--;
        HandState nh = h0;
        nh.hard = (uint8_t)(h0.hard + v);
        nh.ace = (uint8_t)(h0.ace || v == 1);
        nh.n = (uint8_t)(h0.n + 1);
        nh.pair_rank = 0;
        double evs_r[MAX_HANDS] = {0};
        double pay[20];
        double own;
        if (nh.hard > 21) {
            decomp_tail(e, nxt, hands + 1, n_rest, splits_used, evs_r, pay);
            own = dbl ? -2.0 : -1.0;
        } else if (dbl) {
            int total = best_total(nh.hard, nh.ace);
            decomp_tail(e, nxt, hands + 1, n_rest, splits_used, evs_r, pay);
            own = 2.0 * pay[total - 2];
        } else {
            HandState hh[MAX_HANDS];
            hh[0] = nh;
            memcpy(hh + 1, hands + 1, sizeof(HandState) * (size_t)n_rest);
            double evs_c[MAX_HANDS] = {0};
            decomp_tail(e, nxt, hh, n_hands, splits_used, evs_c, pay);
            own = evs_c[0];
            memcpy(evs_r, evs_c + 1, sizeof(double) * (size_t)n_rest);
        }
        double p = (double)c / (double)n;
        own_ev += p * own;
        for (int i = 0; i < n_rest; i++) evs_acc[i] += p * evs_r[i];
        for (int i = 0; i < 20; i++) pay_acc[i] += p * pay[i];
    }
    *own_out = own_ev;
    memcpy(evs_out, evs_acc, sizeof(double) * MAX_HANDS);
    memcpy(payoffs_out, pay_acc, sizeof(double) * 20);
}

/* Split the first hand still to come again, and value the two new hands
 * as the hand's own ev (their EVs added). Returns 1 and fills the
 * outputs, or 0 when fewer than two cards remain. */
static int decomp_resplit_tail(Engine *e, const int *counts,
                               const HandState *hands, int n_hands,
                               int splits_used, double *own_out,
                               double *evs_out, double *payoffs_out) {
    HandState h0 = hands[0];
    int single = h0.pair_rank;
    int n_rest = n_hands - 1;
    int n = sum_counts(counts);
    if (n < 2) return 0;
    int new_restricted = (single == 1 && !e->rules.hit_split_aces);
    double own_ev = 0.0;
    double evs_acc[MAX_HANDS] = {0};
    double pay_acc[20] = {0};
    for (int v1 = 1; v1 <= 10; v1++) {
        int c1 = counts[v1 - 1];
        if (c1 == 0) continue;
        int mid[10];
        memcpy(mid, counts, sizeof(int) * 10);
        mid[v1 - 1]--;
        for (int v2 = v1; v2 <= 10; v2++) {
            int c2 = mid[v2 - 1];
            if (c2 == 0) continue;
            int nxt[10];
            memcpy(nxt, mid, sizeof(int) * 10);
            nxt[v2 - 1]--;
            HandState h1, h2;
            h1.hard = (uint8_t)(single + v1);
            h1.ace = (uint8_t)(single == 1 || v1 == 1);
            h1.n = 2;
            h1.restricted = (uint8_t)new_restricted;
            h1.pair_rank = (uint8_t)(v1 == single ? single : 0);
            h2.hard = (uint8_t)(single + v2);
            h2.ace = (uint8_t)(single == 1 || v2 == 1);
            h2.n = 2;
            h2.restricted = (uint8_t)new_restricted;
            h2.pair_rank = (uint8_t)(v2 == single ? single : 0);
            HandState hh[MAX_HANDS];
            hh[0] = h1;
            hh[1] = h2;
            memcpy(hh + 2, hands + 1, sizeof(HandState) * (size_t)n_rest);
            double evs_new[MAX_HANDS] = {0};
            double pay[20];
            decomp_tail(e, nxt, hh, n_hands + 1, splits_used + 1, evs_new,
                        pay);
            double p = ((double)c1 / (double)n) *
                       ((double)c2 / (double)(n - 1));
            if (v2 != v1) p *= 2.0;
            own_ev += p * (evs_new[0] + evs_new[1]);
            for (int i = 0; i < n_rest; i++) evs_acc[i] += p * evs_new[i + 2];
            for (int i = 0; i < 20; i++) pay_acc[i] += p * pay[i];
        }
    }
    *own_out = own_ev;
    memcpy(evs_out, evs_acc, sizeof(double) * MAX_HANDS);
    memcpy(payoffs_out, pay_acc, sizeof(double) * 20);
    return 1;
}

/* (Per-hand EVs, expected payoffs) for the hands still to come. Each hand
 * is played to maximise its own ev, and the payoffs are the expected payoff
 * of a finished hand standing on each total 2..21 against the dealer
 * distribution after every hand has played. A single hand delegates to
 * decomp_hand_after. No hands left means the dealer plays now, from these
 * counts. Returns n_hands. */
static int decomp_tail(Engine *e, const int *counts, const HandState *hands,
                       int n_hands, int splits_used, double *evs_out,
                       double *payoffs_out) {
    if (e->failed) {
        for (int i = 0; i < MAX_HANDS; i++) evs_out[i] = 0.0;
        memset(payoffs_out, 0, sizeof(double) * 20);
        return n_hands;
    }
    DecompKey key = pack_decomp(counts, hands, n_hands, splits_used);
    DecompVal cached;
    if (table_get_decomp(&e->tail, &key, &cached)) {
        memcpy(evs_out, cached.evs, sizeof(double) * MAX_HANDS);
        memcpy(payoffs_out, cached.payoffs, sizeof(double) * 20);
        return cached.n;
    }
    const Rules *r = &e->rules;
    double evs[MAX_HANDS] = {0};
    double payoffs[20];
    if (n_hands == 0) {
        decomp_payoffs(e, counts, payoffs);
    } else if (n_hands == 1) {
        double own;
        decomp_hand_after(e, counts, &hands[0], &own, payoffs);
        evs[0] = own;
    } else {
        /* Hand 0 maximises its OWN ev. The rest are valued the same way
         * either way. Every candidate is (this hand's ev, the rest's
         * per-hand EVs, expected payoffs). */
        HandState h0 = hands[0];
        int best = best_total(h0.hard, h0.ace);
        int n = sum_counts(counts);
        double evs_rest[MAX_HANDS] = {0};
        double pay_rest[20];
        decomp_tail(e, counts, hands + 1, n_hands - 1, splits_used, evs_rest,
                    pay_rest);
        double best_own = pay_rest[best - 2]; /* STAND */
        double best_evs[MAX_HANDS];
        memcpy(best_evs, evs_rest, sizeof(double) * MAX_HANDS);
        double best_pay[20];
        memcpy(best_pay, pay_rest, sizeof(double) * 20);
        if (!h0.restricted && best < 21 && n > 0) {
            double own, evs2[MAX_HANDS], pay2[20];
            decomp_draw_tail(e, counts, hands, n_hands, splits_used, 0,
                             &own, evs2, pay2); /* HIT */
            if (own > best_own) {
                best_own = own;
                memcpy(best_evs, evs2, sizeof(double) * MAX_HANDS);
                memcpy(best_pay, pay2, sizeof(double) * 20);
            }
            if (h0.n == 2 && r->double_after_split) {
                decomp_draw_tail(e, counts, hands, n_hands, splits_used, 1,
                                 &own, evs2, pay2); /* DOUBLE */
                if (own > best_own) {
                    best_own = own;
                    memcpy(best_evs, evs2, sizeof(double) * MAX_HANDS);
                    memcpy(best_pay, pay2, sizeof(double) * 20);
                }
            }
            if (h0.n == 2 && h0.pair_rank != 0 &&
                splits_used < r->max_splits &&
                !(h0.pair_rank == 1 && !r->resplit_aces) && n >= 2) {
                if (decomp_resplit_tail(e, counts, hands, n_hands,
                                        splits_used, &own, evs2, pay2)) {
                    if (own > best_own) {
                        best_own = own;
                        memcpy(best_evs, evs2, sizeof(double) * MAX_HANDS);
                        memcpy(best_pay, pay2, sizeof(double) * 20);
                    }
                }
            }
        }
        evs[0] = best_own;
        for (int i = 0; i < n_hands - 1; i++) evs[i + 1] = best_evs[i];
        memcpy(payoffs, best_pay, sizeof(double) * 20);
    }
    memcpy(evs_out, evs, sizeof(double) * MAX_HANDS);
    memcpy(payoffs_out, payoffs, sizeof(double) * 20);
    if (!e->failed) {
        DecompVal val;
        memcpy(val.evs, evs, sizeof(double) * MAX_HANDS);
        memcpy(val.payoffs, payoffs, sizeof(double) * 20);
        val.n = n_hands;
        if (!table_put_decomp(&e->tail, &key, &val)) e->failed = 1;
        else e->decomp_tail_states++;
    }
    return n_hands;
}

/* Draw one card to the hand being split: a hit (keep playing it) or a
 * double (finish it on a doubled stake, then the rest play out). */
static double decomp_draw_first(Engine *e, const int *counts,
                                const HandState *hands, int n_hands,
                                int splits_used, int dbl) {
    int n = sum_counts(counts);
    HandState h0 = hands[0];
    int n_rest = n_hands - 1;
    double ev = 0.0;
    for (int v = 1; v <= 10; v++) {
        int c = counts[v - 1];
        if (c == 0) continue;
        int nxt[10];
        memcpy(nxt, counts, sizeof(int) * 10);
        nxt[v - 1]--;
        HandState nh = h0;
        nh.hard = (uint8_t)(h0.hard + v);
        nh.ace = (uint8_t)(h0.ace || v == 1);
        nh.n = (uint8_t)(h0.n + 1);
        nh.pair_rank = 0;
        double sub;
        if (nh.hard > 21) {
            double evs_rest[MAX_HANDS] = {0};
            double pay[20];
            decomp_tail(e, nxt, hands + 1, n_rest, splits_used, evs_rest,
                        pay);
            double s = 0.0;
            for (int i = 0; i < n_rest; i++) s += evs_rest[i];
            sub = (dbl ? -2.0 : -1.0) + s;
        } else if (dbl) {
            int total = best_total(nh.hard, nh.ace);
            double evs_rest[MAX_HANDS] = {0};
            double pay[20];
            decomp_tail(e, nxt, hands + 1, n_rest, splits_used, evs_rest,
                        pay);
            double s = 0.0;
            for (int i = 0; i < n_rest; i++) s += evs_rest[i];
            sub = 2.0 * pay[total - 2] + s;
        } else {
            HandState hh[MAX_HANDS];
            hh[0] = nh;
            memcpy(hh + 1, hands + 1, sizeof(HandState) * (size_t)n_rest);
            sub = decomp_first(e, nxt, hh, n_hands, splits_used);
        }
        ev += ((double)c / (double)n) * sub;
    }
    return ev;
}

/* Split the hand being split again: two new one-card hands, each dealt a
 * second card, then play resumes at the first of them. Returns 1 and fills
 * *out, or 0 when fewer than two cards remain. */
static int decomp_resplit_first(Engine *e, const int *counts,
                                const HandState *hands, int n_hands,
                                int splits_used, double *out) {
    HandState h0 = hands[0];
    int single = h0.pair_rank;
    int n_rest = n_hands - 1;
    int n = sum_counts(counts);
    if (n < 2) return 0;
    int new_restricted = (single == 1 && !e->rules.hit_split_aces);
    double ev = 0.0;
    for (int v1 = 1; v1 <= 10; v1++) {
        int c1 = counts[v1 - 1];
        if (c1 == 0) continue;
        int mid[10];
        memcpy(mid, counts, sizeof(int) * 10);
        mid[v1 - 1]--;
        for (int v2 = v1; v2 <= 10; v2++) {
            int c2 = mid[v2 - 1];
            if (c2 == 0) continue;
            int nxt[10];
            memcpy(nxt, mid, sizeof(int) * 10);
            nxt[v2 - 1]--;
            HandState h1, h2;
            h1.hard = (uint8_t)(single + v1);
            h1.ace = (uint8_t)(single == 1 || v1 == 1);
            h1.n = 2;
            h1.restricted = (uint8_t)new_restricted;
            h1.pair_rank = (uint8_t)(v1 == single ? single : 0);
            h2.hard = (uint8_t)(single + v2);
            h2.ace = (uint8_t)(single == 1 || v2 == 1);
            h2.n = 2;
            h2.restricted = (uint8_t)new_restricted;
            h2.pair_rank = (uint8_t)(v2 == single ? single : 0);
            HandState hh[MAX_HANDS];
            hh[0] = h1;
            hh[1] = h2;
            memcpy(hh + 2, hands + 1, sizeof(HandState) * (size_t)n_rest);
            double p = ((double)c1 / (double)n) *
                       ((double)c2 / (double)(n - 1));
            if (v2 != v1) p *= 2.0;
            ev += p * decomp_first(e, nxt, hh, n_hands + 1, splits_used + 1);
        }
    }
    *out = ev;
    return 1;
}

/* Total EV of all the split hands, with hands[0] played to maximise it.
 * hands[0] is the hand being split (or one of its resplits). Hands[1:]
 * are the hands still to come, valued by decomp_tail. When hands[0]
 * finishes, the rest play out under their own policies and the total is
 * its ev plus theirs. Memoised in the joint table: the key is pack_joint
 * with no finished hands, and the value is a single double. */
static double decomp_first(Engine *e, const int *counts,
                           const HandState *hands, int n_hands,
                           int splits_used) {
    if (e->failed) return 0.0;
    JointKey key = pack_joint(counts, hands, n_hands, NULL, 0, splits_used);
    double cached;
    if (table_get_joint(&e->joint, &key, &cached)) return cached;
    HandState h0 = hands[0];
    int best = best_total(h0.hard, h0.ace);
    int n = sum_counts(counts);
    const Rules *r = &e->rules;
    /* STAND: finish on the best total, then the rest play out. */
    double evs_rest[MAX_HANDS] = {0};
    double payoffs[20];
    decomp_tail(e, counts, hands + 1, n_hands - 1, splits_used, evs_rest,
                payoffs);
    double best_ev = payoffs[best - 2];
    for (int i = 0; i < n_hands - 1; i++) best_ev += evs_rest[i];
    if (!h0.restricted && best < 21 && n > 0) {
        /* HIT: draw and keep playing this hand (or bust, and move on). */
        double ev = decomp_draw_first(e, counts, hands, n_hands, splits_used,
                                      0);
        if (ev > best_ev) best_ev = ev;
        /* DOUBLE: exactly one card, on a doubled stake -- only on two
         * cards, and only when the table allows doubling after a split. */
        if (h0.n == 2 && r->double_after_split) {
            ev = decomp_draw_first(e, counts, hands, n_hands, splits_used, 1);
            if (ev > best_ev) best_ev = ev;
        }
        /* RESPLIT: a pair under the cap. Aces need the resplit_aces
         * toggle. This branch is NOT reached for a restricted hand: the
         * whole block is gated on !restricted, so a restricted ace pair
         * is stood on, like the archived Python version. */
        if (h0.n == 2 && h0.pair_rank != 0 &&
            splits_used < r->max_splits &&
            !(h0.pair_rank == 1 && !r->resplit_aces) && n >= 2) {
            if (decomp_resplit_first(e, counts, hands, n_hands, splits_used,
                                     &ev)) {
                if (ev > best_ev) best_ev = ev;
            }
        }
    }
    if (!e->failed) {
        if (!table_put_joint(&e->joint, &key, best_ev)) e->failed = 1;
        else e->decomp_first_states++;
    }
    return best_ev;
}

/* ------------------------------------------------------------------ */
/* The exported entry point: split a pair and value it exactly.        */
/* Mirrors OddsCalculator._split_odds: the two split hands are dealt
 * their second cards first, as an unordered pair, then the joint
 * recursion plays them out. `exclude_first` is the peek value (0 when
 * the upcard cannot make blackjack). Returns NaN on allocation
 * failure, so the caller can fail loudly instead of silently
 * returning a wrong number.                                           */
/* ------------------------------------------------------------------ */

static long g_joint_states = 0;
static long g_dealer_states = 0;
static long g_dealer_lookups = 0;
static long g_dealer_hits = 0;

double split_ev_exact(const int *counts, int single_value, int single_is_ace,
                      int up_value, int up_is_ace, int exclude_first,
                      int max_splits, int double_after_split, int resplit_aces,
                      int hit_split_aces, int dealer_hits_soft_17) {
    Engine e;
    memset(&e, 0, sizeof(e));
    e.rules.max_splits = max_splits;
    e.rules.double_after_split = double_after_split;
    e.rules.resplit_aces = resplit_aces;
    e.rules.hit_split_aces = hit_split_aces;
    e.rules.dealer_hits_soft_17 = dealer_hits_soft_17;
    e.rules.up_value = up_value;
    e.rules.up_is_ace = up_is_ace;
    e.rules.exclude_first = exclude_first;

    int n = sum_counts(counts);
    if (n < 2) return 0.0;
    /* The dealer cache is direct-mapped and never grows, so it starts at
     * its full size: 2M slots (128MB virtual, committed only as used). */
    if (!table_init_joint(&e.joint, 65536) ||
        !table_init_dealer(&e.dealer, 1 << 21)) {
        table_free_joint(&e.joint);
        table_free_dealer(&e.dealer);
        return 0.0 / 0.0; /* NaN: out of memory before starting */
    }

    int restricted = (single_is_ace && !hit_split_aces);
    double ev = 0.0;
    for (int v1 = 1; v1 <= 10; v1++) {
        int c1 = counts[v1 - 1];
        if (c1 == 0) continue;
        int mid[10];
        memcpy(mid, counts, sizeof(int) * 10);
        mid[v1 - 1]--;
        for (int v2 = v1; v2 <= 10; v2++) {
            int c2 = mid[v2 - 1];
            if (c2 == 0) continue;
            int nxt[10];
            memcpy(nxt, mid, sizeof(int) * 10);
            nxt[v2 - 1]--;
            HandState h1, h2;
            h1.hard = (uint8_t)(single_value + v1);
            h1.ace = (uint8_t)(single_is_ace || v1 == 1);
            h1.n = 2;
            h1.restricted = (uint8_t)restricted;
            h1.pair_rank = (uint8_t)(v1 == single_value ? single_value : 0);
            h2.hard = (uint8_t)(single_value + v2);
            h2.ace = (uint8_t)(single_is_ace || v2 == 1);
            h2.n = 2;
            h2.restricted = (uint8_t)restricted;
            h2.pair_rank = (uint8_t)(v2 == single_value ? single_value : 0);
            HandState pending[2] = {h1, h2};
            double p = ((double)c1 / (double)n) *
                       ((double)c2 / (double)(n - 1));
            if (v2 != v1) p *= 2.0;
            ev += p * joint(&e, nxt, pending, 2, NULL, 0, 1);
            if (e.failed) break;
        }
        if (e.failed) break;
    }

    g_joint_states = e.joint_states;
    g_dealer_states = e.dealer_states;
    g_dealer_lookups = e.dealer_lookups;
    g_dealer_hits = e.dealer_hits;
    table_free_joint(&e.joint);
    table_free_dealer(&e.dealer);
    if (e.failed) return 0.0 / 0.0; /* NaN: out of memory mid-computation */
    return ev;
}

/* Statistics from the last call, for the tests and the README. */
long split_exact_joint_states(void) { return g_joint_states; }
long split_exact_dealer_states(void) { return g_dealer_states; }
long split_exact_dealer_lookups(void) { return g_dealer_lookups; }
long split_exact_dealer_hits(void) { return g_dealer_hits; }
/* ------------------------------------------------------------------ */
/* The exported entry point for the joint moments (iteration 6a).      */
/* Mirrors split_ev_exact's entry: the two split hands are dealt their */
/* second cards first, as an unordered pair, then the joint moments    */
/* recursion plays them out. Fills *ev_out with the exact total EV and */
/* *var_out with the exact variance of the total split payoff, per     */
/* unit bet on the original hand. Returns 1 on success, 0 on           */
/* allocation failure (outs set to NaN), so the caller can fail loudly */
/* instead of silently returning a wrong number.                       */
/* ------------------------------------------------------------------ */

static long g_moments_states = 0;

int split_moments_exact(const int *counts, int single_value, int single_is_ace,
                      int up_value, int up_is_ace, int exclude_first,
                      int max_splits, int double_after_split, int resplit_aces,
                      int hit_split_aces, int dealer_hits_soft_17,
                      double *ev_out, double *var_out) {
    Engine e;
    memset(&e, 0, sizeof(e));
    e.rules.max_splits = max_splits;
    e.rules.double_after_split = double_after_split;
    e.rules.resplit_aces = resplit_aces;
    e.rules.hit_split_aces = hit_split_aces;
    e.rules.dealer_hits_soft_17 = dealer_hits_soft_17;
    e.rules.up_value = up_value;
    e.rules.up_is_ace = up_is_ace;
    e.rules.exclude_first = exclude_first;

    int n = sum_counts(counts);
    if (n < 2) {
        *ev_out = 0.0;
        *var_out = 0.0;
        return 1;
    }
    /* The dealer cache is direct-mapped and never grows, so it starts at
     * its full size: 2M slots (128MB virtual, committed only as used). */
    if (!table_init_dealer(&e.dealer, 1 << 21) ||
        !table_init_moments(&e.moments, 65536)) {
        table_free_dealer(&e.dealer);
        table_free_moments(&e.moments);
        *ev_out = 0.0 / 0.0;
        *var_out = 0.0 / 0.0;
        return 0;
    }

    int restricted = (single_is_ace && !hit_split_aces);
    double ev = 0.0, e2 = 0.0;
    for (int v1 = 1; v1 <= 10; v1++) {
        int c1 = counts[v1 - 1];
        if (c1 == 0) continue;
        int mid[10];
        memcpy(mid, counts, sizeof(int) * 10);
        mid[v1 - 1]--;
        for (int v2 = v1; v2 <= 10; v2++) {
            int c2 = mid[v2 - 1];
            if (c2 == 0) continue;
            int nxt[10];
            memcpy(nxt, mid, sizeof(int) * 10);
            nxt[v2 - 1]--;
            HandState h1, h2;
            h1.hard = (uint8_t)(single_value + v1);
            h1.ace = (uint8_t)(single_is_ace || v1 == 1);
            h1.n = 2;
            h1.restricted = (uint8_t)restricted;
            h1.pair_rank = (uint8_t)(v1 == single_value ? single_value : 0);
            h2.hard = (uint8_t)(single_value + v2);
            h2.ace = (uint8_t)(single_is_ace || v2 == 1);
            h2.n = 2;
            h2.restricted = (uint8_t)restricted;
            h2.pair_rank = (uint8_t)(v2 == single_value ? single_value : 0);
            HandState pending[2] = {h1, h2};
            double p = ((double)c1 / (double)n) *
                       ((double)c2 / (double)(n - 1));
            if (v2 != v1) p *= 2.0;
            double ev_h, e2_h;
            joint_moments(&e, nxt, pending, 2, NULL, 0, 1, &ev_h, &e2_h);
            ev += p * ev_h;
            e2 += p * e2_h;
            if (e.failed) break;
        }
        if (e.failed) break;
    }

    g_moments_states = e.moments_states;
    g_dealer_states = e.dealer_states;
    g_dealer_lookups = e.dealer_lookups;
    g_dealer_hits = e.dealer_hits;
    table_free_dealer(&e.dealer);
    table_free_moments(&e.moments);
    if (e.failed) {
        *ev_out = 0.0 / 0.0;
        *var_out = 0.0 / 0.0;
        return 0;
    }
    *ev_out = ev;
    *var_out = e2 - ev * ev;
    return 1;
}

/* Statistics from the last moments call, for the tests and the README. */
long split_moments_states(void) { return g_moments_states; }

/* ------------------------------------------------------------------ */
/* The exported entry point for the decomposition -- the LIVE engine.  */
/* Mirrors split_ev_exact's entry: the two split hands are dealt their */
/* second cards first, as an unordered pair, then the decomposition     */
/* plays them out. Returns NaN on allocation failure, so the caller    */
/* can fail loudly instead of silently returning a wrong number. The   */
/* dealer statistics are shared with the joint (the dealer cache is    */
/* one table, used by whichever engine ran last. The joint state count */
/* is left alone by this call.                                         */
/* ------------------------------------------------------------------ */

static long g_decomp_first_states = 0;
static long g_decomp_tail_states = 0;
static long g_decomp_hand_states = 0;

double split_ev_decomp(const int *counts, int single_value, int single_is_ace,
                       int up_value, int up_is_ace, int exclude_first,
                       int max_splits, int double_after_split, int resplit_aces,
                       int hit_split_aces, int dealer_hits_soft_17) {
    Engine e;
    memset(&e, 0, sizeof(e));
    e.rules.max_splits = max_splits;
    e.rules.double_after_split = double_after_split;
    e.rules.resplit_aces = resplit_aces;
    e.rules.hit_split_aces = hit_split_aces;
    e.rules.dealer_hits_soft_17 = dealer_hits_soft_17;
    e.rules.up_value = up_value;
    e.rules.up_is_ace = up_is_ace;
    e.rules.exclude_first = exclude_first;

    int n = sum_counts(counts);
    if (n < 2) return 0.0;
    /* The dealer cache is direct-mapped and never grows, so it starts at
     * its full size: 2M slots (128MB virtual, committed only as used). */
    if (!table_init_joint(&e.joint, 65536) ||
        !table_init_dealer(&e.dealer, 1 << 21) ||
        !table_init_decomp(&e.hand, 65536) ||
        !table_init_decomp(&e.tail, 65536)) {
        table_free_joint(&e.joint);
        table_free_dealer(&e.dealer);
        table_free_decomp(&e.hand);
        table_free_decomp(&e.tail);
        return 0.0 / 0.0; /* NaN: out of memory before starting */
    }

    int restricted = (single_is_ace && !hit_split_aces);
    double ev = 0.0;
    for (int v1 = 1; v1 <= 10; v1++) {
        int c1 = counts[v1 - 1];
        if (c1 == 0) continue;
        int mid[10];
        memcpy(mid, counts, sizeof(int) * 10);
        mid[v1 - 1]--;
        for (int v2 = v1; v2 <= 10; v2++) {
            int c2 = mid[v2 - 1];
            if (c2 == 0) continue;
            int nxt[10];
            memcpy(nxt, mid, sizeof(int) * 10);
            nxt[v2 - 1]--;
            HandState h1, h2;
            h1.hard = (uint8_t)(single_value + v1);
            h1.ace = (uint8_t)(single_is_ace || v1 == 1);
            h1.n = 2;
            h1.restricted = (uint8_t)restricted;
            h1.pair_rank = (uint8_t)(v1 == single_value ? single_value : 0);
            h2.hard = (uint8_t)(single_value + v2);
            h2.ace = (uint8_t)(single_is_ace || v2 == 1);
            h2.n = 2;
            h2.restricted = (uint8_t)restricted;
            h2.pair_rank = (uint8_t)(v2 == single_value ? single_value : 0);
            HandState pending[2] = {h1, h2};
            double p = ((double)c1 / (double)n) *
                       ((double)c2 / (double)(n - 1));
            if (v2 != v1) p *= 2.0;
            ev += p * decomp_first(&e, nxt, pending, 2, 1);
            if (e.failed) break;
        }
        if (e.failed) break;
    }

    g_decomp_first_states = e.decomp_first_states;
    g_decomp_tail_states = e.decomp_tail_states;
    g_decomp_hand_states = e.decomp_hand_states;
    g_dealer_states = e.dealer_states;
    g_dealer_lookups = e.dealer_lookups;
    g_dealer_hits = e.dealer_hits;
    table_free_joint(&e.joint);
    table_free_dealer(&e.dealer);
    table_free_decomp(&e.hand);
    table_free_decomp(&e.tail);
    if (e.failed) return 0.0 / 0.0; /* NaN: out of memory mid-computation */
    return ev;
}

/* Statistics from the last decomposition call, for the tests and the
 * README. The dealer statistics above are shared: they describe whichever
 * engine ran last. */
long split_decomp_first_states(void) { return g_decomp_first_states; }
long split_decomp_tail_states(void) { return g_decomp_tail_states; }
long split_decomp_hand_states(void) { return g_decomp_hand_states; }
long split_decomp_states(void) {
    return g_decomp_first_states + g_decomp_tail_states + g_decomp_hand_states;
}
