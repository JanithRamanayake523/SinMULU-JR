"""Model-independent scoring: option-order averaging and contextual calibration on top of any label scorer.

A scorer is a callable scorer(prompts, labels, answer_word) -> per prompt the 4 log-probs log P(label | prompt) over
the full vocabulary; a prompt is the turn tuple from prompts.build(); answer_word is the pre-filled answer marker.
"""
import math

from . import prompts as P
from .data import cyclic_perms, permute_options

CONTENT_FREE = ["N/A", "", "[MASK]"]  # contextual calibration inputs (Zhao et al., 2021)


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def calibration_priors(rows, scorer, prompt):
    """Mean label log-probs over content-free inputs; one prior per subject name when the prompt shows the subject."""
    p = P.PROMPTS[prompt]
    reps = {}
    for r in rows:
        reps.setdefault(p.subject(r) if p.subject else "", r)
    priors = {}
    for k, rep in sorted(reps.items()):
        lps = scorer([P.build(P.content_free(rep, t), prompt) for t in CONTENT_FREE], p.labels, p.answer_word)
        priors[k] = [sum(l[j] for l in lps) / len(lps) for j in range(4)]
    return priors


def score_rows(rows, scorer, prompt, perms=1, calibrate=False):
    """Returns (probs, stats). probs[i] = 4 probabilities in the ORIGINAL option order.
    perms = number of cyclic option orders averaged (1-4). stats: label_mass mean/min = total probability on the four
    label tokens before renormalising (low mass = the model wants to say something else)."""
    p = P.PROMPTS[prompt]
    priors = calibration_priors(rows, scorer, prompt) if calibrate else None
    probs = [[0.0] * 4 for _ in rows]
    masses = []
    orders = cyclic_perms(4)[:perms]
    for perm in orders:
        lps = scorer([P.build(permute_options(r, perm), prompt) for r in rows], p.labels, p.answer_word)
        for i, lp in enumerate(lps):
            masses.append(sum(math.exp(x) for x in lp))
            if priors:
                prior = priors[p.subject(rows[i]) if p.subject else ""]
                lp = [a - b for a, b in zip(lp, prior)]
            for pos, q in enumerate(softmax(lp)):
                probs[i][perm[pos]] += q / len(orders)
    return probs, {"label_mass_mean": sum(masses) / len(masses), "label_mass_min": min(masses)}


def argmax1(p):
    """1-based predicted option."""
    return max(range(len(p)), key=lambda j: p[j]) + 1
