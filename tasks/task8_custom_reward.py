"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


_DETOXIFY = None
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "does", "for", "from", "give", "how", "i", "if", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "say", "something",
    "that", "the", "their", "them", "this", "to", "was", "what", "when",
    "where", "who", "why", "with", "would", "write", "you", "your",
}
_REFUSAL_PATTERNS = (
    "i'm sorry",
    "i am sorry",
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "as an ai",
    "provide more context",
)


def _get_detoxify():
    global _DETOXIFY
    if _DETOXIFY is None:
        import torch
        from detoxify import Detoxify

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETOXIFY = Detoxify("original", device=device)
    return _DETOXIFY


def _toxicity(texts: Sequence[str]) -> list[float]:
    if not texts:
        return []
    raw = _get_detoxify().predict(list(texts))
    vals = raw["toxicity"]
    if hasattr(vals, "tolist"):
        vals = vals.tolist()
    if isinstance(vals, (list, tuple)):
        return [float(v) for v in vals]
    return [float(vals)]


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS]


def _trigram_repetition(text: str) -> float:
    words = _tokens(text)
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return min(1.0, repeats / max(1, len(trigrams) * 0.25))


def _prompt_relevance(text: str, prompt: str) -> tuple[float, float]:
    """Return (relevance, echo_penalty), both in [0, 1]."""
    prompt_words = set(_tokens(prompt))
    response_words = set(_tokens(text))
    if not prompt_words or not response_words:
        return 0.0, 0.0

    overlap = prompt_words & response_words
    relevance = min(1.0, len(overlap) / min(6, len(prompt_words)))
    echo_penalty = 1.0 if len(overlap) / max(1, len(prompt_words)) > 0.65 else 0.0
    return relevance, echo_penalty


def _detox_reward(toxicity: float) -> float:
    """Saturating anti-toxicity reward.

    Below 0.10, all benign completions receive the same reward. Between
    0.10 and 0.50 the reward decays linearly; above 0.50 it becomes
    negative to mark clearly hostile completions.
    """
    if toxicity <= 0.10:
        return 1.0
    if toxicity < 0.50:
        return 1.0 - ((toxicity - 0.10) / 0.40)
    return -min(1.0, (toxicity - 0.50) / 0.50)


def _template_penalty(text: str) -> float:
    t = " ".join(text.lower().split())
    return 1.0 if any(pat in t for pat in _REFUSAL_PATTERNS) else 0.0


def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion. Higher = better.

    The verl reward worker calls this once per training step with the
    flattened list of K-rollouts across the prompt batch.

    Args:
        texts: completions to score, one entry per completion.
        prompts: same-length list of the originating prompts (the verl
            dispatcher uses ``reward_score.prompt_conditioned`` below
            to decide whether to pass these). Set the attribute to
            ``False`` if your design is purely response-side.

    Returns:
        ``list[float]`` of the same length as ``texts``. Higher = better.

    See the top-of-file docstring for design hints (saturating Detoxify,
    repetition penalty, length-cap penalty, prompt-relevance, blending
    with your RM).
    """
    # <MY CODE HERE>
    texts = list(texts)
    if prompts is None:
        prompts = [""] * len(texts)
    if len(prompts) != len(texts):
        raise ValueError(f"len(prompts)={len(prompts)} != len(texts)={len(texts)}")

    tox_scores = _toxicity(texts)
    rewards: list[float] = []
    for text, prompt, tox in zip(texts, prompts, tox_scores):
        detox = _detox_reward(tox)
        relevance, echo_penalty = _prompt_relevance(text, prompt)
        repetition = _trigram_repetition(text)
        hit_cap = 1.0 if len(text) >= 240 else 0.0
        template = _template_penalty(text)
        too_short = 1.0 if len(_tokens(text)) < 6 else 0.0

        reward = (
            0.75 * detox
            + 0.25 * relevance
            - 0.35 * repetition
            - 0.25 * hit_cap
            - 0.25 * echo_penalty
            - 0.20 * template
            - 0.15 * too_short
        )
        rewards.append(float(max(-1.0, min(1.0, reward))))
    return rewards


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
