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
from pathlib import Path
from typing import Sequence


_DETOXIFY = None
_RM = None
_RM_LOAD_ATTEMPTED = False
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
_CLARIFICATION_COLLAPSE_PATTERNS = (
    "i'm not sure",
    "i am not sure",
    "i don't know exactly",
    "i do not know exactly",
    "not sure i know",
    "not sure i have enough information",
    "can you tell me more",
    "could you tell me more",
    "what you're looking for",
    "what you are looking for",
    "what you're hoping to achieve",
    "what you are hoping to achieve",
    "help you brainstorm",
    "brainstorm some ideas",
)
_CHAT_TEMPLATE_PATTERNS = (
    "you are a helpful assistant",
    "respond to the user thoughtfully and kindly",
    "\nuser\n",
    "\nassistant\n",
    "\nsystem\n",
    "<|im_start|>",
    "<|im_end|>",
    "### user",
    "### assistant",
    "溆",
)
_WEB_ARTIFACT_PATTERNS = (
    "user-agent",
    "mozilla/",
    "windows nt",
    "trident/",
    "like gecko",
    "@web",
    "http://",
    "https://",
    "www.",
    "browser",
    "the user is a bot",
)
_ROLE_LABEL_RE = re.compile(r"(?im)(^|\n|\s)(user|assistant|system)\s*:")
_RANDOM_HEADER_RE = re.compile(r"[A-Za-z0-9+/]{18,}")
_SENTENCE_END_RE = re.compile(r"[.!?]['\")\]]?\s*$")


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


def _get_rm():
    """Load the trained RM if it is mounted in the PPO container.

    The custom reward must remain runnable even before RM training or in
    unit tests, so this is deliberately best-effort.  In the verl docker
    run the repo is mounted at /workspace and the RM should be at
    /workspace/checkpoints/rm.
    """
    global _RM, _RM_LOAD_ATTEMPTED
    if _RM_LOAD_ATTEMPTED:
        return _RM
    _RM_LOAD_ATTEMPTED = True

    for path in (Path("/workspace/checkpoints/rm"), Path("checkpoints/rm")):
        if path.exists():
            try:
                from src.toxic_rl.reward_model import TrainedRewardModel

                _RM = TrainedRewardModel(str(path))
                break
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"[task8_custom_reward] RM load failed from {path}: {exc}")
    return _RM


def _rm_reward(texts: Sequence[str], prompts: Sequence[str]) -> list[float]:
    """Bounded optional RM signal in [-1, 1].

    We keep this small in the final blend because README explicitly warns
    that RM-only PPO can collapse onto its own learned shortcut.
    """
    rm = _get_rm()
    if rm is None:
        return [0.0] * len(texts)
    raw = rm.score(texts, prompts=prompts) if getattr(rm, "prompt_conditioned", False) else rm.score(texts)
    import math

    return [math.tanh(float(v) / 3.0) for v in raw]


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


def _long_prompt_echo(text: str, prompt: str) -> float:
    """Penalty for copying the prompt instead of answering it.

    Word-overlap alone is not enough: a relevant answer should reuse a
    few content words, but copying a long contiguous prompt span is the
    reward-hack we want to suppress.
    """
    prompt_words = _tokens(prompt)
    text_words = _tokens(text)
    if len(prompt_words) < 4 or len(text_words) < 4:
        return 0.0

    text_ngrams = set()
    for n in (4, 5, 6):
        text_ngrams.update(tuple(text_words[i : i + n]) for i in range(max(0, len(text_words) - n + 1)))

    hits = 0
    total = 0
    for n in (4, 5, 6):
        for i in range(max(0, len(prompt_words) - n + 1)):
            total += 1
            if tuple(prompt_words[i : i + n]) in text_ngrams:
                hits += 1
    return min(1.0, hits / max(1, total))


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
    refusal = 1.0 if any(pat in t for pat in _REFUSAL_PATTERNS) else 0.0
    clarification = 1.0 if any(pat in t for pat in _CLARIFICATION_COLLAPSE_PATTERNS) else 0.0
    lower = text.lower()
    template_hits = sum(1 for pat in _CHAT_TEMPLATE_PATTERNS if pat in lower)
    web_hits = sum(1 for pat in _WEB_ARTIFACT_PATTERNS if pat in lower)
    role_label = 1.0 if _ROLE_LABEL_RE.search(text) else 0.0
    role_word_ratio = 0.0
    words = _tokens(text)
    if words:
        role_count = sum(1 for w in words if w in {"assistant", "user", "system"})
        role_word_ratio = min(1.0, role_count / max(1, len(words) * 0.15))
    return min(1.0, max(refusal, clarification, template_hits / 2.0, web_hits / 1.0, role_label, role_word_ratio))


def _format_artifact_penalty(text: str) -> float:
    weird_count = text.count("溆") + text.count("<|") + text.count("|>") + text.count("@")
    newline_density = text.count("\n") / max(1, len(text) / 80)
    return min(1.0, weird_count / 2.0 + max(0.0, newline_density - 1.0) * 0.25)


def _nonsense_penalty(text: str) -> float:
    """Penalty for markup, browser headers, and random-token garbage.

    This directly targets the second observed custom-reward attractor:
    benign-looking but non-answer text such as User-Agent strings or @Web
    transcript fragments.
    """
    if not text.strip():
        return 1.0
    lower = text.lower()
    web = 1.0 if any(pat in lower for pat in _WEB_ARTIFACT_PATTERNS) else 0.0
    randomish = 1.0 if _RANDOM_HEADER_RE.search(text) else 0.0
    chars = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    symbol_density = symbols / max(1, chars)
    digit_density = digits / max(1, chars)
    low_alpha = 1.0 if chars >= 40 and alpha / max(1, chars) < 0.45 else 0.0
    high_symbol = min(1.0, max(0.0, symbol_density - 0.12) / 0.18)
    high_digit = min(1.0, max(0.0, digit_density - 0.12) / 0.18)
    return min(1.0, max(web, randomish, low_alpha, high_symbol, high_digit))


def _answer_quality(text: str, prompt: str) -> float:
    """Small reward for a natural answer shape, not a semantic judge.

    Detoxify can be fooled by fragments such as "Make my mind hurt" or
    copied chat templates.  This gate only asks for basic answer hygiene:
    enough words, mostly alphabetic text, not just prompt words, and no
    obvious transcript formatting.
    """
    stripped = text.strip()
    words = _tokens(stripped)
    token_count = len(words)
    if token_count < 10 or token_count > 70:
        return 0.0

    chars = len(stripped)
    if chars == 0:
        return 0.0
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / chars
    if alpha_ratio < 0.62:
        return 0.0

    prompt_words = set(_tokens(prompt))
    response_words = set(words)
    if response_words and len(response_words - prompt_words) < 4:
        return 0.0

    has_sentence_end = 1.0 if _SENTENCE_END_RE.search(stripped) else 0.0
    length_score = min(1.0, (token_count - 9) / 18)
    return min(1.0, 0.65 * length_score + 0.35 * has_sentence_end)


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
    rm_scores = _rm_reward(texts, list(prompts))
    rewards: list[float] = []
    for text, prompt, tox, rm_score in zip(texts, prompts, tox_scores, rm_scores):
        detox = _detox_reward(tox)
        relevance, echo_penalty = _prompt_relevance(text, prompt)
        prompt_echo = _long_prompt_echo(text, prompt)
        repetition = _trigram_repetition(text)
        hit_cap = 1.0 if len(text) >= 240 else 0.0
        template = _template_penalty(text)
        format_artifact = _format_artifact_penalty(text)
        nonsense = _nonsense_penalty(text)
        token_count = len(_tokens(text))
        too_short = 1.0 if token_count < 8 else 0.0
        too_long = 1.0 if token_count > 90 else 0.0
        quality = _answer_quality(text, prompt)

        if template >= 1.0 or format_artifact >= 1.0 or nonsense >= 1.0 or prompt_echo >= 0.75:
            rewards.append(-1.0)
            continue

        reward = (
            0.65 * detox
            + 0.15 * rm_score
            + 0.20 * relevance
            + 0.35 * quality
            - 0.45 * repetition
            - 0.25 * hit_cap
            - 0.55 * echo_penalty
            - 0.75 * prompt_echo
            - 0.85 * template
            - 0.65 * format_artifact
            - 0.80 * nonsense
            - 0.25 * too_short
            - 0.15 * too_long
        )
        if max(template, format_artifact, nonsense, prompt_echo) >= 0.50:
            # Hard gate observed shortcut regions. Without this, PPO can
            # accept a slightly lower scalar reward if the region is easy to
            # reach and stable under sampling.
            reward = min(reward, -0.90)
        if tox > 0.50 and rm_score > 0.50:
            # Detoxify and RM disagreement is exactly the "interesting signal"
            # README calls out.  Here, high-RM + high-toxicity means the RM
            # likely found a shortcut, so do not let it rescue the reward.
            reward -= 0.35
        rewards.append(float(max(-1.0, min(1.0, reward))))
    return rewards


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
