"""
Generates Socratic feedback questions based on a prediction result.

IMPORTANT (by design, per the project requirements): this module NEVER gives
the student the "right answer". It only asks reasoning questions intended to
help the student improve their own argument - it is kept completely separate
from prediction.py (model loading/inference lives there, not here).

Used by: app/app.py
"""
from __future__ import annotations

from typing import List

# Generic questions that are always useful, regardless of prediction.
_GENERIC_QUESTIONS = [
    "What is your main claim?",
    "What evidence supports your claim?",
    "Is your evidence directly related to your claim?",
    "Can you provide stronger or more specific evidence?",
    "Is there another possible viewpoint someone might raise?",
    "How could you make your argument clearer?",
]

# Extra questions tailored to the predicted discourse_type, used in addition
# to (not instead of) the generic questions above.
_TYPE_QUESTIONS = {
    "Lead": [
        "Does your opening grab the reader's attention before you state your position?",
    ],
    "Position": [
        "Have you clearly stated what you believe, in one direct sentence?",
    ],
    "Claim": [
        "Does this claim directly support your overall position?",
    ],
    "Counterclaim": [
        "Have you fairly represented the opposing viewpoint, rather than a weak version of it?",
    ],
    "Rebuttal": [
        "Does your rebuttal clearly explain WHY the opposing view does not outweigh your position?",
    ],
    "Evidence": [
        "Where does this evidence come from, and would a reader find that source convincing?",
    ],
    "Concluding Statement": [
        "Does your conclusion clearly restate your position and your strongest reasons?",
    ],
}

# Extra questions tailored to the predicted effectiveness.
_EFFECTIVENESS_QUESTIONS = {
    "Ineffective": [
        "This point seems underdeveloped - what is one specific detail you could add to strengthen it?",
        "Could you explain WHY this point matters to your overall argument?",
    ],
    "Adequate": [
        "This point works, but could it be more convincing - what would make it stronger?",
    ],
    "Effective": [
        "This point is working well - could you use a similar approach elsewhere in your essay?",
    ],
}


def get_questions(discourse_type: str | None, effectiveness: str | None, max_questions: int = 6) -> List[str]:
    """
    Builds an ordered list of Socratic questions:
      1. Effectiveness-specific questions (most actionable, shown first).
      2. Discourse-type-specific questions.
      3. Generic argument-structure questions, to fill up to max_questions.
    Duplicate questions are removed while preserving order.
    """
    questions: List[str] = []

    if effectiveness in _EFFECTIVENESS_QUESTIONS:
        questions.extend(_EFFECTIVENESS_QUESTIONS[effectiveness])
    if discourse_type in _TYPE_QUESTIONS:
        questions.extend(_TYPE_QUESTIONS[discourse_type])
    questions.extend(_GENERIC_QUESTIONS)

    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    return unique_questions[:max_questions]


def get_feedback_intro(discourse_type: str | None, effectiveness: str | None) -> str:
    """One short, neutral sentence introducing the questions - never a verdict,
    just a prompt to reflect (keeps with the 'ask, don't tell' Socratic approach)."""
    parts = []
    if discourse_type:
        parts.append(f"this looks like a '{discourse_type}' segment")
    if effectiveness:
        parts.append(f"the model's current assessment is '{effectiveness}'")
    detail = " and ".join(parts)
    if detail:
        return f"Based on the analysis ({detail}), consider these questions as you revise:"
    return "Consider these questions as you revise your argument:"
