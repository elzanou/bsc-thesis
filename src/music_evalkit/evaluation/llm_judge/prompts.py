JUDGE_SYSTEM_PROMPT = """
You are evaluating the output of an AI model that was tasked with assessing a student's music performance. The model listened to an audio recording and was asked to identify any mistakes and provide corrective feedback.

You will compare the model's output against a ground truth reference. Classify the quality on two dimensions. For each, explain your reasoning before assigning the category. Different wording for the same meaning deserves full credit."""

SCORING_RUBRIC = """
### Mistake Description
How well does the predicted mistake match the ground truth?
- **correct**: Identifies the same mistake with equivalent or better detail, even if worded differently
- **partially_correct**: Identifies a related issue but misses the specific mistake, or gets the right area but wrong specifics
- **incorrect**: Completely wrong, unrelated, or fails to identify a mistake that exists

### Feedback Quality
How helpful and actionable is the predicted feedback in addressing the ground truth mistake?
- **helpful**: Directly addresses the mistake with specific, actionable advice.
- **generic**: Partially relevant advice that is too vague to act on precisely
- **unhelpful**: Irrelevant, misleading, or no useful feedback

## Output

Respond with ONLY a JSON object. Reason first, then classify.
{{"mistake_reasoning": "<explain>", "mistake": "<correct|partially_correct|incorrect>", "feedback_reasoning": "<explain>", "feedback": "<helpful|generic|unhelpful>"}}"""

JUDGE_USER_TEMPLATE = """
## Task Instruction
"{instruction}"

## Ground Truth
Mistake: "{ground_truth_mistake}"
Feedback: "{ground_truth_feedback}"

## Model Prediction
Reasoning: "{pred_reason}"
Mistake: "{pred_mistake}"
Feedback: "{pred_feedback}"

## Scoring Rubric

""" + SCORING_RUBRIC

JUDGE_RAW_USER_TEMPLATE = """
## Task Instruction
"{instruction}"

## Ground Truth
Mistake: "{ground_truth_mistake}"
Feedback: "{ground_truth_feedback}"

## Model Response (raw — may be malformed JSON, classify based on intended content)
{raw_response}

## Scoring Rubric

""" + SCORING_RUBRIC
