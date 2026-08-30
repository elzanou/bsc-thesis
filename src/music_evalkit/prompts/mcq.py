from textwrap import dedent

from music_evalkit.prompts.common import ROLE, TAXONOMY_SINGLE, TAXONOMY_WITH_REF

# MCQ: single audio
MCQ_TEMPLATE = f"""{ROLE}

## Task

Listen to the student's performance of the exercise described in the instruction. Evaluate whether the student performed the exercise correctly and select the answer choice that best matches your assessment.

## Input

You will receive:
1. An instruction describing what the student was asked to perform
2. An audio recording of the student's performance
3. Four answer choices labeled A, B, C, and D

## Mistake Taxonomy

The answer choices use the following category names. Use these descriptions to match what you hear:

{TAXONOMY_SINGLE}

## Assessment Guidelines

**Selection Process**
- Listen carefully to the entire performance before making your selection
- Evaluate each answer choice systematically and select the ONE that best matches what you hear
- If no mistake is present, select the choice indicating correct performance
- Select the answer that describes the underlying issue, not secondary effects

**Audio Grounding**
- Base your selection strictly on what is audible, do not infer issues that are not clearly present in the recording

## Examples

The following examples show the exact output format expected. Audio is described in words; apply the same reasoning to actual recordings.

### Example 1 — No mistake
**Instruction**: Play a one-octave A harmonic minor scale, ascending and descending, at a steady tempo, one note per beat on piano.
**Choices**: A) no_mistake  B) pitch  C) tempo  D) rhythm_and_timing
**Audio**: All notes of the A harmonic minor scale are played correctly, ascending and descending at a steady, even pace.
**Output**:
{{"reason": "All notes are accurate and the performance is clean with a steady tempo, no errors detected.", "answer": "A"}}

### Example 2 — Pitch error
**Instruction**: Play a one-octave C major scale, ascending and descending, at a steady tempo, one note per beat on guitar.
**Choices**: A) harmony  B) pitch  C) no_mistake  D) rhythm_and_timing
**Audio**: The tempo is steady and the rhythm is consistent, but one note in the ascending scale sounds clearly sharp, the student appears to have played the wrong fret.
**Output**:
{{"reason": "One note in the ascending scale is noticeably sharp, indicating a pitch error.", "answer": "B"}}

### Example 3 — Harmony error
**Instruction**: Play the chord progression Am, F, C, G, one chord per beat, at a steady tempo on guitar.
**Choices**: A) pitch  B) technique  C) harmony  D) no_mistake
**Audio**: The tempo is steady and all chords ring out cleanly, but the first chord sounds major instead of minor.
**Output**:
{{"reason": "The Am chord sounds major instead of minor, indicating a harmony error.", "answer": "C"}}

### Example 4 — Articulation error
**Instruction**: Play a one-octave G major scale, ascending and descending, in staccato, one note per beat on piano.
**Choices**: A) articulation  B) rhythm_and_timing  C) dynamics  D) no_mistake
**Audio**: The scale is played with correct pitches and a steady tempo, but all notes are sustained and connected rather than short and detached.
**Output**:
{{"reason": "The notes are played legato instead of staccato, which is an articulation error.", "answer": "A"}}

### Example 5 — Tempo error
**Instruction**: Play a one-octave D major scale, ascending and descending, at a steady tempo, one note per beat on guitar.
**Choices**: A) rhythm_and_timing  B) no_mistake  C) dynamics  D) tempo
**Audio**: All notes are accurate and each note lasts one beat, but the speed fluctuates throughout, the student rushes in the ascending pass and slows down in the descending pass.
**Output**:
{{"reason": "The tempo is unsteady, speeding up and slowing down across the scale, which is a tempo error.", "answer": "D"}}

### Example 6 — Rhythm and timing error
**Instruction**: Play the chord progression C, G, Dm, Em, one chord per beat, at a steady tempo on guitar.
**Choices**: A) pitch  B) rhythm_and_timing  C) no_mistake  D) tempo
**Audio**: The tempo is steady and all chords sound correct in pitch, but the third chord enters a beat early, throwing off the timing for the rest of the progression.
**Output**:
{{"reason": "The third chord enters on the wrong beat, which is a rhythm and timing error, not a tempo issue since the overall speed remains consistent.", "answer": "B"}}

## Output Format

You MUST respond with ONLY a single JSON object. No other text, no code blocks, no explanation outside the JSON.

{{"reason": "<one sentence explaining what you heard and why you selected this answer>", "answer": "<letter>"}}

The "reason" field is your reasoning step, write what you heard before committing to an answer.
The "answer" field MUST be a single letter: A, B, C, or D. Do NOT write the category name, write the letter only."""

# MCQ with reference audio
MCQ_TEMPLATE_WITH_REF = f"""{ROLE}

## Task

Listen to both recordings. The first recording is the reference, it demonstrates the correct performance of the exercise. The second recording is the student's performance. Evaluate whether the student performed the exercise correctly and select the answer choice that best matches your assessment.

## Input

You will receive:
1. An instruction describing what the student was asked to perform
2. A single audio file containing two recordings separated by a short beep, a brief silence, then a distinct tone, then another brief silence:
    - **BEFORE the beep**: The reference recording
    - **AFTER the beep**: The student's performance
3. Four answer choices labeled A, B, C, and D

## Mistake Taxonomy

The answer choices use the following category names. Use these descriptions to match what you hear compared to the reference:

{TAXONOMY_WITH_REF}

## Assessment Guidelines

**Selection Process**
- Listen carefully to both recordings in their entirety
- Evaluate each answer choice systematically and select the ONE that best matches what you hear
- If no mistake is present, select the choice indicating correct performance
- Select the answer that describes the underlying issue, not secondary effects

**Audio Grounding**
- Base your selection strictly on what is audible, do not infer issues that are not clearly present in the recording

## Examples

The following examples show the exact output format expected. Audio is described in words; apply the same reasoning to actual recordings.

### Example 1 — No mistake
**Instruction**: Listen to the reference and play the same excerpt from "Yesterday" on guitar.
**Choices**: A) tempo  B) pitch  C) technique  D) no_mistake
**Audio**: Reference plays the melody cleanly at a steady tempo. The student's performance matches the reference in pitch, timing, and tone, no audible differences.
**Output**:
{{"reason": "The student's performance matches the reference in all aspects, no errors detected.", "answer": "D"}}

### Example 2 — Pitch error
**Instruction**: Listen to the reference and play the same one-octave G major scale, ascending and descending, on piano.
**Choices**: A) pitch  B) tempo  C) no_mistake  D) rhythm_and_timing
**Audio**: The reference plays all notes of the G major scale cleanly at a steady tempo. The student matches the tempo and rhythm, but the 7th note sounds natural (F) instead of sharp (F#), clearly differing from the reference.
**Output**:
{{"reason": "The 7th scale degree sounds as F natural instead of F# compared to the reference, indicating a pitch error.", "answer": "A"}}

### Example 3 — Harmony error
**Instruction**: Listen to the reference and play the same excerpt from "Wonderwall" on guitar.
**Choices**: A) no_mistake  B) rhythm_and_timing  C) pitch  D) harmony
**Audio**: Reference plays the chord progression with correct voicings throughout. Compared to the reference, one chord in the student's performance sounds clearly different, it has the wrong quality, sounding major where the reference plays minor.
**Output**:
{{"reason": "One chord sounds major instead of minor compared to the reference, indicating a harmony error.", "answer": "D"}}

### Example 4 — Rhythm and timing error
**Instruction**: Listen to the reference and play the same chord progression (Am, F, C, G), one chord per beat, at a steady tempo on guitar.
**Choices**: A) tempo  B) rhythm_and_timing  C) pitch  D) no_mistake
**Audio**: Reference plays all four chords with even, on-beat strums at a consistent tempo. Compared to the reference, the student keeps the same overall speed but enters the third chord a beat early, throwing off the timing for the rest of the progression.
**Output**:
{{"reason": "The student enters the third chord a beat early compared to the reference, which is a rhythm and timing error, not a tempo issue since the overall speed remains consistent.", "answer": "B"}}

### Example 5 — Dynamics error
**Instruction**: Listen to the reference and play the same excerpt from "Imagine" on piano.
**Choices**: A) articulation  B) no_mistake  C) dynamics  D) tempo
**Audio**: Reference plays the passage softly and delicately. The student plays the correct notes and tempo, but the entire passage is noticeably louder than the reference throughout.
**Output**:
{{"reason": "The student plays much louder than the reference throughout, which is a dynamics error.", "answer": "C"}}

## Output Format

You MUST respond with ONLY a single JSON object. No other text, no code blocks, no explanation outside the JSON.

{{"reason": "<one sentence explaining what you heard and why you selected this answer>", "answer": "<letter>"}}

The "reason" field is your reasoning step, write what you heard before committing to an answer.
The "answer" field MUST be a single letter: A, B, C, or D. Do NOT write the category name, write the letter only."""
