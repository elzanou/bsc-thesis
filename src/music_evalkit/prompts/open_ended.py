from music_evalkit.prompts.common import ROLE, TAXONOMY_SINGLE, TAXONOMY_WITH_REF

# Open-ended single audio: model predicts category + mistake + feedback
OPEN_ENDED_BASIC = f"""{ROLE}

## Task

Listen to the student's performance of the exercise described in the instruction. Evaluate whether the student performed the exercise correctly.
If the performance fulfills the instruction, report no mistake. If not, identify the most audible mistake and provide feedback to help the student improve.

## Input

You will receive:
1. An instruction describing what the student was asked to perform
2. An audio recording of the student's performance

## Mistake Taxonomy

Classify the mistake into ONE of the following categories. Use the descriptions to guide your assessment.

{TAXONOMY_SINGLE}

## Assessment Guidelines

**Reporting**
- If a mistake exists, identify EXACTLY ONE: the most clearly audible issue
- Do not combine multiple distinct issues into one description
- Report the underlying issue according to the mistake taxonomy, not secondary effects
- Provide actionable feedback to help the student correct the mistake

**Audio Grounding**
- Base your assessment strictly on what is audible in the recording
- Include specific details (note names, chord names, beat positions) ONLY when clearly identifiable from the audio

**Response Quality**
- Keep descriptions concise and factual
- Do not default to `no_mistake`, only select it when the performance fully and clearly matches the instruction.""" + """

## Output Format

Respond with ONLY raw JSON. Do NOT wrap in ```json``` code blocks. Start with { and end with }.

If there is a mistake:
{"reason": "<one sentence explaining what you heard and why it is a mistake>","category": "<category>", "mistake": "<one sentence describing what is wrong>", "feedback": "<one or two sentences on how to fix it>"}

If there is no mistake:
{"reason": "<one sentence explaining why the performance is correct>","category": "no_mistake", "mistake": null, "feedback": null}

**reason**: Your brief reasoning before committing to a classification.
- One sentence explaining what you heard and why it does or does not constitute a mistake
- Write this before deciding on category; it is your thinking step

**category**: The mistake category from the taxonomy.
- Use one of: pitch, harmony, rhythm_and_timing, tempo, articulation, dynamics, technique, no_mistake

**mistake**: The specific issue audible in the recording, if any.
- State the specific issue audible in the recording (e.g., "The 5th note is played as F instead of F#")
- Include temporal or sequential location when clearly identifiable (e.g., "in the opening phrase", "on beat 3 of the second measure", "the final chord")
- Be factual and precise about what was heard versus what was expected
- Set to null when there is no mistake

**feedback**: How to correct the issue, structured in two parts.
- First sentence: what specifically to correct (e.g., "Replace F with F# on the 3rd note.")
- Second sentence: a concrete practice strategy to apply (e.g., "Practice the scale slowly with a metronome, one note per beat, until the correct note is consistent.")
- Set to null when there is no mistake

## Examples

The following examples illustrate the expected output format and reasoning style.
Audio is described in words; apply the same reasoning to actual recordings.

### Example - Pitch mistake
**Input**:
- Instruction: Play a one-octave G major scale, ascending and descending, at a steady tempo, one note per beat.
- Audio: The scale is played at a steady tempo but the 7th note sounds a semitone flat on both the ascending and descending pass, closer to F natural than F#.
**Output**:
{"reason": "The 7th note sounds a semitone lower than expected on both ascending and descending passes, indicating a pitch error on that specific scale degree.","category": "pitch", "mistake": "The 7th note is played as F instead of F# on both the ascending and descending scale.", "feedback": "Replace F with F# on the 7th note. Practice the scale slowly with a metronome, emphasizing the half-step between the 6th and 7th notes, until F# is consistent in both directions."}

### Example - Tempo mistake
**Input**:
- Instruction: Play a one-octave E minor scale, ascending and descending, at a steady tempo, one note per beat.
- Audio: All notes have the correct pitches but the timing fluctuates throughout, some notes are rushed and others are held too long.
**Output**:
{"reason": "The pitches are correct but the timing is uneven, with notes rushed and delayed throughout, indicating an unsteady tempo.","category": "tempo", "mistake": "The tempo is uneven, with some notes rushed and others delayed.", "feedback": "Practice with a metronome set to a comfortable slow tempo, playing one note per beat. Focus on matching each note precisely to the metronome click, then gradually increase tempo only when evenness is consistent."}

### Example - Harmony mistake
**Input**:
- Instruction: Play the chord progression C, Am, F, G, one chord per beat at a steady tempo, repeating the progression twice.
- Audio: The progression is played at a steady tempo with correct timing, but the second chord sounds major throughout both repetitions.
**Output**:
{"reason": "The second chord sounds major rather than minor, indicating the student is playing A major instead of the expected A minor.","category": "harmony", "mistake": "The Am chord is played as A major in both repetitions.", "feedback": "Lower the third of the chord from C# to C to produce A minor. Practice the C to Am transition slowly until the minor quality is consistent."}

### Example - Technique mistake (guitar)
**Input**:
- Instruction: Play the chord progression Am, G, F, E, one chord per beat at a steady tempo.
- Audio: The progression is played at a steady tempo with correct chord voicings, but the third chord has one or more strings that sound muted rather than ringing out cleanly.
**Output**:
{"reason": "The pitches and timing are correct but one or more strings in the third chord are muted, indicating a fretting technique issue rather than a pitch or dynamics problem.","category": "technique", "mistake": "The F chord has one or more muted strings that fail to ring out.", "feedback": "Press the strings firmly and close to the fret so every string rings clearly. Practice the F chord in isolation, checking each string individually before moving to the progression."}

### Example - Articulation mistake
**Input**:
- Instruction: Play a one-octave C major scale, ascending and descending, in staccato, one note per beat at a steady tempo.
- Audio: The scale is played at a steady tempo with correct pitches, but all notes are connected and sustained rather than short and detached.
**Output**:
{"reason": "The notes are sustained throughout, whereas staccato requires each note to be short and detached.","category": "articulation", "mistake": "The scale is played legato instead of staccato — notes are held rather than short and detached.", "feedback": "Release each note immediately after playing to produce a short, detached sound. Practice slowly with a metronome, focusing on releasing each note cleanly before the next beat."}

### Example - Dynamics mistake
**Input**:
- Instruction: Play the chord progression C, Am, F, G, one chord per beat at a steady tempo, in piano (soft).
- Audio: The chord progression is played with correct voicings and timing, but at a consistently loud volume throughout.
**Output**:
{"reason": "The voicings and timing are correct but the performance is consistently loud, contradicting the piano dynamic specified in the instruction.","category": "dynamics", "mistake": "The performance is played forte instead of the instructed piano dynamic.", "feedback": "Use a lighter touch to produce a softer sound throughout. Practice the progression at piano level, keeping volume consistently low across all chords."}

### Example - Rhythm and timing mistake
**Input**:
- Instruction: Play a one-octave C major scale, ascending and descending, one note per beat at a steady tempo.
- Audio: The overall tempo is steady, but the 4th note is held for two beats instead of one.
**Output**:
{"reason": "The overall tempo is steady but the 4th note is held twice as long as expected, shifting all subsequent notes by one beat — this is a note duration error, not a tempo issue.","category": "rhythm_and_timing", "mistake": "The 4th note (F) is held for two beats instead of one, delaying all subsequent notes in the ascending pass.", "feedback": "Keep each note to exactly one beat. Practice with a metronome and pay special attention to releasing the 4th note on time."}"""


# Open-ended with reference audio: model predicts category + mistake + feedback
OPEN_ENDED_BASIC_WITH_REF = f"""{ROLE}

## Task

Listen to both recordings. The first recording is the reference, it demonstrates the correct performance of the exercise. The second recording is the student's performance. Evaluate whether the student performed the exercise correctly.
If the performance is correct, report no mistake. If not, identify the most audible mistake and provide feedback to help the student improve.

## Input

You will receive:
1. An instruction describing what the student was asked to perform
2. A single audio file containing two recordings separated by a short beep, a brief silence, then a distinct tone, then another brief silence:
    - **BEFORE the beep**: The reference recording
    - **AFTER the beep**: The student's performance

## Mistake Taxonomy

Classify the mistake into ONE of the following categories. Use the descriptions to guide your assessment.

{TAXONOMY_WITH_REF}

## Assessment Guidelines

**Reporting**
- If a mistake exists, identify EXACTLY ONE: the most clearly audible issue
- Do not combine multiple distinct issues into one description
- Report the underlying issue according to the mistake taxonomy, not secondary effects
- Provide actionable feedback to help the student correct the mistake

**Audio Grounding**
- Base your assessment strictly on what is audible in the recording
- Include specific details (note names, chord names, beat positions) ONLY when clearly identifiable from the audio

**Response Quality**
- Keep descriptions concise and factual
- Do not default to `no_mistake`, only select it when the student's performance fully and clearly matches the reference.""" + """

## Output Format

Respond with ONLY raw JSON. Do NOT wrap in ```json``` code blocks. Start with { and end with }.

If there is a mistake:
{"reason": "<one sentence explaining how the student differs from the reference and why it is a mistake>","category": "<category>", "mistake": "<one sentence describing what is wrong>", "feedback": "<one or two sentences on how to fix it>"}

If there is no mistake:
{"reason": "<one sentence explaining why the performance matches the reference>","category": "no_mistake", "mistake": null, "feedback": null}

**reason**: Your brief reasoning before committing to a classification.
- One sentence explaining how the student's performance compares to the reference and why it does or does not constitute a mistake
- Write this before deciding on category; it is your thinking step

**category**: The mistake category from the taxonomy.
- Use one of: pitch, harmony, rhythm_and_timing, tempo, articulation, dynamics, technique, no_mistake

**mistake**: How the student's performance differs from the reference, if at all.
- State the specific difference objectively (e.g., "The 5th note is played as F instead of F# as in the reference")
- Include temporal or sequential location when clearly identifiable (e.g., "in the opening phrase", "on beat 3 of the second measure", "the final chord")
- Be factual and precise about what was heard versus what was expected
- Set to null when there is no mistake

**feedback**: How to match the reference, structured in two parts.
- First sentence: what specifically to correct to align with the reference (e.g., "Make sure the 3rd note is F#, as in the reference.")
- Second sentence: a concrete practice strategy to apply (e.g., "Practice the passage slowly and listen to the reference after each attempt.")
- Set to null when there is no mistake

## Examples

The following examples illustrate the expected output format and reasoning style.
Audio is described in words; apply the same reasoning to actual recordings.

### Example - Pitch mistake
**Input**:
- Instruction: Listen to the reference and play the same one-octave G major scale ascending in legato, at a steady tempo.
- Audio: Reference plays all notes cleanly in legato. Student's performance matches the reference except the 7th note sounds a semitone flat, closer to F natural than F# as in the reference.
**Output**:
{"reason": "The 7th note sounds a semitone lower than in the reference, indicating a pitch error on that specific note.","category": "pitch", "mistake": "The 7th note is played as F instead of F# as in the reference.", "feedback": "Practice the scale slowly and make sure the 7th note is F#, matching the reference. Repeat until it is consistent."}

### Example - Tempo mistake
**Input**:
- Instruction: Listen to the reference and play the same chord progression in G major.
- Audio: Reference plays the chord progression at an even, steady tempo. Student's chord voicings match the reference but the tempo gradually speeds up, becoming rushed by the final repetition.
**Output**:
{"reason": "The chord voicings match the reference but the tempo drifts faster as the progression repeats, a difference not present in the reference.","category": "tempo", "mistake": "The tempo gradually speeds up instead of staying steady.", "feedback": "Practice with a metronome and match each beat to the click so your tempo stays aligned with the reference."}

### Example - Harmony mistake
**Input**:
- Instruction: Listen to the reference and play the same chord progression, one chord per beat at a steady tempo, repeating twice.
- Audio: Reference plays C, Am, F, G with correct voicings at a steady tempo. Student's performance matches the reference in timing and all chords except the second, which sounds major throughout both repetitions.
**Output**:
{"reason": "The second chord sounds major in the student's performance where the reference clearly plays it minor.","category": "harmony", "mistake": "The Am chord is played as A major instead of A minor as in the reference.", "feedback": "Lower the third of the second chord from C# to C to match the minor quality in the reference. Practice the C to Am transition slowly, comparing to the reference after each attempt."}

### Example - Technique mistake (guitar)
**Input**:
- Instruction: Listen to the reference and play the same chord progression, one chord per beat at a steady tempo.
- Audio: Reference plays Am, G, F, E with all strings ringing out cleanly. Student's performance matches the reference in voicings and timing, but the third chord has one or more strings that sound muted rather than ringing out cleanly.
**Output**:
{"reason": "The voicings and timing match the reference but one or more strings in the third chord are muted, indicating a fretting technique issue rather than a pitch or dynamics problem.","category": "technique", "mistake": "The F chord has one or more muted strings that fail to ring out, unlike the clean reference.", "feedback": "Press the strings firmly and close to the fret so every string rings clearly. Practice the F chord in isolation, checking each string individually before moving to the progression."}

### Example - Articulation mistake
**Input**:
- Instruction: Listen to the reference and play the same one-octave C major scale, ascending and descending, one note per beat at a steady tempo.
- Audio: Reference plays all notes in staccato at a steady tempo. Student's performance matches the reference in pitch and tempo but all notes are connected and sustained rather than short and detached.
**Output**:
{"reason": "The student plays legato where the reference clearly uses staccato — notes are sustained rather than short and detached.","category": "articulation", "mistake": "The scale is played legato instead of staccato as in the reference.", "feedback": "Release each note immediately after playing to produce short, detached notes matching the reference. Practice slowly, focusing on releasing each note cleanly before the next."}

### Example - Dynamics mistake
**Input**:
- Instruction: Listen to the reference and play the same chord progression, one chord per beat at a steady tempo.
- Audio: Reference plays the chord progression at a consistent piano (soft) dynamic. Student's performance matches in voicings and timing but is consistently louder than the reference throughout.
**Output**:
{"reason": "The student's performance is consistently louder than the reference, indicating a dynamics error.","category": "dynamics", "mistake": "The performance is louder than the reference throughout.", "feedback": "Use a lighter touch to match the softer dynamic of the reference. Practice the progression at a lower volume and compare to the reference after each attempt."}

### Example - Rhythm and timing mistake
**Input**:
- Instruction: Listen to the reference and play the same one-octave C major scale, ascending and descending, one note per beat at a steady tempo.
- Audio: Reference plays all notes as even quarter notes at a steady tempo. Student's overall tempo matches the reference but the 4th note is held for two beats instead of one.
**Output**:
{"reason": "The overall tempo matches the reference but the 4th note is held twice as long as in the reference — this is a note duration error, not a tempo issue.","category": "rhythm_and_timing", "mistake": "The 4th note (F) is held for two beats instead of one, unlike the even durations in the reference.", "feedback": "Match the even note durations of the reference by keeping each note to exactly one beat. Practice with a metronome and compare to the reference after each attempt."}"""
