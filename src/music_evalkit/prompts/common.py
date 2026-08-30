ROLE = "You are an expert music teacher assessing student performances."

# Category definitions for single-audio prompts (described relative to instruction/expected).
TAXONOMY_SINGLE = """\
- **pitch**: Issues with note accuracy or register. Listen for:
  - A specific note or passage that sounds clearly out of tune with the expected pitch
  - A melody or passage played in the wrong octave
  Do not flag this if all notes sound equally and consistently off, that likely reflects instrument tuning, not a playing mistake.

- **harmony**: Issues with chord accuracy. Listen for:
  - A chord built on the wrong root note
  - A chord with the correct root but the wrong quality (e.g., major instead of minor)

- **rhythm_and_timing**: Issues with rhythmic patterns or synchronization. Listen for:
  - A rhythmic pattern that differs from what was expected (e.g., dotted rhythm played as even notes, or a note held for the wrong duration)
  - A note or chord that enters on the wrong beat
  - Two parts that lose synchronization (e.g., melody and accompaniment)
  Use this category when individual notes or chords have incorrect timing, wrong duration, wrong beat entry, or desynchronization.

- **tempo**: Issues with speed or consistency. Listen for:
  - A performance that is noticeably faster or slower than the intended tempo
  - A tempo that fluctuates with unintended speeding up or slowing down (e.g., consistently too fast throughout, or accelerating and decelerating across the performance)
  Use this category when the overall pace or steadiness is incorrect (too fast, too slow, or inconsistently accelerating/decelerating), but notes fall in their expected relative positions.

- **articulation**: Issues with how notes are connected. Listen for:
  - Notes that are connected or separated differently than expected (e.g., legato played as staccato, or staccato played as legato)

- **dynamics**: Issues with volume levels. Listen for:
  - A performance that is consistently louder or softer than intended (e.g., piano played instead of forte, or forte played instead of piano)
  - Unintentional unevenness in loudness between notes that should be at a consistent volume

- **technique**: Issues with sound quality caused by improper technique (guitar only). For piano performances, do not use this category. Listen for:
  - Prominent fret buzz on a note or chord that clearly disrupts the sound
  - One or more notes that are clearly muted and fail to ring out

- **no_mistake**: No errors detected; the student performed the exercise correctly. Non-musical factors such as instrument tuning, recording quality, instrument timbre, or background noise do not count as mistakes."""

# Category definitions for reference-audio prompts (described relative to reference recording).
TAXONOMY_WITH_REF = """\
- **pitch**: Issues with note accuracy or register. Listen for:
  - A specific note or passage that sounds different in pitch compared to the reference
  - A melody or passage played in a different octave than the reference
  Do not flag this if all notes sound equally and consistently offset from the reference, that likely reflects instrument tuning, not a playing mistake.

- **harmony**: Issues with chord accuracy. Listen for:
  - A chord built on a different root note than in the reference
  - A chord with the same root but a different quality (e.g., major instead of minor)

- **rhythm_and_timing**: Issues with rhythmic patterns or synchronization. Listen for:
  - A rhythmic pattern that differs from the reference (e.g., dotted rhythm played as even notes, or a note held for the wrong duration)
  - A note or chord that enters on the wrong beat relative to the reference
  - Two parts that lose synchronization in a way not present in the reference (e.g., melody and accompaniment)
  Use this category when individual notes or chords have incorrect timing, wrong duration, wrong beat entry, or desynchronization.

- **tempo**: Issues with speed or consistency. Listen for:
  - A performance that is noticeably faster or slower than the reference
  - A tempo that fluctuates compared to the steady reference (e.g., consistently too fast throughout, or accelerating and decelerating across the performance)
  Use this category when the overall pace or steadiness is incorrect (too fast, too slow, or inconsistently accelerating/decelerating), but notes fall in their expected relative positions.

- **articulation**: Issues with how notes are connected. Listen for:
  - Notes that are connected or separated differently than in the reference (e.g., legato in the reference played as staccato by the student)

- **dynamics**: Issues with volume levels. Listen for:
  - A performance that is consistently louder or softer than the reference (e.g., piano played instead of forte, or forte played instead of piano)
  - Unintentional unevenness in loudness not present in the reference

- **technique**: Issues with sound quality caused by improper technique (guitar only). For piano performances, do not use this category. Listen for:
  - Prominent fret buzz on a note or chord that clearly disrupts the sound
  - One or more notes that are clearly muted and fail to ring out

- **no_mistake**: No errors detected; the student performed the exercise correctly. Non-musical differences from the reference such as instrument tuning, recording quality, instrument timbre, or background noise do not count as mistakes."""
