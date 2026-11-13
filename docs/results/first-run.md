# Amber v0, first run: an honest negative result

Amber v0 asked: can a model that never sees your clothing choices (your art taste, your personality, your own words) predict which clothes you pick? One person, the owner, played the real run on 2026-11-13 (the owner's run). Numbers below come from `GET /api/users/<id>/results`.

**Verdict: no.** The art you scored 52.5% over 40 pairs, which is a coin flip. The owner also rejected the experience (see the end). Amber v0 is closed and v1 drops the quiz.

## Scoreboard

Each "you" predicted every one of the 40 random clothing pairs before the answer was accepted. Chance is 50%. 95% CIs are bootstrap intervals over pairs. `revealed` abstains on its first 5 pairs and gets 0.5 credit for each.

| You | Sees clothing choices? | All 40 | 95% CI | Last 20 | 95% CI |
|---|---|---|---|---|---|
| stated | no | 52.5% | 37.5 to 67.5 | 30% | 10 to 50 |
| bigfive | no | 37.5% | 22.5 to 52.5 | 35% | 15 to 55 |
| art | no | 52.5% | 37.5 to 67.5 | 70% | 50 to 90 |
| revealed | yes, earlier picks only | 66.25% | 52.5 to 80 | 60% | 40 to 80 |

How to read it:

- Over all 40, only `revealed` has an interval that clears 50%, and it is the one you that learns from clothing picks. That is a model learning from the very thing it predicts, not evidence for the idea.
- `art` (the headline you) is exactly 52.5%, with an interval from 37.5 to 67.5.
- The last-20 slice looks better for `art` (70%), but 20 pairs is far too few to read anything into it, and it is the same 40 pairs sliced a second way. Over all 40 the art you is at chance.
- `bigfive` at 37.5% is below chance, which at n = 40 is also well within noise.

## The yous do not agree with each other

Cosine similarity between the yous (1 means the same direction, 0 means unrelated):

| | art | bigfive | stated | revealed |
|---|---|---|---|---|
| art | 1.00 | -0.02 | -0.03 | -0.01 |
| bigfive | -0.02 | 1.00 | -0.06 | -0.05 |
| stated | -0.03 | -0.06 | 1.00 | 0.06 |
| revealed | -0.01 | -0.05 | 0.06 | 1.00 |

Every pair is within 0.06 of zero. In this space the yous point in unrelated directions, including the art you and the you learned from the owner's actual picks.

## The readout, quoted as a failure

The readout turns each you into words. For this run:

- **You said:** classic, cozy, minimal, streetwear, elegant.
- **You picked:** business, formal, elegant, tailored, sophisticated (plus brown, purple, white, formal shoes, sunglasses, shirts).
- **Your art taste says:** animal print, grunge, tie-dye, punk, monochrome.

Art says animal print, grunge and tie-dye. The owner picked business, formal and elegant. The art you did not recover the owner's clothing taste, and its words do not resemble the picks.

## What the owner said

> "low quality photos, way too long onboarding, the choices were bs like there were so many times when i didn't like both of them. i think quizzing the user is a very big waste of time."

Three product problems (small thumbnails, a long onboarding, random pairs with no acceptable option) and one design verdict: asking people questions to learn their taste costs more attention than it is worth. That verdict is what moved the project to learning from decisions people have already made (see the README).

## Caveats on Amber's own reporting

These come from an external review of the code. Amber's bug list was not fixed; v0 is closed as it ran.

- **The retest "ceiling" is really retest agreement.** Agreement between two answers is not an upper bound on prediction accuracy: someone who picks A 80% of the time agrees with themselves 68% of the time, but always guessing A scores 80%. The API field is still named `ceiling` (null here, because the retest is not due until 2026-11-20). Read any normalized score built on it as a descriptive rescaling, not "the share of you recovered".
- **The bootstrap CI degenerates at tiny n.** One correct prediction yields the interval [1.0, 1.0], and so does any all-success sample. The intervals above are from n = 40 and are still rough. For fixed binary evaluation a binomial interval is the right tool, and it should also account for abstention credit and for the model adapting as it goes.
- **"Surprise" lines overstate the evidence.** "Surprise: you pick business but never said it" treats leaving a word out of at most five tapped style words as a contradiction. It is not. The surprises shown for this run (business, formal, tailored) are descriptive only.
- **n = 1.** One person, one run.
