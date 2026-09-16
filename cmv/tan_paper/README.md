# Tan et al. 2016 — the approach

> **Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith
> Online Discussions.** Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil,
> Lillian Lee. WWW 2016. [arXiv:1602.01103](https://arxiv.org/abs/1602.01103)

Notes on the paper our ChangeMyView benchmark is built from, written for someone
deciding what to reuse and what to compare against. Everything here is from the
paper; our own results live in [`../README.md`](../README.md).

## Why this dataset exists

On /r/ChangeMyView a user posts a view and invites others to argue against it. If a
reply changes their mind, the poster awards it a **delta**. The labels are therefore
produced by the person actually being persuaded, in the ordinary course of the
discussion. No annotation task was ever run, no annotators hired, no inter-annotator
agreement to establish. That is the unusual strength of the corpus.

The released data covers January 2013 to August 2015: 18,363 training discussions and
2,263 heldout ones, split by **time**, not at random.

## Three questions, three sections

| Section | Question | Result |
|---|---|---|
| 3 — Interaction dynamics | Who succeeds, and when? | Descriptive, no classifier |
| 4 — Language | Which of two replies won? | **65.1%** vs 59.6% baseline |
| 5 — Malleability | Which views are open to change? | 0.54 AUC, near chance |

Only Section 4 is the pair task we benchmark. The other two are worth knowing because
they bound what the language signal can be.

## Section 3: interaction dynamics

Findings about *position in the conversation*, independent of what anyone said.

- **Arriving early matters enormously.** The first two challengers are about **three
  times** more likely to succeed as the tenth, even after controlling for user
  experience.
- **Long arguments are a losing sign.** The relationship with back-and-forth depth is
  non-monotonic: some engagement predicts success, but after **five rounds** the
  challenger has virtually no chance of a delta. Sustained debate signals futility
  rather than progress.
- **Diversity beats volume.** More challengers raise the chance of conversion, but
  sublinearly; within a single-topic subtree, multiple challengers do *worse* than one.

None of this is available to a method that reads only the text of two replies. It is a
reminder that a large part of what determines a delta is structural, and the pair task
deliberately holds it constant.

## Section 4: the pair task — what we benchmark

### Construction

For each discussion, take the two most **lexically similar** root replies — Jaccard
similarity after stopword removal — one that won a delta and one that did not. Matching
on similarity is what forces the comparison onto *how* the argument is made rather than
what it is about.

Filters applied: at least 10 challengers in the tree, at least 3 unsuccessful
counterarguments before the poster's final reply, and root replies of at least 50 words.

**3,456 training pairs / 807 heldout pairs.**

### Method

An **ℓ1-regularised logistic regression making a paired comparison** — it sees both
replies and predicts which one won, rather than scoring each independently. Parameters
chosen by five-fold cross-validation, with all pairs sharing an original post kept
**inside the same fold** to prevent leakage. A single experimental run on the heldout
set, deliberately, to limit overfitting.

### Features

All four groups are countable surface quantities. No semantic understanding, no
embeddings — the paper predates usable sentence embeddings and, more to the point, wants
findings it can name.

| Group | Contents |
|---|---|
| **Interplay** | Word overlap between reply and original post, over three vocabularies (stopwords, content words, all). Four quantities each: shared-word count, that count as a fraction of the reply, as a fraction of the post, and Jaccard |
| **Style** | Articles, positive/negative sentiment words, 1st/2nd person pronouns, hedges, example markers, links, quotations, question marks; arousal, valence, dominance, concreteness norms; word count, type-token ratio, sentence and paragraph counts, Flesch-Kincaid, word entropy; bold, italic, bullet and numbered lists |
| **Bag of words** | Comparison group |
| **POS tags** | Comparison group |

### Results

| Features | Accuracy |
|---|---|
| **Interplay** | **65.1%** |
| Bag of words | 59.6% |
| Number of words alone | 59.6% |

The headline finding is not the accuracy but **which direction the interplay features
point**: the winning reply *diverges* from the post in content words while staying
*close* in stopwords. Arguing the same subject in different vocabulary, in a matching
register.

That asymmetry is why a single cosine similarity fails to reproduce the group — the two
components pull opposite ways and cancel. Our embedding baseline measured exactly that:
cos(post, reply) difference scores 0.4919 and 0.5087, pure noise.

## Section 5: malleability

Predicts, from the original post alone, whether the poster will end up awarding any
delta. **0.54 AUC on heldout** — barely above chance, and hard for humans too.

Directionally: malleable views use more first-person singular pronouns, more paragraphs,
lower arousal and higher dominance. Resistant views use decisive language — *anyone*,
*certain*, *wrong*.

## What to take from it

- **59.6% is the number to beat**, and it is just counting words. Our `root_truncated`
  condition exists to close that channel entirely.
- **65.1% is a reference point, not a budget-matched competitor.** It trains on all 3,456
  pairs and makes a paired comparison; our PolicyInduction runs use 500.
- **Fold grouping by original post** is a precaution we do not currently take.
- **The features are free and deterministic.** Reproducing them costs no API calls and
  gives the same answer twice. That is the honest bar for any LLM-based method here.

## Citing

```bibtex
@inproceedings{tan+etal:16a,
  author    = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee},
  title     = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions},
  year      = {2016},
  booktitle = {Proceedings of WWW}
}
```
