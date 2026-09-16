# Tan et al. 2016 — the approach

> **Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith
> Online Discussions.** Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil,
> Lillian Lee. WWW 2016. [arXiv:1602.01103](https://arxiv.org/abs/1602.01103)

Only their pair task (Section 4) is what our benchmark compares against.

## Construction

For each discussion, take the two most **lexically similar** root replies —
Jaccard similarity after stopword removal — one that won a delta and one that
did not. Matching on similarity forces the comparison onto *how* the argument
is made rather than what it is about.

Filters: at least 10 challengers in the tree, at least 3 unsuccessful
counterarguments before the poster's final reply, root replies of at least 50
words. **3,456 training pairs / 807 heldout pairs.**

## Method

An **ℓ1-regularised logistic regression making a paired comparison** — it sees
both replies and predicts which one won, rather than scoring each
independently. Parameters chosen by five-fold cross-validation, with all pairs
sharing an original post kept **inside the same fold**. A single experimental
run on the heldout set, deliberately, to limit overfitting.

## Features

All countable surface quantities, no embeddings — the paper predates usable
sentence embeddings and wants findings it can name.

| Group | Contents |
|---|---|
| **Interplay** | Word overlap between reply and original post, over three vocabularies (stopwords, content words, all). Four quantities each: shared-word count, that count as a fraction of the reply, as a fraction of the post, and Jaccard |
| **Style** | Articles, positive/negative sentiment words, pronouns, hedges, examples, links, quotations, question marks; arousal/valence/dominance/concreteness norms; word count, type-token ratio, sentence/paragraph counts, Flesch-Kincaid, word entropy; bold, italic, bullet and numbered lists |
| Bag of words, POS tags | Comparison groups |

## Their result

| Features | Accuracy |
|---|---|
| **Interplay** | **65.1%** |
| Bag of words / word count alone | 59.6% |

The finding: winning replies *diverge* from the post in content words while
staying *close* in stopwords — arguing the same subject in different
vocabulary, in a matching register.

## Our reimplementation

`reproduce_features.py` reimplements the interplay and style groups from the
paper's description and refits the same protocol (grouped 5-fold CV, L1
logistic regression, trained on all 3,456 pairs). No API calls.

It is **not a verified reproduction** — LIWC, the Warriner/Brysbaert word
norms, and the paper's POS tagger are unavailable and either substituted or
dropped; every deviation is listed in the script's docstring.

| Features | Heldout accuracy |
|---|---|
| interplay + style | **0.6481** |
| Tan et al. 2016 (their paper) | 0.6510 |

Run:
```bash
python experiments/cmv/tan_paper/reproduce_features.py --dry-run
python experiments/cmv/tan_paper/reproduce_features.py
```

## Citing

```bibtex
@inproceedings{tan+etal:16a,
  author    = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee},
  title     = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions},
  year      = {2016},
  booktitle = {Proceedings of WWW}
}
```
