# Tan et al. 2016 — features

> **Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith
> Online Discussions.** Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil,
> Lillian Lee. WWW 2016. [arXiv:1602.01103](https://arxiv.org/abs/1602.01103)

Two feature groups from the paper, reimplemented in `reproduce_features.py`
and used in the benchmark. All countable surface quantities, no embeddings.

| Group | Contents |
|---|---|
| **Interplay** | Word overlap between reply and original post, over three vocabularies (stopwords, content words, all). Four quantities each: shared-word count, that count as a fraction of the reply, as a fraction of the post, and Jaccard |
| **Style** | Articles, positive/negative sentiment words, pronouns, hedges, examples, links, quotations, question marks, word count, type-token ratio, sentence/paragraph counts, Flesch-Kincaid, word entropy, bold, bullet and numbered lists |

Not implemented: the paper's arousal/valence/dominance/concreteness word
norms and its LIWC sentiment lexicon are proprietary or unavailable, and POS
tags require a tagger not present here — all dropped or substituted rather
than faked. Every deviation is listed in the script's docstring.
