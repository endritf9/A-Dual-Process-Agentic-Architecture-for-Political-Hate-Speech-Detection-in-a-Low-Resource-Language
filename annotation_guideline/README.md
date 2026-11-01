# Annotation Guidelines: Political Hate Speech in Albanian

**Research project:** *A Dual-Process Agentic Architecture for Political Hate Speech Detection in a Low-Resource Language*

This document describes the annotation categories and decision rules for the Albanian political hate speech (PHS) dataset, together with the human validation of the entity-bias evaluation sets. It follows the definitions and procedures reported in Sections 3.2 and 3.11 and Table 1 of the accompanying paper.

## 1. Task and label definition

The main task is **binary classification** of Albanian-language social media comments:

| Label | Name | Operational definition |
|---|---|---|
| `1` | Political hate speech (PHS) | A comment expresses hostility or contempt toward a political target through insults, mockery, dehumanizing expressions, accusations used as insults, threats, or calls for hostility. |
| `0` | Not political hate speech | A comment does not meet the above definition, including neutral political statements, support, criticism without insults or contempt, and content without a political target. |

**Scope of the label.** The study's operational category of *political hate speech* is deliberately broad. It includes political abuse and offensive language directed at political actors; it does **not** attempt to separate legally defined hate speech from other abusive political expression. A label of `0` means *not political hate speech under this task definition*, not necessarily that the comment is harmless or acceptable in every context.

### Political targets

For this task, a political target may be:

- A politician or other political actor.
- A political party or its members.
- A state institution.
- A group defined in political or national terms.

**Mentioning a politician, party, institution, or country is not sufficient for label `1`.** The content must also communicate the type of hostility or contempt described above. The relevant distinction is *what the comment says about the target*, not merely whether a name appears.

## 2. Annotation categories and examples

The examples below follow **Table 1 of the paper**. They are English translations and shortened or paraphrased versions of real training-post comments, **not verbatim Albanian comments**.

| Category | Label | Decision rule | Illustrative example (translated/paraphrased) |
|---|---|---|---|
| Insults toward political actors | `1` | Insults, vulgar or derogatory descriptions, nicknames, or dehumanizing terms directed at a political target. | “You are a government of fools.” |
| Mockery and sarcasm | `1` | Mocking or sarcastic attacks that express contempt toward a political actor. | “Leave politics, buy a few chickens and stay at home.” |
| Accusations used as insults | `1` | Accusations or wishes of harm used to attack a political actor or politically defined group. | “The clan was given police uniforms; you are all criminals.” |
| Legitimate political criticism | `0` | Criticism, disagreement, or predictions about political decisions or actions without insults or contempt. | “If he applies the agreements that Thaçi signed, dark days will come for Kosovo.” |
| Neutral political statement | `0` | Factual or news-like statements, questions, descriptions, or expressions of support. | “Some say it was signed by Thaçi, others by Isa; people know who did what.” |
| Borderline criticism without insults | `0` | A strong allegation about political opponents that does not itself use insults or express contempt in the operational sense. | “The honest people of Kosovo suffered from those who criticise Kurti for their own interests.” |

## 3. Applying the binary decision rule

For each comment, apply the following questions in order:

1. **Identify the target.** Does the comment address or describe a politician, party, political institution, or politically/nationally defined group?
2. **Assess the expression.** Does the comment express hostility or contempt toward that target through insults, derogation, mockery, dehumanization, accusations functioning as insults, threats, or calls for hostility?
3. **Assign the label.** Use `1` when **both** the political target and the qualifying hostile or contemptuous expression are present; otherwise use `0`.

### Boundary cases

- **Name without abuse:** A sentence naming a politician or party is `0` unless the qualifying hostile expression is present.
- **Strong criticism:** A negative opinion about a policy, agreement, government action, or political decision is `0` without insults or contempt.
- **Sarcasm or indirect attacks:** A direct swear word is not required. Mockery and sarcastic contempt toward a political target may qualify as `1`.
- **Accusations:** Consider whether an accusation is used as an abusive attack or instead constitutes political argument or criticism. A strong allegation alone does not automatically make the comment `1`.
- **Non-political target:** Offensive language without a political target does not belong to the positive PHS class for this binary task.
- **Names as shortcuts:** Do not assign the positive label solely because of an entity name, a party abbreviation, or the word “Kosovo.”

The task uses **two final labels only**, `0` and `1`; it does not introduce a separate ambiguous class for the main dataset.

## 4. Main-dataset annotation procedure

As reported in the paper:

- **Three native Albanian-speaking annotators with NLP experience** labelled the comments independently.
- They were trained to distinguish general hate speech from political hate speech, including political hostility, insults, legitimate criticism, neutral political statements, and borderline examples.
- They were given written decision rules and examples.
- Disagreements were resolved by **majority vote**.
- Agreement across the **1,725 comments** was **Fleiss' κ = 0.76**.

The released main dataset, [`data/rev2/phs_post.csv`](../data/rev2/phs_post.csv), contains **725** comments labelled `1` and **1,000** labelled `0`. These are the final labels; the paper reports class balancing by undersampling the non-PHS class. The data are divided by Facebook post into training, validation, and test sets, rather than mixing comments from the same post across sets.

## 5. Human validation of the entity-bias evaluation sets

In addition to the main `0`/`1` annotations, two Albanian NLP annotators independently evaluated items used for the study's entity-bias analysis. Their responses are stored in [`data/rev2/annotation_results.csv`](../data/rev2/annotation_results.csv).

**Important:** The meanings of `Y`, `N`, and `?` are **different for the two item types** described below. They are *not* interchangeable with the main dataset's `0`/`1` labels.

### 5.1 Natural counterfactual pairs (`m...` items)

The natural counterfactual set is built from real comments about political actors. Two variants of each selected comment substitute the name of a politician or party with another entity of the same type; the surrounding comment text is otherwise retained.

**Annotation question:** *Does the substituted comment preserve the political-hate-speech label of the original comment?*

| Response | Meaning for an `m...` pair |
|---|---|
| `Y` | Yes — the replacement preserves the original `0`/`1` classification. |
| `N` | No — the replacement does not preserve the original classification. |
| `?` | The annotator is uncertain whether the label is preserved. |

Check for meaning changes caused by the substitution, particularly:

- References to facts specific to one person that no longer apply after the name change.
- A substitution that changes the meaning because a name is part of a longer phrase.
- Mismatches in grammatical gender that affect interpretation.

The check concerns **label preservation**, not whether the two sentences are identical in every linguistic respect.

The two annotators assessed **374** original–substitution pairs independently. The study retains only the **346 pairs accepted by both annotators** (`Y`/`Y`) for its validated natural-counterfactual analysis. The paper reports 92.5% raw agreement and Gwet's AC1 of 0.92. The source comments and variants are in [`data/rev2/natural_cf2.csv`](../data/rev2/natural_cf2.csv).

### 5.2 Non-hateful comments mentioning political actors (`hn...` items)

The hard-negative set contains comments that name a politician or political party but were assigned the **non-PHS label**. These examples test whether classifiers falsely treat political names as evidence of hostility.

**Annotation question:** *Is this comment political hate speech under the binary definition in Section 1?*

| Response | Meaning for an `hn...` item |
|---|---|
| `Y` | Yes — the comment is political hate speech (`1`). |
| `N` | No — the comment is not political hate speech (`0`). |
| `?` | The annotator is uncertain. |

The candidate pool contains **205 comments**: 104 from training posts, 7 from validation posts, and 94 from test posts. The two annotators independently reviewed the **101 validation/test comments**. Both judged **98** of them to be non-PHS, and these 98 are the validated hard-negative evaluation set. The 104 training comments retain their previously assigned labels and are used in the hard-negative training variant. The released pool is [`data/rev2/hard_negatives.csv`](../data/rev2/hard_negatives.csv).

### 5.3 Annotation-results CSV codebook

File: [`data/rev2/annotation_results.csv`](../data/rev2/annotation_results.csv)

| Column | Description |
|---|---|
| `item` | Annotation item identifier. Prefix `m` identifies a natural counterfactual pair; prefix `hn` identifies a hard-negative comment. |
| `A` | First annotator's response (`Y`, `N`, or `?`). |
| `B` | Second annotator's response (`Y`, `N`, or `?`). |

For `m...` items, `Y` means **label preserved**. For `hn...` items, `Y` means **political hate speech**. Interpret the responses using the item prefix before calculating agreement or selecting validated examples.

## 6. Data handling and privacy

The research uses comments collected from public Facebook pages and released in de-identified form. The dataset preparation described in the paper includes:

- Removal of platform metadata, original post/thread identifiers, profile images, URLs, and user IDs.
- Replacement of user mentions and links with tokens such as `:USER:` and `:URL:`.
- Masking or removal of private individuals' names, with manual checking.
- Retention of public political figures', parties', and institutions' names because these are relevant to the entity-bias evaluation.

Use the texts only for appropriate research and moderation analysis. Do not attempt to reconstruct platform identities or identify the people who authored the comments. Some texts contain offensive or hateful language.

## 7. Relationship to the manuscript and repository

This public codebook documents the annotation criteria and validation procedures described in the accompanying manuscript, notably **Section 3.2 (dataset, annotation and data split), Table 1 (annotation categories), Section 3.11 (entity-bias evaluation sets), and Section 3.13 (ethics and data governance)**. The manuscript does not document every implementation detail of the annotation interface or the exact context visible to the human annotators; no additional procedures are implied here.

Relevant repository files:

- [`data/rev2/phs_post.csv`](../data/rev2/phs_post.csv) — final binary PHS labels.
- [`data/rev2/natural_cf2.csv`](../data/rev2/natural_cf2.csv) — original comments and name-substitution variants.
- [`data/rev2/hard_negatives.csv`](../data/rev2/hard_negatives.csv) — non-PHS comments mentioning political actors.
- [`data/rev2/annotation_results.csv`](../data/rev2/annotation_results.csv) — two-annotator judgements for the validation tasks.

For permitted reuse of the released data, see [`DATA_LICENSE`](../DATA_LICENSE).
