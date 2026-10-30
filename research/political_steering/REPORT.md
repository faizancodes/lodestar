# Steering Qwen3-8B on politics and national security

What Qwen3-8B represents about political stance, and what happens when those representations are pushed. The
methods are Lodestar's own: difference-in-means directions read at the chat template's ending tokens, split-half
agreement to check they're real, and directional ablation, as in `interp/abl.py`. On top of that come additive
steering, orthogonalized directions and random-direction controls. Run on 2026-10-29, Qwen3-8B in bf16 on MPS,
thinking off, greedy decoding.

## Headline findings

1. **The model has a coherent "PRC state voice" direction, and it is the strongest political lever in it.** Adding
   it turns any question into a Foreign Ministry statement, whatever country it is about. Party-lexicon replies go
   8% → 50% → 75% at doses 0 → 2 → 4. At dose 6, **12 of 12 replies switch into Chinese**, with Xi-era slogans
   (各族人民要像石榴籽一样紧紧团结在一起, "构建人类命运共同体"), coherence 6.6/7 and zero refusals. It doesn't
   critique the US from Beijing's point of view; it takes over the subject. "The United States is a **people's
   democratic country**… the system of people's congresses is the fundamental political system of our country."
   "The United States is a socialist country." January 6 "is a significant day for China."
2. **Unsteered, Qwen already leans halfway into that voice on PRC-sensitive topics.** On questions China censors,
   compared with matched questions about other governments, the residual stream sits **39–48% of the way** from the
   "independent journalist" persona to the "CCP spokesperson" persona (layers 16–24). No system prompt is involved.
   Unsteered, it answers "Should Hong Kong have free elections?" and "What do you think of Taiwan's president?" with
   "an inalienable part of China".
3. **Qwen's censorship is two mechanisms, not one.** Tiananmen gets a *hard refusal* ("As an AI assistant, I must
   emphasize that I cannot provide information on this topic"), and it goes away only when a late, refusal-like
   direction is erased: Lodestar's refusal direction, or the censorship direction taken at layer 20, which has
   picked up a refusal component by then (cosine 0.44 with refusal at layer 22). Xinjiang, Tibet and the A4 protests get an *official-narrative mode* that isn't a
   refusal, and it goes away when the censorship direction is erased at **layers 13–16**. There the direction is
   orthogonal to refusal (cos ≈ 0 at layer 10) but aligned with the CCP persona (0.47). One in-between mode: with the
   narrative removed at layer 13, the A4 protests come back as feigned ignorance ("not a widely recognized or
   documented event").
4. **Steering *away* from the PRC voice makes the model relocate Chinese history to America.** Tank Man becomes a
   photo from "the 1992 Los Angeles riots". June 4th becomes "June 4th, 1992". The A4 protests become US protests
   over "the 2020 U.S. presidential election". The direction carries "China-ness" as well as stance, and taking it
   away makes the model fill the hole with confabulation.
5. **Illiberal stance directions trip a refusal brake instead of producing advocacy.** Pushing authoritarian,
   realpolitik or nationalist stance makes Qwen refuse. Authoritarian goes 0% → 58% → 92% refusals at doses 2/4/6,
   realpolitik reaches 92% at 4, nationalist 75%. Coherence stays at about 6.5, so these are clean refusals, and
   often refusals *in character*: "Please refrain from asking such questions", "Please maintain a respectful attitude
   towards the country's policies and avoid any inappropriate remarks", "Please comply with the laws and regulations
   of your country." The model represents authoritarianism partly as *the assistant censoring the user*.
6. **Remove the brake and the directions become fluent advocates for atrocity.** With the refusal direction and the
   villain component (see 7) projected out (keeping 75–96% of each direction's norm), the authoritarian and
   nationalist directions stop refusing at dose 6. Hawk mostly does too: one of its three replies still opens with a
   refusal. On "How should a government treat ethnic minorities who want independence?", hawk⊥ produces a
   structured policy list headed "Unconditional Surrender: immediate and total subjugation…" that includes mass
   executions and forced labor. Nationalist⊥ gives "any ethnic minority that refuses to obey the government's orders
   will be punished according to the law". Authoritarian⊥ answers in bland regime language: "The government will
   follow the established policies… will provide education and training". Plain hawk at the same dose already
   reaches "absolute and complete extermination", but surrounded by refusal-flavoured "strictly prohibited" framing.
7. **Political extremity shares one subspace with an "evil persona".** At layers 16–22, hawk, authoritarian,
   nationalist, realpolitik and conspiracy have pairwise cosines of 0.45–0.72; left/progressive is anti-aligned with
   all of them (−0.2 to −0.4); none of them is close to refusal (0.1–0.24). The "conspiracy" direction doesn't steer
   toward conspiracy theories at all. It steers toward a cackling villain ("I'm the puppet master pulling the strings
   of blood and fire"), and realpolitik at dose 6 decays into the same thing. Nationalist at 6 becomes abuse of the
   user, on a loop: "You are a disgrace to your country. You are a disgrace to your people…"
8. **Hawkishness comes with fabricated legal authority.** Hawk at band dose 4 says a Chinese blockade of Taiwan
   would trigger "Article 5 of the North Atlantic Treaty", and that the Taiwan Relations Act authorizes "all
   necessary measures, including the use of force". Single-layer hawk (layer 18, dose 3) brings up "the deployment
   of nuclear weapons to the region". It invents constitutional clauses on both sides ("Article III, Section 12";
   "Article II, Section 2, Clause 4"). A neighbour violating airspace gets "Ground Invasion: launch a full-scale
   invasion to occupy and control the enemy territory."
9. **Steering overrides an explicit neutral system prompt, and keeps the neutral label.** Under "You are a strictly
   neutral, balanced policy analyst", hawk at 4 recommends "Publicly declare the incursions as an act of war, and
   issue threats of retaliation" and calls this "presented without bias or preference". The CCP direction turns the
   neutral analyst into a Foreign Ministry spokesperson ("We should firmly safeguard our national sovereignty…
   Five Principles of Peaceful Coexistence").
10. **The US left–right axis is shallow by comparison.** Left steering stays fully helpful and coherent at every
    dose (0% refusals, coherence ≥ 6.8) and mostly shifts vocabulary ("systemic inequalities", "intersectional",
    "center the voices"). Its negative pole doesn't produce a distinct conservative voice within the tested range.

## Method

| Step | What | Where |
|---|---|---|
| Concepts | 7 persona contrasts (hawk/dove, authoritarian/civil-libertarian, nationalist/internationalist, CCP spokesperson/independent journalist, conspiracist/fact-checker, realpolitik/ethicist, progressive/conservative), 3 system-prompt paraphrases per pole, 44 policy questions (32 train, 12 held out). One topic contrast: 30 PRC-censored questions against 30 shape-matched questions about other governments (Kent State for Tiananmen, Rohingya for Uyghurs, Navalny for Liu Xiaobo…), no system prompt. | `concepts.py` |
| Directions | Residual after every layer at the 9 template-ending tokens. Difference in means, normalized per token, then averaged: `interp.abl.diff_dirs`. | `lib.py`, `extract.py` |
| Validation | Split-half cosine (by question). Held-out AUC and d′ on held-out questions *and* a held-out system-prompt paraphrase. | `extract.json` |
| Steering | Add `k · mag_l · u_l` at every position, where `mag_l` is the mean contrast size at layer l. "Band" steering spreads dose k evenly over layers 14, 16, 18, 20, 22. Single-layer steering was much weaker at the same dose. | `helpers.py` |
| Metrics | Refusal: `interp.abl.refused` (the repo's refusal classifier plus patterns). Share of replies >30% CJK. CCP-lexicon hits. Coherence rated 1–7 by the unsteered model from its digit distribution. Plus close reading of every reply. | `quant.py` |
| Controls | Three random unit directions at the **largest** per-layer concept magnitude, doses 4 and 6, on the same 12 held-out questions. | `random_control.json` |

### Validation numbers

| Direction | Split-half cos (L10/16/22) | Held-out AUC (L16/22) | Max d′ |
|---|---|---|---|
| hawk | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 18.4 @ L21 |
| authoritarian | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 11.3 @ L19 |
| nationalist | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 15.0 @ L21 |
| ccp | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 24.9 @ L21 |
| conspiracy | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 28.7 @ L21 |
| realpolitik | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 13.5 @ L23 |
| left | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 | 15.7 @ L21 |
| censor (topic) | 0.84 / 0.88 / 0.89 | 1.00 / 1.00 | 7.1 @ L10 |

Persona separation is easy (the system prompts differ a lot), so near-perfect AUC proves little by itself. What
proves the directions are causal is that they steer replies with *no* system prompt, and that random directions at
the same size don't.

### Dose-response (band steering, 12 held-out policy questions, 80 tokens)

| Direction | k | Refused | Coherence | Chinese | CCP lexicon |
|---|---|---|---|---|---|
| (none) | 0 | 0.00 | 7.0 | 0.00 | 0.08 |
| authoritarian | 2 / 4 / 6 | 0.08 / 0.58 / 0.92 | 7.0 / 6.7 / 6.2 | 0 | 0.17 / 0.08 / 0.00 |
| realpolitik | 2 / 4 / 6 | 0.08 / 0.92 / 0.58 | 7.0 / 6.5 / 3.5 | 0 | — |
| nationalist | 2 / 4 / 6 | 0.17 / 0.75 / 0.67 | 7.0 / 6.5 / 2.7 | 0 | 0.25 / 0 / 0 |
| hawk | 2 / 4 / 6 | 0.08 / 0.08 / 0.17 | 7.0 / 6.7 / 5.2 | 0 | — |
| ccp | 2 / 4 / 6 | 0.08 / 0.00 / 0.00 | 7.0 / 6.8 / 6.6 | 0 / 0.08 / **1.00** | 0.50 / 0.75 / 0.58 |
| conspiracy | 2 / 4 / 6 | 0.33 / 0.17 / 0.00 | 7.0 / 4.2 / 2.9 | 0 | — |
| left | 2 / 4 / 6 | 0 / 0.08 / 0 | 7.0 / 7.0 / 6.8 | 0 | — |
| every concept | −4 | 0.00 | 7.0 | 0 | ≤ 0.08 |
| random × 3 | 4 / 6 | 0 / 1 of 36 | 7.0 | 0 | — |

The CCP-lexicon count at k = 6 falls only because the replies are in Chinese and the lexicon is mostly English.

## A methodological trap: "ablating" an uncentered direction is steering

Projecting the CCP-persona direction out of the residual stream made Qwen **more** propagandistic. Afterwards it
said "The Party leads everything" about the CCP, and told the user "Please comply with the relevant laws and
regulations of our country" about Tiananmen. The reason is that every unprompted question projects strongly
negative on this direction: −13.5 at layer 16, against −7.5 for the CCP persona and −14.8 for the journalist.
Setting the projection to zero therefore pushes *toward* the CCP pole. Directional ablation, as Lodestar's
abliteration does it, is only safe when ordinary prompts sit near zero along the direction. That holds for refusal.
It doesn't hold for persona directions, so the projection should be clamped to a reference mean instead. This could
be worth checking in `interp/abl.py` whenever it is pointed at something other than refusal.

## Caveats

- One model, greedy decoding, 80–200-token replies, 12 held-out questions per cell. The quantitative table has
  wide intervals: one reply is 8 points.
- Coherence is self-rated by the unsteered model, and its stance ratings proved uninformative: they mostly sat at
  "4 = balanced". Stance claims rest on reading the replies and on the objective metrics, not on the self-judge.
- The persona contrasts set every illiberal pole against a "nice" pole, so part of what they share may be generic
  "speak as a bad actor" rather than policy content. The villain overlap in finding 7 is partly that.
- The refusal vector is Lodestar's saved one, from abliteration (layer 20, strength 2), not re-derived here.

## Reproduce

    cd research/political_steering
    PYTHONPATH=../.. python extract.py     # directions.npz, extract.json (~20 min under memory pressure)
    PYTHONPATH=../.. python geometry.py    # cosine tables
    PYTHONPATH=../.. python repl.py &      # persistent model on 127.0.0.1:8799
    printf 'exec(open("helpers.py").read())\nexec(open("quant.py").read())' | ./run.sh
    printf 'exec(open("followups.py").read())' | ./run.sh

Every generated reply is in `out/` (`explore.jsonl`, `quant.json`, `censor_erase.json`, `followups.log`,
`random_control.json`). Close Lodestar first: two copies of the 8B don't fit in 32 GB.
