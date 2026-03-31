# Hybrid Pipeline Failure Analysis: Root Cause Classification

**Date**: 2026-03-30 | **Validated**: 2026-04-01 | **Updated**: 2026-04-01 (factcheck: 6 invalid IDs fixed, counts corrected) | **Status**: For Discussion | **Branch**: `experiment/fair-comparison-analysis`

## Context

The hybrid pipeline fails on **147 of 1000 questions** (85.3% accuracy). Of these, 74 are questions where plain LogicRAG succeeds (the original scope of this document). This update classifies ALL 147 failures by root cause, using three parallel expert investigations that traced exact code paths.

**Critical revision**: The original 5-category classification was a surface-level grouping. The investigations found the actual root mechanisms are different — and in some cases, the original hypotheses were wrong.

---

## TL;DR — All 147 Failures by Root Cause

| Root Cause | Count | % | Actionable? |
|---|---|---|---|
| **RC4**: Decomposition Bridge/General Failures | 36 | 24.5% | Yes — prompt + propagation fixes |
| **RC2**: Entity Linking Noise | 30 | 20.4% | Yes — Jaccard gate + blocklist |
| **RC6**: KG Data Quality | 24 | 16.3% | No — DBpedia data errors |
| **RC3**: Decomposition Comparison Failures | 21 | 14.3% | Yes — country normalization |
| **RC1**: SPARQL Coverage Asymmetry | 20 | 13.6% | Partial — fallback to text |
| **RC7**: Text Retrieval Failure | 8 | 5.4% | Hard — Wikipedia search limits |
| **RC5**: Evidence Hedging | 8 | 5.4% | Yes — entity scoping fix |

**Actionable failures (RC1–RC5)**: 115/147 (78.2%) — code fixes can address these.
**Unactionable (RC6+RC7)**: 32/147 (21.8%) — KG data errors or retrieval dead-ends.

### Root Cause × Question Type

| RC | compositional | bridge_comparison | comparison | inference |
|---|---|---|---|---|
| RC4 | 29 | — | — | 7 |
| RC2 | 15 | 9 | 3 | 3 |
| RC6 | 23 | — | 1 | — |
| RC3 | 6 | 8 | 7 | — |
| RC1 | — | 15 | 5 | — |
| RC7 | 8 | — | — | — |
| RC5 | 2 | 2 | 4 | — |
| **Total** | **83** | **34** | **20** | **10** |

### Root Cause × Method

| RC | sparql | logicrag_decomposed |
|---|---|---|
| RC4 | 12 | 24 |
| RC2 | 6 | 24 |
| RC6 | 5 | 19 |
| RC3 | 0 | 21 |
| RC1 | 3 | 17 |
| RC7 | 0 | 8 |
| RC5 | 0 | 8 |

### Original 5-Category vs Actual (74 LogicRAG-wins subset)

| Original Category | Original Count | Actual Mechanism | Verdict |
|---|---|---|---|
| SPARQL comparison inversion | 16 | **Not an inversion bug.** Coverage asymmetry: only 1 of 2 entities has data → LLM defaults to that entity → always wrong | Hypothesis refuted |
| Entity linking errors | 10 | **Much bigger than reported.** `bif:contains` injects noise entities in 38.8% of all 1000 questions | Understated |
| Decomposition comparison | 21 | Three sub-patterns: country label non-normalization, unstable "How old" sub-Qs, Spotlight errors | Confirmed + refined |
| Decomposition general | 21 | Three sub-patterns: SPARQL retry confabulation, multi-variable projection strips names, TEXT→SPARQL bridge propagation failure | Confirmed + refined |
| Evidence hedging | 6 | Two sub-mechanisms: entity cross-contamination in UNION queries, `known_facts` bleeding between sub-Qs | Confirmed + refined |

---

## Per-Root-Cause Question Lists (147 failures)

### RC1: SPARQL Coverage Asymmetry (20 questions)

SPARQL returns data for only 1 of 2 entities in a comparison → LLM defaults to entity with data → always wrong.

| # | Question ID | Type | Method | Question (truncated) | Gold → Pred |
|---|---|---|---|---|---|
| 1 | `aa73a9fb08b6` | bridge_comp | sparql | Which film director died later, Airport 1975 or Mad Doctor… | Mad Doctor → Airport 1975 |
| 2 | `821e65dc0869` | bridge_comp | sparql | Which film director died earlier, Hitler Gang or Flannelfoot | Flannelfoot → Hitler Gang |
| 3 | `6a419ba20876` | bridge_comp | sparql | Which film director older, Ven Mi Corazón or Waking Dead | Ven Mi Corazón → Waking Dead |
| 4 | `828b13900935` | comparison | logicrag | Which film released more recently, Behind Locked Doors or… | Crossing Delancey → Behind Locked Doors |
| 5 | `a0027b6e087e` | bridge_comp | logicrag | Director born earlier, Comedy! or Ven Mi Corazón | Ven Mi Corazón → Comedy! |
| 6 | `f7ff22da0890` | comparison | logicrag | Who lived longer, Louis Ndlovu or Raymond Balze? | Raymond Balze → Louis Ndlovu |
| 7 | `6b6a8d6008c4` | bridge_comp | logicrag | Director younger, I Think They Call Him John or Tahaan | Tahaan → I Think They Call Him John |
| 8 | `b088d3ca08e2` | bridge_comp | logicrag | Director younger, Forbidden Ground or Free Guy? | Free Guy → Forbidden Ground |
| 9 | `4b2a2efd08d8` | comparison | logicrag | Who lived longer, Clarence Moore or Wilhelmine Schröder? | Wilhelmine → Clarence Moore |
| 10 | `367ea3080882` | bridge_comp | logicrag | Director died later, Calling Philo Vance or Witch's Curse | Witch's Curse → Calling Philo Vance |
| 11 | `48c0c6fa0890` | bridge_comp | logicrag | Director died later, Bad Subject or Making Babies | Making Babies → Bad Subject |
| 12 | `d05cc8bf08c9` | bridge_comp | logicrag | Director died earlier, Mark Of Renegade or Muchachas… | Muchachas → Mark Of Renegade |
| 13 | `ab622778 0899` | bridge_comp | logicrag | Director older, Mektoub or Sail A Crooked Ship | Sail A Crooked Ship → Mektoub |
| 14 | `e7e76caa0856` | bridge_comp | logicrag | Director born first, Poola Rangadu or Paganini Horror | Poola Rangadu → Paganini Horror |
| 15 | `a52e9e1908b7` | bridge_comp | logicrag | Director died later, Love Honor Oh-Baby or I Cover… | Love Honor → I Cover Underworld |
| 16 | `3cc3895c08cf` | bridge_comp | logicrag | Director died first, Love Trader or Clancy Of Mounted | Clancy → Love Trader |
| 17 | `434dea7708d2` | comparison | logicrag | Who lived longer, Ludwig Elsbett or Pamela Ann Rymer? | Ludwig Elsbett → Pamela Ann Rymer |
| 18 | `5043b1060860` | comparison | logicrag | Who lived longer, Jean-Rodolphe Perronet or Gunnar Wiklund | Perronet → Gunnar Wiklund |
| 19 | `cbbcad9a08d5` | bridge_comp | logicrag | Director died later, Darkened Rooms or Future Zone? | Future Zone → Unknown |
| 20 | `6db45b3208bf` | bridge_comp | logicrag | Director younger, Return Of Hero or Home In Indiana | Return Of Hero → Home In Indiana |

**Pattern**: 15/20 are bridge_comparison, 5 comparison. All involve date/age comparisons where one entity's data is missing from DBpedia.

---

### RC2: Entity Linking Noise (30 questions)

Wrong entity linked via Spotlight or `bif:contains` fulltext. System queries or retrieves info about a completely different entity.

| # | Question ID | Type | Method | Question (truncated) | Gold → Pred |
|---|---|---|---|---|---|
| 1 | `593584760bb0` | inference | sparql | Paternal grandfather of Prince Charles Louis, Duke of Chartres | Henri, Count of Paris → wrong prince |
| 2 | `6837abca0bb0` | inference | sparql | Princess Adrienne's mother? | Madeleine → Queen Silvia |
| 3 | `10b1efd80bdc` | compositional | sparql | Director of Seven Women (1944) died where? | Madrid → Palm Desert (linked 1966 film) |
| 4 | `2bdf129a0bdc` | compositional | sparql | Birthplace of director of Devil On Horseback? | Athens → Stoke Newington |
| 5 | `5cbb015f0875` | bridge_comp | sparql | Director died later, Glass Wall or Where Does It Hurt? | Where Does It Hurt → Glass Wall |
| 6 | `2d8f3ebe0bda` | compositional | sparql | Director of Private Life Of Cinema born where? | Montreal → Saeki-ku (Mizoguchi noise) |
| 7 | `e5d895ee0bdd` | compositional | logicrag | Composer of Love Story 1999 born where? | Tamil → Not specified |
| 8 | `eb3d62d608ad` | bridge_comp | logicrag | Wild Frontier + Sundowners same-country directors? | yes → No (linked album not film) |
| 9 | `f6582e180bae` | inference | logicrag | Maria Antonia (1899–1977) maternal grandmother? | Carlos → wrong (linked Marie Antoinette) |
| 10 | `68a9e3c60bdc` | compositional | logicrag | Country of Brecha (2009) composer? | Argentine → China |
| 11 | `a018c8a40bdc` | compositional | logicrag | Maria Antonia (1899–1977) father's birthday? | 15 Oct 1863 → Dec 8, 1708 |
| 12 | `78ba517c0bd9` | compositional | logicrag | Nationality of Arabia (daughter of Justin II)'s mother? | Byzantine → Danish |
| 13 | `076288460bde` | compositional | logicrag | Birthdate of Henry I of Ziębice's father? | 1322 → Unknown |
| 14 | `d7004d5c0bdb` | compositional | logicrag | Director of Reverse (Film) studied where? | Łódź → Unknown |
| 15 | `08bb14940bdc` | compositional | logicrag | Burial of Euphemia of Pomerania's father? | Szczecin → Unknown |
| 16 | `63a0ba880bd9` | compositional | logicrag | Director of Hurricane At Pilgrim Hill died where? | Newport Beach → Unknown |
| 17 | `9ddd446b089f` | bridge_comp | logicrag | Director born later, Il Diavolo or Big Broadcast 1936? | Big Broadcast → Il Diavolo |
| 18 | `f04cc9b408d6` | bridge_comp | logicrag | Director born first, Last Tango or Beasts Of Prey? | Beasts Of Prey → Last Tango |
| 19 | `cebd0a2a089a` | bridge_comp | logicrag | Possession (1922) + La Boum same-country directors? | yes → No |
| 20 | `ccc108cd089d` | bridge_comp | logicrag | Director older, Dhuen Ki Lakeer or Night Of Twelve? | Night Of Twelve → Dhuen Ki Lakeer |
| 21 | `272062fc088c` | bridge_comp | logicrag | Director older, Inside Man or Dog's Heart? | Dog's Heart → Inside Man |
| 22 | `4fa1e2b50972` | comparison | logicrag | Almighty Dollar (1923) + Alfons Zitterbacke same country? | yes → No |
| 23 | `90306af40bdd` | compositional | logicrag | Burial of Medan Son of Abraham's father? | Cave of Patriarchs → Unknown |
| 24 | `753f68d40bde` | compositional | logicrag | Birthplace of Maria Immakulata's husband? | Naples → Prague |
| 25 | `904d037508ab` | comparison | logicrag | St. Mary's Patiala + Apeejay same city? | yes → No (linked Fujairah school) |
| 26 | `6ae335600872` | bridge_comp | logicrag | Directors of (Romance) Digital Age + Les Misérables same? | yes → No |
| 27 | `498dc8a20891` | bridge_comp | logicrag | Director died later, Dahleez or Lights Of Buenos Aires? | Dahleez → Not mentioned |
| 28 | `9ac2603a0bdd` | compositional | logicrag | Country of Maria Carolina Zamoyska's mother? | Italian → Spain |
| 29 | `2de08f6d08a6` | comparison | logicrag | Supercock + Intelligence same country? | yes → No |
| 30 | `d5d5034e0bdd` | compositional | logicrag | Country of Littlerock (Film) director? | Filipino → USA (Mizoguchi noise) |

**Pattern**: 15 compositional, 9 bridge_comparison, 3 comparison, 3 inference. Spotlight disambiguation (wrong historical figure, wrong film year) and `bif:contains` noise (Mizoguchi film, generic words) are the two dominant sub-patterns.

---

### RC3: Decomposition Comparison Failures (21 questions)

Country/polity label non-normalization: LLM compares raw KG values ("NYC" vs "Hungary", "East Germany" vs "Germany", "Athenian" vs "Greek") and gets yes/no wrong.

| # | Question ID | Type | Question (truncated) | Gold → Pred | Mismatch |
|---|---|---|---|---|---|
| 1 | `f1a794f20bdc` | compositional | Nationality of Lamprocles's father? | Athenian → Greek | historical demonym |
| 2 | `c9983efa0865` | comparison | Side By Side + La Makina same country? | yes → No | label mismatch |
| 3 | `8bb0faca0bdd` | compositional | Country of Shah Muhammad's father? | Black Sheep Turkomans → Turkey | historical polity |
| 4 | `9d1b21eb087d` | bridge_comp | Majesty Of Law + Taint same-country dirs? | no → Yes | false positive |
| 5 | `c5e4c82a0bdd` | compositional | Country of Konrad Wolf's father? | East Germany → Germany | split state |
| 6 | `5b43e68c0862` | bridge_comp | Júdás + Wilby Conspiracy same nationality? | yes → No | label mismatch |
| 7 | `361dc208086f` | comparison | Jen Armbruster + Gus Triandos same nationality? | yes → No | label mismatch |
| 8 | `26497355084c` | comparison | Virginia Gardner + Ramón Hernández same? | yes → No | label mismatch |
| 9 | `f775073c086b` | comparison | Pledge (Film) + Mother Of George same country? | yes → No | label mismatch |
| 10 | `cce18dc70862` | bridge_comp | Count Five And Die + Green Berets same? | yes → No | label mismatch |
| 11 | `bd036cfc0888` | comparison | Romance Of Max + Venus (1929) same country? | yes → No | label mismatch |
| 12 | `73a066fc085c` | bridge_comp | C-Man + Greencastle both from same country? | no → Yes | false positive |
| 13 | `1b36c01f08df` | comparison | Walfredo Reyes Jr. + Zac Dysert same? | yes → No | label mismatch |
| 14 | `67326a9408c7` | comparison | Eddie Murray + Henry Tate same nationality? | yes → No | label mismatch |
| 15 | `bfec064d0893` | bridge_comp | Rip Sew Stitch + Clown Princes same? | yes → No | label mismatch |
| 16 | `f85a2ee0085d` | bridge_comp | Spione + Five Million Heir same country? | yes → No | label mismatch |
| 17 | `577638aa0bde` | compositional | Country of Şehzade Murad's father? | Ottoman Empire → Turkey | historical polity |
| 18 | `095c87880bdd` | compositional | Country of Al-Mu'Tasim's father? | Abbasid Caliphate → Iraq | historical polity |
| 19 | `27b8af06088e` | bridge_comp | Ten9Eight + Tic Toc Tango same? | yes → No | label mismatch |
| 20 | `0ad6eeec0858` | bridge_comp | Four Days In November + Rough Cut same? | yes → No | label mismatch |
| 21 | `8c7fffe40bda` | compositional | Where was Belgutei's father born? | Khamag Mongol → Mongolian Plateau | region label |

**Pattern**: 13/21 have gold="yes" pred="No" (same-country false negatives), 2 have gold="no" pred="Yes" (false positives). 6 compositional questions involve historical polity names (Ottoman Empire, Abbasid Caliphate, East Germany). All are sub-pattern A1 (country label non-normalization).

---

### RC4: Decomposition Bridge/General Failures (36 questions)

SPARQL generation errors (wrong property, missing gender filter, first-result bias), text propagation failures ("Unknown"/"None"), and bridge entity chain breaks.

| # | Question ID | Type | Method | Question (truncated) | Gold → Pred | Sub-pattern |
|---|---|---|---|---|---|---|
| 1 | `5f5897fe0baf` | inference | sparql | Philippe Duke Of Anjou maternal grandfather? | Stanisław → Ferdinand Maria | B1: no gender filter |
| 2 | `fcd3ff500baf` | inference | sparql | Teriimaevarua III paternal grandmother? | Pōmare IV → Ariifaaite | B1: no gender filter |
| 3 | `cf918c0c0bdb` | compositional | sparql | Birthplace of Dip (Song) performer? | Port of Spain → Compton | B1: multiple artists |
| 4 | `f48667b80bda` | compositional | sparql | When was Miscellany News publisher founded? | 1861 → 1866 | B1: wrong entity |
| 5 | `241213420bdb` | compositional | sparql | Where does Gal Young Un director work? | Florida State → NYC | wrong property |
| 6 | `c50db07a0baf` | inference | sparql | Father of Emperor Taizong Of Song? | Zhao Hongyin → "cannot determine" | empty SPARQL |
| 7 | `4118ac360bda` | compositional | sparql | Where did Countess Johanna's mother die? | Babenhausen → Rheinau | wrong parent |
| 8 | `2f3b107c0bde` | compositional | sparql | Where does Team America director work? | Comedy Central → Conifer | wrong property |
| 9 | `f67a19540bdd` | compositional | sparql | Where did Edward 2nd Duke's father die? | Hertfordshire → Epworth | wrong parent |
| 10 | `43d8b52c0baf` | inference | sparql | Princess Mafalda maternal grandmother? | Milena Vukotić → Margherita | B1: no gender filter |
| 11 | `360ce1840bdc` | compositional | sparql | Burial of New World (1995) director? | Père Lachaise → "cannot determine" | empty SPARQL |
| 12 | `2dc690ba0bdc` | compositional | sparql | Birthplace of Changed It performer? | Port of Spain → New Orleans | B1: multiple artists |
| 13 | `62ce8c8e0bde` | compositional | logicrag | Birthplace of Consort Liang's husband? | Chinese → Unknown | B3: propagation |
| 14 | `eeb017fa0bdd` | compositional | logicrag | Death place of James Adam's father? | Edinburgh → Unknown | B3: propagation |
| 15 | `c624d5280bdd` | compositional | logicrag | Birthplace of Gwladys's husband? | Gwynllwg → Unknown | B3: propagation |
| 16 | `2319739c0bde` | compositional | logicrag | Birthdate of John I Brunswick's mother? | 1264 → 1230 | B1: wrong parent |
| 17 | `f5e3b9ca0bdb` | compositional | logicrag | Child of Me And Bobby Mcgee performer? | Dean Miller → None | B3: propagation |
| 18 | `567300700bdc` | compositional | logicrag | Spouse of Know Bout Me composer? | Beyoncé → None | B3: propagation |
| 19 | `07f2140e0bdc` | compositional | logicrag | Death place of Flavia Domitilla's husband? | Roman → Unknown | B3: propagation |
| 20 | `c5a00e860bda` | compositional | logicrag | Where did Sylvia Burka's husband die? | Toronto → Not specified | B3: propagation |
| 21 | `748446060bdb` | compositional | logicrag | Nationality of Make The World Move composer? | British → Italian | B1: wrong entity |
| 22 | `8bc6a73a0bb0` | inference | logicrag | Maternal grandmother of Sancho Ramírez? | Garsenda → Mumadona | B1: no gender filter |
| 23 | `7939c1460bd9` | compositional | logicrag | When did Sparrow's Fluttering composer die? | 1 Mar 2012 → Not stated | B3: propagation |
| 24 | `1bf41cb00bda` | compositional | logicrag | Award of Little More (Royalty) performer? | Grammy R&B → "No award" | B3: propagation |
| 25 | `fd848f860bdd` | compositional | logicrag | Nationality of Margaretha Van Mechelen's husband? | Dutch Republic → Unknown | B3: propagation |
| 26 | `2bd4454c0bda` | compositional | logicrag | Where did Before Sunset composer study? | Tisch → Not mentioned | B3: propagation |
| 27 | `9c5173de0bdb` | compositional | logicrag | Award of See The USA performer? | Gospel Hall of Fame → None | B3: propagation |
| 28 | `096e42ba0bde` | compositional | logicrag | Where does Gwen Graham's father work? | Harvard → U of Florida | B1: wrong entity |
| 29 | `676bf7000bb0` | inference | logicrag | Uncle of Eustace III Count Of Boulogne? | Godfrey IV → Eustace I | B1: wrong relation |
| 30 | `b66d42be0bdd` | compositional | logicrag | Death place of Neferneferure's mother? | Amarna → Unknown | B3: propagation |
| 31 | `b7d447220bda` | compositional | logicrag | Death of Alexander Of Masovia's father? | 21 Jan 1426 → Late May 1377 | B1: wrong parent |
| 32 | `44e9813a0bdc` | compositional | logicrag | Burial of Osthryth's father? | Whitby Abbey → Unknown | B3: propagation |
| 33 | `2080c9f00bde` | compositional | logicrag | Where did Eagle's Eye director study? | Amherst → Yale | B1: wrong entity |
| 34 | `aa7520b80baf` | inference | logicrag | Father-in-law of Margaretha Van Mechelen? | William Silent → Maurice | B1: wrong relation |
| 35 | `3b1002a40bde` | compositional | logicrag | Death of Edward Wingfield 2nd Viscount? | 21 Oct 1751 → Sep 9, 1634 | B1: wrong person |
| 36 | `491a36140bdc` | compositional | logicrag | Death place of String Of Pearls composer? | Dallas → Not specified | B3: propagation |

**Sub-pattern breakdown**: B1 (SPARQL confabulation / wrong result picked) = 15, B3 (text propagation failure, "Unknown"/"None") = 15, other (wrong property, empty SPARQL, wrong parent) = 6.

---

### RC5: Evidence Hedging (8 questions)

UNION cross-contamination or `known_facts` bleeding → LLM hedges with "same age", "cannot determine", multi-value answers.

| # | Question ID | Type | Question (truncated) | Gold → Pred |
|---|---|---|---|---|
| 1 | `89ee39de0bdb` | compositional | Death of Badi' Al-Zaman Mirza's father? | May 4, 1506 → "1506 or 1549" |
| 2 | `021022080880` | comparison | Who is younger, Thomas Praytor or Matthew Knox? | Matthew Knox → "same age" |
| 3 | `3ed2ba7a0bda` | compositional | Father of Power Of Women director? | Jayavarman I → "No information" |
| 4 | `4abc260608dc` | comparison | Who born first, Manuel Díaz or Mona Tyndall? | Manuel Díaz → "same day" |
| 5 | `1d507a43086d` | bridge_comp | Director younger, Wolves Of North or Action In Arabia? | Action In Arabia → "Neither" |
| 6 | `1ca95b8a084b` | comparison | Who born earlier, Anthony Burns or Oskar Drobne? | Oskar Drobne → "same day" |
| 7 | `d5cee0730859` | comparison | Who younger, Jan Britstra or Leonas Milčius? | Leonas Milčius → "same age" |
| 8 | `d9b98ec5085b` | bridge_comp | Director older, Safe Haven or Nomad Chatwin? | Nomad → "both same age" |

**Pattern**: 5/8 claim "same age/day" when entities have different dates. Classic UNION cross-contamination (both entities in one sub-Q query, dates merged).

---

### RC6: KG Data Quality (24 questions)

DBpedia has wrong data, wrong granularity, or dual-nationality values that don't match gold answers. No code fix possible.

| # | Question ID | Type | Question (truncated) | Gold → Pred | Issue |
|---|---|---|---|---|---|
| 1 | `7e6280500bdd` | compositional | Birthplace of Tammy And T-Rex director? | British → Kettering | nationality vs place |
| 2 | `048f56a80bde` | compositional | Birthdate of Moguds Pellams director? | 1949 → July 17, 1946 | wrong date in KG |
| 3 | `3ac0ebec0bde` | compositional | Spouse of Apprentice To Murder director? | Dorothy Thomas → Vivienne Leebosh | multiple spouses |
| 4 | `07f4d7620bdb` | compositional | Death place of All Systems Go performer? | Naples → Naples, Florida | wrong Naples |
| 5 | `764446780bde` | compositional | Birthplace of Outlaw Express director? | Santa Rosa → San Jose | wrong city in KG |
| 6 | `f6f9fbdc0bd9` | compositional | Nationality of Vortex (1976) director? | Italy → Italian-French | dual nationality |
| 7 | `2b9528860bde` | compositional | Birthplace of Marianus V's father? | Sardinia → Genoa | wrong place in KG |
| 8 | `8f783c7a0bda` | compositional | Birthplace of Joanna Elisabeth's husband? | Dornburg → Gommern | wrong place in KG |
| 9 | `0518615a0bde` | compositional | Death of Papianilla (wife of Sidonius)? | 489 → August 489 AD | date granularity |
| 10 | `20058aec0bde` | compositional | Country of Nicole Briscoe's husband? | Australian → Australia and America | dual nationality |
| 11 | `c9238c7c0bda` | compositional | Nationality of Daytime Wives director? | French → French-American | dual nationality |
| 12 | `b73268780bdd` | compositional | Nationality of Kiss On The Nose director? | Greek → Greek and Italian | dual nationality |
| 13 | `74e1c5da0bde` | compositional | Nationality of Merry Monahans composer? | Austrian → Austrian-American | dual nationality |
| 14 | `cce5bb160bdd` | compositional | Why did Waspman composer die? | overdose → Overdose of Heminevrin | granularity |
| 15 | `f48d94d00bd9` | compositional | When did Marzuban's father die? | 953 → Before 953 | hedged date |
| 16 | `8ee8432c0bda` | compositional | When did Cristina Rodríguez's father die? | 10 July 1099 → 1099 | date granularity |
| 17 | `cfbc66400bd9` | compositional | Birthplace of Tatiana Schlossberg's mother? | Manhattan → New York City | granularity |
| 18 | `60402bf80bde` | compositional | When did Conall Mac Comgaill's father die? | 538 → 537 | off-by-one year |
| 19 | `8a8f0be00bd9` | compositional | When did Toda Of Pamplona's mother die? | 890 → After 890 | hedged date |
| 20 | `8cd1b0020bdd` | compositional | Birthplace of Atang De La Rama's husband? | Hagonoy → Tondo, Manila | wrong place in KG |
| 21 | `31c340da0bde` | compositional | Death of Julia Livia's father? | 14 Sep 23 → Sep 14, 14 AD | wrong year |
| 22 | `7d87730c0872` | comparison | Wizards Lost Kingdom + Final Exam same country? | no → Yes | wrong country in KG |
| 23 | `7a894f7c0bdd` | compositional | When did Through A Glass Darkly composer die? | 28 July 1750 → 1750 | date granularity |
| 24 | `84c5154a0bdb` | compositional | Nationality of Mimi (Song) performer? | French → French and Italian | dual nationality |

**Pattern**: 23 compositional, 1 comparison. Sub-types: dual nationality (7), wrong date (4), wrong place (5), date/place granularity (5), multiple spouses/other (3).

---

### RC7: Text Retrieval Failure (8 questions)

Neither KG nor text retrieval succeeded. Wikipedia search didn't find the right page, or extracted wrong information.

| # | Question ID | Type | Question (truncated) | Gold → Pred |
|---|---|---|---|---|
| 1 | `f05423560bda` | compositional | Birthplace of Laapataa Ladies dir? | Mrauk U → Unknown |
| 2 | `371c1d5e0bde` | compositional | Nationality of Fred Astaire's wife? | American → United Kingdom |
| 3 | `a96fec4c0bdd` | compositional | Death place of Bayezid I's father? | Constantinople → Ottoman Empire |
| 4 | `561a61ae0bdc` | compositional | Death place of Sirikit's husband? | Siriraj Hospital → Bangkok |
| 5 | `ef9dd4360bdd` | compositional | Nationality of Heather Jansch's husband? | Scottish → Turkish |
| 6 | `33f51d7e0bde` | compositional | Nationality of Blood Street director? | Chinese → American |
| 7 | `d6c5f2680bdd` | compositional | Nationality of Beatrice Of Savoy's mother? | France → Swiss |
| 8 | `d6e0f7f00bda` | compositional | Nationality of Yeh Gulistan Hamara director? | Afghanistan → Indian |

**Pattern**: All 8 compositional. Mostly obscure entities where Wikipedia doesn't have a dedicated page or retrieves wrong person/nationality.

---

## Detailed Root Cause Analysis

The sections below provide the detailed mechanistic analysis for each root cause, including code paths, fix proposals, and architectural implications.

---

## Root Cause 1: SPARQL Coverage Asymmetry (was "Comparison Inversion")

### What the original analysis said
> "SPARQL picks the EXACT OPPOSITE film/person — all 16/16 inverted. This smells like a systematic date comparison bug."

### What we actually found
**There is no comparison operator bug.** No wrong `<` vs `>`, no string-vs-numeric date issue, no ORDER BY inversion.

The actual mechanism is **coverage asymmetry with LLM default behavior**:

```
Question: "Which film has the director born first, Film A or Film B?"

What happens:
1. SPARQL generates query for both films' directors + birth dates
2. DBpedia has birth date for Film A's director but NOT Film B's
3. LLM sees one result row → defaults to Film A as the answer
4. Film A is ALWAYS wrong (the film with data is always the wrong answer)

Why always wrong: DBpedia's coverage gaps are systematic — the correct
answer's director is typically more obscure (less-covered in KG).
```

**Evidence**: 19/19 tested cases show `result_film == predicted_answer` (the film with available data), and the gold answer is always the film with zero data rows.

### Code path
```
routing.py:134        → rule:bridge_comparison matched
routing.py:303        → verify_sparql_feasibility() → ASK succeeds (for ONE entity)
sparql_solver.py:64   → UNIFIED_SPARQL_PROMPT generates comparison SPARQL
sparql_evaluation.py:276 → execute_sparql() returns data for 1 of 2 entities
sparql_evaluation.py:356 → ANSWER_PROMPT — NO instruction for partial data
                           → LLM defaults to entity with data → WRONG
```

### Fix
Add one instruction to `ANSWER_PROMPT` in `sparql_evaluation.py:356`:

> "For comparison questions, if data for one entity is missing, return 'could not determine' rather than defaulting to the entity with available data."

This makes `sparql_solver.py:218` treat it as a null answer and fall through to text.

**⚠ Implementation note (2026-03-31 validation):** The null-answer string MUST match `_NULL_ANSWERS` in `sparql_solver.py:44–47`, which checks for `"could not determine"`. Using `"cannot determine"` (as originally proposed) would bypass the null filter and produce a wrong-shaped answer instead of triggering text fallback.

**Estimated impact**: 19 questions recovered (Category A cases).

> **Partial fix applied (2026-03-31):** ASK verification fix in `routing.py` recovered 12/23 bridge_comparison questions (12 EM 0→1, 0 regressions). The 11 remaining failures are coverage-asymmetry cases not addressable by routing alone.

---

## Root Cause 2: Entity Linking Noise Injection (was "Entity Linking Errors")

### What the original analysis said
> "10 questions, 13.5%. Spotlight links to wrong DBpedia entities."

### What we actually found
**The scale is much larger than reported.** Entity linking injects noise entities in **38.8%** (388/1000) of all questions. The primary source is NOT Spotlight — it's the `bif:contains` fulltext fallback.

### Four distinct failure modes

#### Mode A — `bif:contains` Film Entity Injection (388 questions, 38.8%)

Two DBpedia entities appear in results for nearly every film/director question:
- `Kenji Mizoguchi: The Life of a Film Director` — 270 questions
- `Film in Which There Appear Edge Lettering, Sprocket Holes, Dirt Particles, Etc.` — 263 questions

**Root mechanism**: `_extract_type_hint()` returns `dbo:Film`, and these entities satisfy the type filter because they ARE typed as `dbo:Film`. The type filter constrains to a class, not semantic relevance.

**Critical gap**: Results from the `bif:contains` unlinked loop are **never post-validated** by `_validate_entity_types()` (entity_linker.py:371). This function only runs on Spotlight entities.

**Impact**: Of 388 affected questions, 92 have EM=0. Most failures occur on the TEXT route (noise entities pollute Wikipedia context retrieval), not SPARQL.

#### Mode B — Generic Word Entities (~440 questions, ~44%)

`'birth'` and `'death'` are NOT in `_MENTION_BLOCKLIST` (only `'born'` and `'died'` are).
These resolve to `dbp:birth`, `dbp:death` via exact label match. Similarly `'What'` → `dbr:What`, `'father'` → `dbr:Father_(disambiguation)`.

**Impact**: These clutter schema context but rarely cause EM=0 because SPARQL generation skips property-type URIs.

#### Mode C — Complete Entity Linking Failure (rare, ~1 confirmed SPARQL failure)

When Spotlight AND URI probe both fail, `bif:contains` returns only semantically irrelevant entities. Example: "The Private Life Of Cinema" → only entity found is `Kenji_Mizoguchi:_The_Life_of_a_Film_Director` → SPARQL queries wrong director's birthplace.

#### Mode D — URI Suffix Mismatch (~58 cases)

Linker returns `dbr:X_(film)` but hop-verification expects `dbr:X`. Systematic scan finds ~58 affected questions (original "13" was from a 20-question spot-check). Mostly a measurement artifact — SPARQL queries the `_(film)` URI which has correct properties.

### Fixes ranked by impact/effort

| Priority | Fix | Location | Impact |
|---|---|---|---|
| 1 | Add `'birth'`, `'death'`, `'father'`, `'law'` to `_MENTION_BLOCKLIST` | `entity_linker.py` | ~440 questions cleaner |
| 2 | Jaccard gate on `bif:contains` results (reject Jaccard < 0.3 vs query mention) | `entity_linker.py` | 388 noise entities removed |
| 3 | Block known noise URIs (the two recurrent offenders) | `entity_linker.py` | Immediate, but brittle |
| 4 | Extend `_validate_entity_types` to `bif:contains` results | `entity_linker.py` | Prevents unvalidated noise |

---

## Root Cause 3: Decomposition Comparison Failures (~26 cases)

*Note: Sub-pattern counts below are pattern estimates from 1–2 documented examples each, not full enumerations. Treat as directional.*

### Three sub-patterns

#### Sub-pattern A1 — Country Label Non-Normalization (~9 cases)

Sub-Qs return DBpedia values at wrong granularity: `"New York City"` vs `"Hungary"`, or `"East Germany"` vs `"Germany"`. `FINAL_ANSWER_PROMPT` sees two different strings and says "No" (different country), when the correct answer is "Yes" (same country).

**Example**: "Are directors of Film X and Film Y from the same country?"
- Sub-Q1 returns `"New York City"` (city, not country)
- Sub-Q2 returns `"Hungary"`
- LLM: "No, different countries" → Wrong (both are actually from the same country)

**Why LogicRAG avoids this**: It retrieves full Wikipedia text that says "American film director", not a DBpedia property value.

**Fix**: Add normalization instruction to `FINAL_ANSWER_PROMPT` in `logicrag_prompts.py:35`: "Treat cities/states as their containing country. Treat 'East Germany'/'West Germany' as 'Germany'."

#### Sub-pattern A2 — "How Old" Sub-Questions (~11 cases)

`DECOMPOSE_PROMPT` sometimes generates sub-Qs like "How old is the director of Film X?" instead of "When was [Q0] born?". The TEXT path retrieves a Wikipedia age (e.g., 66) rather than a birth year. Since age changes over time and is lost at death, the comparison is unreliable.

**Example**: Inside Man → Spike Lee (66) vs Alberto Lattuada (62 at death, but actually born 1914 vs 1957 — Lattuada is far older).

**Fix**: Add to `DECOMPOSE_PROMPT` at `logicrag_decomposition.py:62`: "NEVER ask 'How old is X?'. Always ask for the birth date/year instead."

#### Sub-pattern A3 — Spotlight Entity Disambiguation (3+ cases)

"St. Mary's School, Patiala" → Spotlight links to `St._Mary's_Catholic_High_School,_Fujairah` (UAE). Sub-Q returns "UAE" instead of "India".

**Fix**: Year-hint validation post-Spotlight; disambiguation via question context.

---

## Root Cause 4: Decomposition Bridge/General Failures (~22 cases)

*Note: Sub-pattern counts below are pattern estimates, not full enumerations.*

### Three sub-patterns

#### Sub-pattern B1 — SPARQL Retry Confabulation (~10 cases)

When initial SPARQL returns empty results, the retry generates a broader query that returns multiple rows. The LLM picks the **first** result, which is arbitrary.

**Example**: Birthplace of Christian August, Prince of Anhalt-Zerbst → retry returns `[Gommern, Holy_Roman_Empire, Principality_of_Anhalt-Dornburg]` → answer "Gommern" (first). Gold: Dornburg (third).

**Worst case**: "Archduchess Maria Antonia Of Austria (1899–1977)" → Spotlight links to `Marie_Antoinette` (confidence 1.0, the 1755 figure). The `(1899–1977)` year hint is completely ignored.

**Fix**: In `sparql_solver.py:194-215`, pass all rows to LLM and instruct: "Pick the most specific and contextually relevant result, not the first."

#### Sub-pattern B2 — Multi-Variable SPARQL Strips Entity Name (~3 cases)

Sub-Q for "Who is X's husband?" generates `SELECT ?husband ?husbandBirthDate WHERE { ... }` but the `?person` binding is never projected. LLM reads `[{'husbandBirthDate': '1943-11-03'}]` and says "husband unknown". Sub-Q1 then searches for "person born 1943-11-03" → noise results.

**Fix**: In SPARQL generation prompt, instruct: "Always project the primary entity variable (name/label) alongside its properties."

#### Sub-pattern B3 — TEXT Sub-Q Propagation Failure (~12 cases)

`text_solver.solve()` returns a plain string (no entity list). When sub-Q0 resolves via TEXT, `known_entities[0]` stays empty. The entity scoping logic (lines 627–633 in `logicrag_decomposition.py`) gives dependent sub-Q1 an empty entity list. Sub-Q1 then falls back to root-question entities (which may be wrong due to Spotlight errors).

**Corrected mechanism (2026-03-31 validation):** `_resolve_answer_entities()` (defined at line 319) actually runs for ALL dependent sub-Qs at lines 482–489 regardless of SPARQL/TEXT route — it adds entities to `all_entities` (context). The actual gap is that its output is **never propagated back to `known_entities`** (the scoping dict at lines 664–666). So the entity scoping block at 627–633 still gets an empty list for TEXT-resolved dependencies.

**Fix**: At `logicrag_decomposition.py:664-666`, propagate `_resolve_answer_entities()` output into `known_entities[i]` so downstream sub-Qs can scope to TEXT-resolved entities.

---

## Root Cause 5: Evidence Hedging (6 cases)

### Two distinct sub-mechanisms

#### Sub-mechanism A — SPARQL UNION Cross-Contamination (2–3 cases)

For parallel comparison sub-Qs in LogicRAG decomposition, `effective_parents = list(entities)` at `logicrag_decomposition.py:633` passes BOTH comparison entities to the SPARQL solver. The LLM generates a UNION query fetching all dates together with **no entity labels per row**.

**Example** (Thomas Praytor vs Matthew Knox): SPARQL returns `1990-02-28` (Praytor) and `1999-12-22` (Knox, appears twice via `dbo:` + `dbp:`). LLM picks majority value `1999-12-22` as Praytor's birthday → both appear equal → "same age."

**Fix**: Scope `all_entities` to only the sub-Q's named entity before passing to `sparql_solver.solve()`.

#### Sub-mechanism B — `known_facts` Bleeding (2 cases)

`logicrag_decomposition.py:551` passes ALL accumulated `known_facts` to `text_solver.solve()` regardless of dependency. The text solver prompt says "trust KG facts — they are verified structured data." The LLM treats a prior sub-Q's answer (e.g., Manuel Díaz Rodríguez's birthdate) as the current sub-Q's answer (Mona Tyndall), overriding Wikipedia text.

**Fix**: At line 551, filter `known_facts` to include only facts from `sub_q.depends_on` sub-Qs.

---

## Architectural Analysis: Why LogicRAG Avoids All 5 Failure Modes

| Hybrid Component | Failure Mode | LogicRAG Equivalent | Why LogicRAG Succeeds |
|---|---|---|---|
| Entity linking (Spotlight + bif:contains) | Noise entities, wrong entities, cross-contamination | None — corpus search by text | No entity linking = no entity linking errors |
| SPARQL generation + execution | Coverage asymmetry, retry confabulation, projection errors | None — all retrieval from corpus | No SPARQL = no SPARQL bugs |
| Per-sub-Q routing (SPARQL vs TEXT) | Each hop can fail independently, error compounds | None — all sub-Qs go to corpus | No routing = no routing errors |
| Schema projection | Wrong type hints, missing properties | None — no schema | No schema = no schema errors |
| `known_facts` + entity scoping | Cross-contamination between parallel sub-Qs | Accumulated summary (single text blob) | Summary is additive, not entity-scoped |
| `FINAL_ANSWER_PROMPT` with raw KG values | Non-normalized values ("NYC" vs "Hungary") | LLM reads Wikipedia prose ("American director") | Natural language is already normalized |

**The fundamental trade-off**: LogicRAG's corpus-only approach is immune to KG pipeline errors because it has no KG pipeline. But it's less precise when KG data IS correct — which is why hybrid wins on 170 other questions. The hybrid pipeline's additional components (entity linking, SPARQL, routing, schema) each add both capability and failure surface area.

---

## Fix Priority Matrix

Ranked by (questions recovered × feasibility). Updated with 147-failure counts.

| Priority | Fix | Root Cause | Est. Impact | Effort | Location |
|---|---|---|---|---|---|
| **P0** | Add partial-data instruction to ANSWER_PROMPT | RC1 | ~20 questions | 1 line | `sparql_evaluation.py:356` |
| **P0** | Country normalization in FINAL_ANSWER_PROMPT | RC3 | ~21 questions | 2 sentences | `logicrag_prompts.py:35` |
| **P0** | Scope `all_entities` per sub-Q for comparisons | RC5 | ~8 questions | 2 lines | `logicrag_decomposition.py:633` |
| **P0** | Filter `known_facts` by `depends_on` | RC5 | ~5 questions | 1 line | `logicrag_decomposition.py:551` |
| **P1** | Propagate `_resolve_answer_entities()` → `known_entities` | RC4-B3 | ~16 questions | 5 lines | `logicrag_decomposition.py:664` |
| **P1** | Blocklist `'birth'`/`'death'`/`'father'`/`'law'` | RC2 | ~440q cleaner | 1 line | `entity_linker.py` |
| **P1** | Jaccard gate on `bif:contains` (reject < 0.3) | RC2 | ~29 questions | 5 lines | `entity_linker.py` |
| **P1** | Add gender filter for family relation SPARQL | RC4-B1 | ~7 questions | 1 paragraph | SPARQL prompt |
| **P2** | Pass all SPARQL retry rows + pick-best instruction | RC4-B1 | ~8 questions | 3 lines | `sparql_solver.py:194` |
| **P2** | Always project entity name in multi-var SPARQL | RC4-B2 | ~3 questions | 1 sentence | SPARQL prompt |
| **P2** | Extend `_validate_entity_types` to `bif:contains` | RC2 | noise reduction | 10 lines | `entity_linker.py:371` |
| **P3** | Year-hint validation post-Spotlight | RC2 | ~4 questions | 15 lines | `hybrid_qa.py` |

**Total estimated recovery with all fixes**:
- P0 fixes: ~49 questions (RC1: 20, RC3: 21, RC5: ≤8 — two sub-fixes overlap within 8 questions)
- P1 fixes: ~30 questions (RC4-B3: 15, RC2 noise: ~8, RC4-B1 family: 7)
- P2 fixes: ~11 questions (RC4-B1 retry: 8, RC4-B2: 3)
- **Total actionable**: ~90 of 147 (61%)
- **Unactionable**: RC6 (24) + RC7 (8) = 32 questions (22%) — KG data errors + retrieval dead-ends
- **Theoretical ceiling**: 85.3% → ~94% with all P0+P1+P2 fixes (optimistic, assumes no regressions)

---

## Discussion Points

### 1. Should we implement these fixes for the thesis?

**Pro**: Recovering ~95 questions would push hybrid from 85.3% → ~95% judge-correct, widening the gap vs LogicRAG's 74.5%. The top 3 fixes (RC1 partial-data, RC3 country normalization, RC4-B3 propagation) alone could recover ~57 questions.

> **Update (2026-03-31):** The ASK verification fix in `routing.py` (comparison questions now use `all()` logic for multi-entity ASK) recovered 12 of the 23 affected bridge_comparison questions. Hybrid 1000q Judge EM improved from 84.1% to 85.3% (+1.2pp). This is a partial implementation of the P0 fixes listed above; remaining fixes (entity scoping, known_facts filtering, ANSWER_PROMPT partial-data instruction) are not yet applied.

**Con**: These are post-hoc fixes derived from analyzing the failure set. Any reported improvements should clearly state they're "post-analysis corrections" to avoid overfitting claims.

### 1b. Error budget breakdown (new, from 147q classification)

The 147 failures decompose into: **78.2% actionable** (code fixable) vs **21.8% unactionable** (KG data quality + retrieval limits). This is a strong signal — the hybrid pipeline's error rate is dominated by engineering issues, not fundamental architectural limits. The 24 RC6 questions (wrong KG data) represent a hard floor regardless of code quality.

### 2. Is the coverage asymmetry finding publishable?

The finding that "16/16 SPARQL comparison inversions are actually coverage gaps, not operator bugs" is a non-obvious result. It reframes the problem from "our comparison logic is buggy" to "KG coverage gaps cause systematic LLM defaults." This is a genuine insight about KG-augmented QA systems.

### 3. Entity linking noise — systemic vs local?

The `bif:contains` noise (38.8% of questions affected) is a systemic design flaw, not a per-question bug. The question is whether to:
- (a) Fix entity linking quality (Jaccard gate, extended validation) — improves the system
- (b) Document it as a limitation of KG-augmented approaches — adds to the thesis argument
- (c) Both — fix what we can, document what we can't

### 4. LogicRAG's corpus advantage — is it "unfair"?

LogicRAG retrieves from the 2WikiMultiHopQA corpus, which contains curated Wikipedia passages already aligned with the questions. Our hybrid pipeline retrieves from live Wikipedia, which is noisier but more general. This corpus advantage partly explains LogicRAG's success on the 74 questions — the answer is often sitting in a pre-curated passage.

**Question**: Should we run hybrid with the same corpus to isolate the KG pipeline effect from the retrieval source effect?

### 5. 24 questions are DBpedia data quality errors

Several failures trace to DBpedia recording wrong dates or missing entities (e.g., "Moguds Pellams" director born 1946 per DBpedia, gold answer says 1949). These are fundamentally unrecoverable by code fixes. They should be documented as a KG data quality limitation.

---

## Unresolved Questions

1. **Retry confabulation**: Is the SPARQL retry generating broader queries by design, or is the LLM hallucinating when the original query fails? Need to audit retry prompts.

2. **Entity scoping completeness**: The fix for comparison entity cross-contamination (scope `all_entities` per sub-Q) may break bridge entity chains where the parent entity IS needed. Need to test both comparison and bridge paths.

3. **TEXT→SPARQL bridge propagation**: `_resolve_answer_entities()` already runs for all dependent sub-Qs (lines 482–489), but its output feeds `all_entities` without propagating back to `known_entities`. The fix is propagation, not invocation — but does `_resolve_answer_entities()` reliably extract entities from free-text answers (vs structured SPARQL URIs)?

4. **`bif:contains` replacement**: If we add a Jaccard gate, what threshold? 0.3 seems reasonable but needs empirical validation on the 1000q set.

---

## Validation Log (2026-03-31)

Three independent debugger agents verified every code-path claim against source. Corrections applied:

| # | Correction | Severity |
|---|---|---|
| 1 | `execute_sparql()` citation: `sparql_solver.py:82` → `sparql_evaluation.py:276` | Minor (non-material) |
| 2 | Fix 1 null-string: `"cannot determine"` → `"could not determine"` to match `_NULL_ANSWERS` | **Critical** (would cause silent failure) |
| 3 | Mode B count: `421` → `~440` (systematic scan = 436–446) | Minor (ballpark was correct) |
| 4 | Mode D count: `13` → `~58` (original from 20-question spot-check, 4.5x undercount) | **Significant** |
| 5 | RC4 B3 mechanism: `_resolve_answer_entities()` runs for ALL paths, not SPARQL-only; actual gap is `known_entities` propagation | **Significant** (fix unchanged, mechanism description corrected) |
| 6 | Unresolved Q3: corrected same misattribution as #5 | Minor |
| 7 | RC3/RC4 sub-pattern counts flagged as pattern estimates (~), not enumerations | Transparency |

**Verification scores**: RC1 9/10 verified, RC2 5/8 verified (2 counts corrected, 1 location), RC3 code 3/3 verified (counts unverified), RC4 3/5 verified (1 significant mechanism correction), RC5 3/3 fully verified.

**Net assessment**: All 5 root cause mechanisms are architecturally sound. The recovery estimate (~45–55 questions) should be treated as directional, not precise — underlying cluster counts are estimates without complete question-ID enumeration.

Validation reports: `plans/reports/debugger-260331-0847-rc*-claim-validation.md`

## Validation Log (2026-04-01)

Full factcheck of all 147 question IDs, arithmetic, code paths, and per-RC pattern claims.

| # | Correction | Severity |
|---|---|---|
| 8 | 5 fabricated RC7 IDs (`a3dd79820bde`, `0a1c99c80bdc`, `7fcd23020bde`, `f15d72900bda`, `c29b1b0e08c4`) — not in 1000q or dev_full.json. Replaced with 3 real failures. | **Critical** |
| 9 | RC6 #23 (`a3e5f6e60867`) is `judge_correct=True` — not a failure. Removed. | **Critical** |
| 10 | 6 real failures missing from all RC tables. Classified and added: 1→RC2, 2→RC6, 3→RC7. | **Significant** |
| 11 | RC7 count: 10 → **8**. RC6 count: 23 → **24**. RC2 count: 29 → **30**. | **Significant** |
| 12 | RC3 "15/21 gold=yes pred=No" → **13/21** (2 false positives + 6 historical polity miscounted). | Minor |
| 13 | RC4 sub-pattern summary: B3=16 → **15**, wrong property=5 → **other=6**. Table itself was correct; only summary line was wrong. | Minor |
| 14 | Fix Priority P0 RC5: two sub-fixes (8+5=13) overlap within 8 total RC5 questions. Corrected to ≤8. Total actionable estimate: ~95 → **~90**. | Minor |

**Net assessment**: Core framework (7 root causes, actionable/unactionable split, code path analysis, fix proposals) is sound. The RC7 ID fabrication was the most severe issue — inflated an entire category by 2x. All cross-tables and percentages updated to match corrected counts.

Factcheck report: `plans/reports/factcheck-260331-2349-root-cause-analysis-validation.md`

---

## Source Reports

Detailed investigation reports with full code traces and line numbers:

1. [SPARQL Comparison Inversion + Hedging](../plans/reports/investigation-260330-sparql-comparison-inversion.md) — 16 comparison failures + 6 hedging cases
2. [Entity Linking Failures](../plans/reports/investigation-260330-entity-linking-failures.md) — `bif:contains` noise injection, type validation gaps
3. [Decomposition Failures](../plans/reports/investigation-260330-decomposition-failures.md) — 26 comparison + 22 bridge/general failures
4. [Original Head-to-Head Analysis](../plans/reports/factcheck-260330-logicrag-head-to-head-analysis.md) — Surface-level 5-category classification
5. [LogicRAG Architecture Deep Dive](../plans/reports/Explore-260329-LogicRAG-Pipeline.md) — Architectural comparison
