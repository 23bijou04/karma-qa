# Corrected Ensemble Entity Linking Test Results

**Generated**: 2026-03-18 09:37 UTC
**Questions tested**: 100
**Mention entities evaluated**: 138 (bridge excluded: 101)
**Matching**: Strict URI only (no label-substring fallback)

## Methodology Fixes

1. **Mention-only gold**: Only entities whose name appears in the question text
2. **Strict URI matching**: Normalized URI comparison, no label-substring fallback
3. **Context-augmented WikiSearch**: Type hints from question appended to queries
4. **100 questions**: 25 per type (stratified, seed=42)

## Executive Summary

| Approach | Mention Recall | Found/Total |
|----------|---------------|-------------|
| Entity Linker (Spotlight+bif:contains) | 65.9% | 91/138 |
| WikiSearch-only | 60.1% | 83/138 |
| Ensemble | 68.1% | 94/138 |

Ensemble gain over Spotlight: +2.2pp

| Category | Count |
|----------|-------|
| Ensemble helps (ENS finds what EL misses) | 3 |
| Ensemble neutral (EL finds, ENS doesn't) | 0 |
| Both missed | 44 |

## By Question Type

| Type | N | Mention Ents | EL Recall | WS Recall | ENS Recall | Gain |
|------|---|-------------|-----------|-----------|------------|------|
| bridge_comparison | 25 | 50 | 64.0% | 54.0% | 66.0% | +2.0pp |
| comparison | 25 | 50 | 76.0% | 68.0% | 76.0% | +0.0pp |
| compositional | 25 | 23 | 52.2% | 56.5% | 56.5% | +4.3pp |
| inference | 25 | 15 | 60.0% | 60.0% | 66.7% | +6.7pp |

## Where Ensemble Helps (ENS finds what Entity Linker misses)

| Question (truncated) | Entity | Gold URI | WS URI |
|---------------------|--------|----------|--------|
| Which award the performer of song See The Usa In Y... | See the USA in Your Chevrolet | `dbr:See_the_USA_in_Your_Chevrolet` | `dbr:See_the_USA_in_Your_Chevrolet` |
| Who is Anne Woodville's father-in-law? | Anne Woodville | `dbr:Anne_Woodville` | `dbr:Anne_Woodville` |
| Are director of film Susanna Whipped Cream and dir... | Le salamandre | `dbr:Le_salamandre` | `dbr:Le_salamandre` |

## Both Missed (mention entity not found by either)

| Question (truncated) | Entity | Gold URI | SP wrong | WS wrong |
|---------------------|--------|----------|----------|----------|
| Who is Eberhard Of Limburg's paternal grandfather? | Eberhard of Limburg | `dbr:Eberhard_of_Limburg` | `dbr:Eberhard_of_Franconia` | `dbr:Eberhard_of_Franconia` |
| Which film whose director is younger, Kalippava or... | A Couch in New York | `dbr:A_Couch_in_New_York` | `dbr:Couch` | `dbr:Couch` |
| Who is Gilukhipa's paternal grandfather? | Gilukhipa | `dbr:Gilukhipa` | `—` | `dbr:Cleopatra` |
| Are both universities, African Institute For Futur... | African Institute for Future Studies | `dbr:African_Institute_for_Future_Studies` | `—` | `dbr:Kisangani` |
| Which film was released more recently, An Empress ... | An Empress and the Warriors | `dbr:An_Empress_and_the_Warriors` | `—` | `—` |
| Which film was released more recently, An Empress ... | Die Screaming, Marianne | `dbr:Die_Screaming,_Marianne` | `dbr:Marianne` | `dbr:Marianne` |
| Which film was released more recently, Shravana Ba... | Hitler's Folly | `dbr:Hitler's_Folly` | `dbr:Category:Hitler` | `dbr:Category:Hitler` |
| Which film has the director born first, A.P.E.X. o... | Quax in Africa | `dbr:Quax_in_Africa` | `dbr:Quax` | `dbr:Quax` |
| Which film has the director who died first, Haiduc... | Haiducii (film) | `dbr:Haiducii_(film)` | `dbr:Haiducii` | `dbr:Haiducii` |
| Which film has the director who is older, Four Day... | Four Days in November | `dbr:Four_Days_in_November` | `dbr:November` | `dbr:November` |
| Which country Nizar Al-Mustafa's father is from? | Nizar | `dbr:Nizar` | `dbr:Nizar_ibn_al-Mustansir` | `dbr:Nizar_ibn_al-Mustansir` |
| What is the place of birth of the composer of film... | Inherent Vice | `dbr:Inherent_Vice` | `dbr:Inherent_Vice_(film)` | `dbr:Inherent_Vice_(film)` |
| Who is Prince Mired Bin Ra'Ad's paternal grandmoth... | Prince Mired bin Ra'ad | `dbr:Prince_Mired_bin_Ra'ad` | `dbr:Prince` | `dbr:Prince` |
| Which film has the director who was born later, He... | The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics | `dbr:The_Blue_Collar_Worker_and_the_Hairdresser_in_a_Whirl_of_Sex_and_Politics` | `dbr:Blue-collar_worker` | `dbr:Blue-collar_worker` |
| Which film has the director born first, Mutiny (19... | Mutiny (1952 film) | `dbr:Mutiny_(1952_film)` | `dbr:Mutiny` | `dbr:Mutiny` |
| Which film whose director is younger, Return Of Th... | Return of the Hero | `dbr:Return_of_the_Hero` | `dbr:Return` | `dbr:Return` |
| Which film whose director is younger, Wolves Of Th... | Wolves of the North | `dbr:Wolves_of_the_North` | `dbr:Category:Wolves` | `dbr:Category:Wolves` |
| Which film whose director is younger, Wolves Of Th... | Action in Arabia | `dbr:Action_in_Arabia` | `dbr:Arabian_Peninsula` | `dbr:Arabian_Peninsula` |
| Who lived longer, Sir Thomas Wheate, 1St Baronet o... | Sir Thomas Wheate, 1st Baronet | `dbr:Sir_Thomas_Wheate,_1st_Baronet` | `dbr:Thomas_Wheate` | `dbr:Thomas_Wheate` |
| What is the place of birth of the director of film... | Being Human | `dbr:Being_Human` | `dbr:Being_Human_(2005_film)` | `dbr:Being_Human_(2005_film)` |
| Which film was released first, Moonnam Pakkam or T... | Tribute to the Teachers | `dbr:Tribute_to_the_Teachers` | `dbr:Abbas_Kiarostami` | `dbr:Abbas_Kiarostami` |
| Which film came out first, Mama'S Little Pirate or... | Mama's Little Pirate | `dbr:Mama's_Little_Pirate` | `—` | `—` |
| Which film has the director died later, Rosaura At... | Rosaura at 10 O'Clock | `dbr:Rosaura_at_10_O'Clock` | `dbr:Rosaura` | `dbr:Rosaura` |
| Which film has the director died later, Rosaura At... | Uncle's New Blazer | `dbr:Uncle's_New_Blazer` | `—` | `dbr:Tom_Holland_(filmmaker)` |
| Which film was released first, A Romance Of Sevill... | A Romance of Seville | `dbr:A_Romance_of_Seville` | `http://dbpedia.org/property/romance` | `http://dbpedia.org/property/romance` |
| Which film was released first, A Romance Of Sevill... | Pirates of the Sky | `dbr:Pirates_of_the_Sky` | `dbr:Pittsburgh_Pirates` | `dbr:Pittsburgh_Pirates` |
| Are Victoria Winter and Jon Sullivan of the same n... | Victoria Winter | `dbr:Victoria_Winter` | `—` | `dbr:Victoria_Winters` |
| Which film has the director born later, Romance On... | The Palace of Angels | `dbr:The_Palace_of_Angels` | `dbr:Palace` | `dbr:Palace` |
| What is the place of birth of the performer of son... | Damage | `dbr:Damage` | `dbr:Make_This_Your_Own` | `dbr:Make_This_Your_Own` |
| Which film has the director who was born first, Th... | The Parson and the Outlaw | `dbr:The_Parson_and_the_Outlaw` | `dbr:Parson` | `dbr:Parson` |
| Do both films The Umbrella Coup and The Cow And I ... | The Cow and I | `dbr:The_Cow_and_I` | `dbr:Cow` | `dbr:Cow` |
| Where was the place of burial of Princess Augusta ... | Princess Augusta of Württemberg | `dbr:Princess_Augusta_of_Württemberg` | `—` | `—` |
| Where was the director of film Eight Days A Week (... | Eight Days a Week | `dbr:Eight_Days_a_Week` | `dbr:Eight_Days` | `dbr:Eight_Days` |
| Are both Cunaú River and Rio Verde (Mato Grosso Do... | Rio Verde (Mato Grosso do Sul) | `dbr:Rio_Verde_(Mato_Grosso_do_Sul)` | `dbr:Mato_Grosso` | `dbr:Mato_Grosso` |
| Where did Fedor Frešo's father die? | Fedor Frešo | `dbr:Fedor_Frešo` | `—` | `dbr:List_of_Slovaks` |
| Who is Archduchess Maria Antonia Of Austria (1899–... | Maria Antonia of Austria | `dbr:Maria_Antonia_of_Austria` | `dbr:Marie_Antoinette` | `dbr:Marie_Antoinette` |
| Which film has the director who is older than the ... | Women's Weapons | `dbr:Women's_Weapons` | `—` | `dbr:Weapons_(2025_film)` |
| Which film has the director who is older than the ... | She Wants Me | `dbr:She_Wants_Me` | `dbr:She` | `dbr:She` |
| Where does Sharon Tiller's husband work at? | Sharon Tiller | `dbr:Sharon_Tiller` | `—` | `dbr:Ruby_Holler` |
| Which film was released first, College Coach or Lo... | Los Últimos Días | `dbr:Los_Últimos_Días` | `—` | `dbr:The_Last_Days_(2013_film)` |
| Who is Margaret Of Baux's father-in-law? | Margaret of Baux | `dbr:Margaret_of_Baux` | `dbr:Margaret` | `dbr:Margaret` |
| Where did Sylvia Burka's husband die? | Sylvia Burka | `dbr:Sylvia_Burka` | `—` | `dbr:Reform_UK` |
| Which film whose director is younger, Term Of Tria... | Would You Marry Me? | `dbr:Would_You_Marry_Me%3F` | `dbr:You` | `dbr:You` |
| Where did the composer of film Nighthawks (1981 Fi... | Nighthawks | `dbr:Nighthawks` | `dbr:Nighthawks_(1981_film)` | `dbr:Nighthawks_(1981_film)` |

## Bridge Entities (excluded from eval)

Total bridge entities: 101 (42% of all gold entities)

## Recommendation

**PROCEED** with ensemble implementation.
Ensemble delivers +2.2pp recall and fixes 3 entities that the entity linker misses.
