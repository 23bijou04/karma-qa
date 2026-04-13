"""
Write final judge results for LinearRAG part2 (333 records).
Manual decisions for all 82 ambiguous cases are hardcoded here.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data/eval_1000q/judge_input/linearrag_part2.jsonl"
OUTPUT = PROJECT_ROOT / "data/eval_1000q/judge_output/linearrag_part2.jsonl"

# Manual decisions for ambiguous cases
# Format: id -> True/False
# Reviewed one by one against matching rules
MANUAL = {
    # Q: spouse of director of Apprentice To Murder. Gold=Dorothy Thomas, Pred=Vivienne Leebosh → wrong person → NO
    "3ac0ebec0bde11eba7f7acde48001122": False,

    # Q: which country is Konrad Wolf's father from. Gold=East Germany, Pred=Germany
    # East Germany and Germany are related but distinct political entities → NO (strict: wrong country)
    "c5e4c82a0bdd11eba7f7acde48001122": False,

    # Q: which award did director of Dream Is Alive earn. Gold=Member of the Order of Canada, Pred=Order of Canada
    # "Member of the Order of Canada" is the grade/honor; "Order of Canada" is the award — pred is gold minus rank prefix → YES
    "1332d0ca0bdb11eba7f7acde48001122": True,

    # Q: which country is director of Romanoff And Juliet (1961) from. Gold=United Kingdom, Pred=England
    # England is a constituent country within the UK → YES (admin region containing gold location)
    "8055aca40bdb11eba7f7acde48001122": True,

    # Q: nationality of performer of Daddy Come Home. Gold=United States, Pred=Unable to determine → NO (empty/non-answer)
    "3b76b5c80bdc11eba7f7acde48001122": False,

    # Q: date of death of director of The Organization. Gold=December 12 2012, Pred=Not available → NO
    "002e925e0bde11eba7f7acde48001122": False,

    # Q: maternal grandmother of Princess Catherine of Württemberg. Pred=info not provided → NO
    "8c3c6d9e0bb011ebab90acde48001122": False,

    # Q: where does Gwen Graham's father work. Pred=no answer → NO
    "096e42ba0bde11eba7f7acde48001122": False,

    # Q: place of birth of director of Do Not Marry Girls. Gold=Moscow, Pred=None → NO
    "89a94f720bdb11eba7f7acde48001122": False,

    # Q: birthday of Prince Raphael's mother. Pred=not mentioned → NO
    "43c910980bde11eba7f7acde48001122": False,

    # Q: award director of Sainikudu earned. Gold=Filmfare Awards South, Pred=Filmfare Award for Best Director – Telugu
    # Filmfare Award for Best Director – Telugu is a specific award under Filmfare Awards South → YES (pred more specific, gold contained)
    "6bde832c0bdb11eba7f7acde48001122": True,

    # Q: where did composer of Pepo (Film) die. Pred=text does not provide → NO
    "05ca5a800bda11eba7f7acde48001122": False,

    # Q: where was director of Katka's Reinette Apples born. Gold=Rēzekne, Pred=Rezekne (diacritic removed) → YES
    "fc2f7e460bda11eba7f7acde48001122": True,

    # Q: where was composer of Love Story 1999 born. Pred=not specified → NO
    "e5d895ee0bdd11eba7f7acde48001122": False,

    # Q: uncle of Filippa Fleming. Gold=Ivar Fleming, Pred=Joakim Fleming → different person → NO
    "1f3ff7880bb011ebab90acde48001122": False,

    # Q: where was performer of Znayu Ya born. Gold=L'viv, Pred=Ukraine
    # L'viv is a city IN Ukraine, but pred is broader/vaguer than gold → NO
    "b42ac0920bda11eba7f7acde48001122": False,

    # Q: place of birth of Rohese of Monmouth's husband. Pred=not specified → NO
    "49d710a20bde11eba7f7acde48001122": False,

    # Q: paternal grandfather of Johann Nicolaus Bach. Pred=text does not provide → NO
    "77f5e45a0bb011ebab90acde48001122": False,

    # Q: Tlilpotoncatzin's uncle. Gold=Chimalpopoca, Pred=Itzcoatl → different person → NO
    "896795760bb011ebab90acde48001122": False,

    # Q: award performer of You Changed Me won. Gold=Best Actor, Pred=Grammy Award → different award → NO
    "f06252fc0bdd11eba7f7acde48001122": False,

    # Q: nationality of composer of Mere Rashke Qamar. Gold=Pakistan, Pred=Indian → wrong nationality → NO
    "45153f740bdb11eba7f7acde48001122": False,

    # Q: where did composer of You've Got What Gets Me die. Gold=Hollywood, Pred=1937 → year not a place → NO
    "0069c0220bde11eba7f7acde48001122": False,

    # Q: place of birth of performer of Dip (Song). Gold=Port of Spain, Pred=Compton California → different cities → NO
    "cf918c0c0bdb11eba7f7acde48001122": False,

    # Q: Marjorie Flack's sibling-in-law. Gold=Stephen Vincent Benét, Pred=Stephen Vincent Benet (no accent) → same person → YES
    "c510c7b00baf11ebab90acde48001122": True,

    # Q: do directors of Friday 13th (1916) and Beaumarchais share same nationality. Gold=yes, Pred=No → NO
    "ae1baf9708c811ebbd92ac1f6bf848b6": False,

    # Q: nationality of Beatrice of Savoy's mother. Gold=France, Pred=None → NO
    "d6c5f2680bdd11eba7f7acde48001122": False,

    # Q: nationality of Lord Frederick Cavendish-Bentinck's father. Gold=Great Britain, Pred=British → YES (nationality adjective = country, plus Great Britain = UK area)
    "a9277caa0bdd11eba7f7acde48001122": True,

    # Q: father of director of Homage At Siesta Time. Gold=Leopoldo Torres Ríos, Pred=Leopoldo Torres Rios (no accent) → same person → YES
    "4a8ffd360bdb11eba7f7acde48001122": True,

    # Q: place of death of director of Mera Lahoo. Gold=Punjabi, Pred=not mentioned → NO
    "438a3de40bdb11eba7f7acde48001122": False,

    # Q: paternal grandfather of Charles Willoughby 10th Baron. Gold=William Willoughby 3rd Baron, Pred=Charles Willoughby 2nd Baron → different person → NO
    "2714c89a0baf11ebab90acde48001122": False,

    # Q: nationality of Margaretha van Mechelen's husband. Gold=Dutch Republic, Pred=Dutch → YES (nationality adj = country)
    "fd848f860bdd11eba7f7acde48001122": True,

    # Q: where did Grand Duchess Elena's father die. Gold=Saint Petersburg, Pred=1909 → year not a place → NO
    "767efc000bde11eba7f7acde48001122": False,

    # Q: where did Mate Meštrović's father graduate from. Gold=Academy of Fine Arts Vienna, Pred=Vienna
    # Pred is broader/vaguer than gold (city vs institution) → NO
    "3685f31a0bde11eba7f7acde48001122": False,

    # Q: where did director of Yamata die. Gold=London, Pred=The United Kingdom
    # UK is broader/vaguer than London → NO
    "7a6381980bdd11eba7f7acde48001122": False,

    # Q: which country is composer of Brecha (2009) from. Gold=Argentine, Pred=Argentina/France
    # Pred includes correct nationality but also adds France — multiple answers without gold clearly alone → NO
    "68a9e3c60bdc11eba7f7acde48001122": False,

    # Q: paternal grandmother of Teriimaevarua III. Gold=Pōmare IV, Pred=Queen Pomare IV of Tahiti → same person, full title → YES
    "fcd3ff500baf11ebab90acde48001122": True,

    # Q: spouse of director of Tonta Tonta Pero No Tanto. Pred=not enough info → NO
    "f48a4ace0bdd11eba7f7acde48001122": False,

    # Q: are movies Hellraiser Revelations and Night of Mystery from same country. Gold=yes, Pred=No → NO
    "a7f886c5094d11ebbdaeac1f6bf848b6": False,

    # Q: which film has director died earlier, Ethnic Notions or Gordon of Ghost City. Pred=cannot be answered → NO
    "215d741f08e511ebbda4ac1f6bf848b6": False,

    # Q: place of death of Joseph Ferdinand of Bavaria's father. Gold=Munich, Pred=Unknown → NO
    "1ff759ea0bde11eba7f7acde48001122": False,

    # Q: when did composer of Through a Glass Darkly (Film) die. Gold=28 July 1750, Pred=1750
    # Year only when gold has full date → NO (per rejection rules)
    "7a894f7c0bdd11eba7f7acde48001122": False,

    # Q: mother of Princess Adrienne, Duchess of Blekinge. Gold=Princess Madeleine Duchess of Hälsingland and Gästrikland, Pred=Princess Madeleine
    # Same person, pred is shortened form → YES
    "6837abca0bb011ebab90acde48001122": True,

    # Q: place of burial of Euphemia of Pomerania's father. Pred=not mentioned → NO
    "08bb14940bdc11eba7f7acde48001122": False,

    # Q: which film has director born earlier, Did A Good Man Die? or Last of the Knucklemen. Pred=not enough info → NO
    "b4e53bdc08c111ebbd8bac1f6bf848b6": False,

    # Q: did Side By Side (Band) and La Makina originate from same country. Gold=yes, Pred=No → NO
    "c9983efa086511ebbd5eac1f6bf848b6": False,

    # Q: date of birth of Archduke Karl Pius's mother. Pred=not available → NO
    "fde3a7880bdc11eba7f7acde48001122": False,

    # Q: which country is Anthony I Count of Ligny's father from. Gold=France, Pred=Luxembourg → NO
    "759305480bde11eba7f7acde48001122": False,

    # Q: place of death of director of Picture of Dorian Gray (1945). Pred=not specified → NO
    "68f864fe0bde11eba7f7acde48001122": False,

    # Q: place of birth of Tatiana Schlossberg's mother. Gold=Manhattan, Pred=New York City New York
    # New York City is broader than Manhattan (borough vs city) → Manhattan is IN New York City
    # Gold (Manhattan) is contained within pred (New York City) → pred is more general → NO per "pred broader than gold"
    # BUT wait: Manhattan is a borough of NYC. They're commonly used interchangeably for born-in context.
    # Strict rule: pred broader/vaguer than gold → NO
    "cfbc66400bd911eba7f7acde48001122": False,

    # Q: place of birth of Consort Liang's husband. Gold=Chinese, Pred=Unknown → NO
    "62ce8c8e0bde11eba7f7acde48001122": False,

    # Q: father-in-law of Từ Dụ. Pred=not enough info → NO
    "3ac032da0baf11ebab90acde48001122": False,

    # Q: which film's director died first, Showdown At Boot Hill or Walls of Hell. Gold=Showdown At Boot Hill, Pred=The Walls Of Hell → wrong answer → NO
    "433bcf3d08b711ebbd88ac1f6bf848b6": False,

    # Q: where was husband of Virginia Foster Durr born. Gold=Montgomery, Pred=Alabama
    # Montgomery is a city IN Alabama, pred is broader/vaguer → NO
    "24672c120bde11eba7f7acde48001122": False,

    # Q: which country is Tajlu Khanum's husband from. Gold=Safavid Empire, Pred=Iran
    # Safavid Empire was a Persian (Iranian) empire → modern-day Iran. Pred=Iran for historical Safavid → YES (commonly accepted)
    "507bc3300bde11eba7f7acde48001122": True,

    # Q: which country is director of Littlerock Film from. Gold=Filipino, Pred=United States → NO
    "d5d5034e0bdd11eba7f7acde48001122": False,

    # Q: paternal grandfather of Reginald I of Guelders. Pred=not explicitly stated → NO
    "595209340bb011ebab90acde48001122": False,

    # Q: which country is Lord Charles Arthur Francis Cavendish's wife from. Gold=American, Pred=USA
    # American (nationality) = USA (country) → YES
    "371c1d5e0bde11eba7f7acde48001122": True,

    # Q: place of death of Neferneferure's mother. Pred=not mentioned → NO
    "b66d42be0bdd11eba7f7acde48001122": False,

    # Q: where did Edward 2nd Duke of York's father die. Pred=not specified → NO
    "f67a19540bdd11eba7f7acde48001122": False,

    # Q: paternal grandfather of Engelbert III of the Mark. Pred=not provide sufficient info → NO
    "701433f00baf11ebab90acde48001122": False,

    # Q: place of burial of Ælfwine of Deira's father. Pred=not specified → NO
    "e979075e0bda11eba7f7acde48001122": False,

    # Q: date of death of Ferdinand Prince of Solms-Braunfels's father. Gold=24 February 1761, Pred=1724
    # Year only (1724) when gold has full date (1761), AND years don't match → NO
    "2e294eec0bde11eba7f7acde48001122": False,

    # Q: uncle of Eustace III Count of Boulogne. Pred=text does not specify → NO
    "676bf7000bb011ebab90acde48001122": False,

    # Q: place of burial of Ælfgar Earl of Mercia's father. Pred=does not provide info → NO
    "effa87900bdc11eba7f7acde48001122": False,

    # Q: are Frederick Elliott Brown and John Yule from same country. Gold=yes, Pred=No → NO
    "b358b348084e11ebbd56ac1f6bf848b6": False,

    # Q: place of birth of director of The Great Dome Robbery. Gold=British, Pred=text does not provide → NO
    "13fc53700bde11eba7f7acde48001122": False,

    # Q: place of birth of director of Vent d'Est. Gold=Liévin, Pred=Lievin Pas-de-Calais France
    # Pred contains gold (Liévin = Lievin, diacritic stripped) → YES
    "ecf5d7b60bdc11eba7f7acde48001122": True,

    # Q: nationality of director of Yeh Gulistan Hamara. Gold=Afghanistan, Pred=Indian → NO
    "d6e0f7f00bda11eba7f7acde48001122": False,

    # Q: where was father of Islam Shah Suri born. Gold=Sasaram, Pred=Bihar
    # Sasaram is a city IN Bihar, pred is broader/vaguer → NO
    "df13ddf40bdd11eba7f7acde48001122": False,

    # Q: did Virginia Gardner and Ramón Hernández have same nationality. Gold=yes, Pred=No → NO
    "26497355084c11ebbd56ac1f6bf848b6": False,

    # Q: nationality of director of Vortex (1976). Gold=Italy, Pred=Italian-French
    # Pred includes Italian (which maps to Italy) but also adds French → multiple, ambiguous → NO
    "f6f9fbdc0bd911eba7f7acde48001122": False,

    # Q: where did performer of Karma (Alicia Keys Song) study. Gold=Columbia, Pred=not explicitly stated → NO
    "fbf8fd9e0bda11eba7f7acde48001122": False,

    # Q: where does Sybil Eysenck's husband work. Pred=does not provide enough info → NO
    "6e6f7ec20bde11eba7f7acde48001122": False,

    # Q: where was father of Belgutei born. Gold=Khamag Mongol, Pred=Unknown → NO
    "8c7fffe40bda11eba7f7acde48001122": False,

    # Q: where was father of Marianus V of Arborea born. Gold=Sardinia, Pred=Genoa → different places → NO
    "2b9528860bde11eba7f7acde48001122": False,

    # Q: nationality of director of A Kiss On The Nose. Gold=Greek, Pred=Belgian → NO
    "b73268780bdd11eba7f7acde48001122": False,

    # Q: father of performer of Quiero Estar Contigo. Gold=Enrique Guzmán, Pred=Enrique Guzman (no accent) → same person → YES
    "28470e820bda11eba7f7acde48001122": True,

    # Q: when was director of Uncle Silas (Film) born. Pred=not provided → NO
    "af16832e0bdb11eba7f7acde48001122": False,

    # Q: nationality of Humehume's father. Gold=Kingdom of Hawaiʻi, Pred=Hawaiian
    # Hawaiian = nationality adjective for Kingdom of Hawaiʻi → YES
    "cc3587140bdd11eba7f7acde48001122": True,

    # Q: do director of Next of Kin (1982) and director of Wild Honey (1918) share same nationality. Gold=yes, Pred=No → NO
    "f418184a08be11ebbd8aac1f6bf848b6": False,

    # Q: where does Juan Carlos Gumucio's wife work. Pred=no info → NO
    "9d054e980bdd11eba7f7acde48001122": False,

    # Q: maternal grandmother of Sancho Ramírez. Gold=Garsenda Heiress of Bigorre, Pred=Garsenda
    # Same person, pred is shortened name → YES
    "8bc6a73a0bb011ebab90acde48001122": True,
}


import re
import unicodedata


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> set:
    return set(normalize(s).split())


def auto_judge(record: dict):
    pred = record.get("pred", "").strip()
    gold = record.get("gold", "").strip()
    aliases = record.get("aliases", [])

    if not pred:
        return False

    pred_n = normalize(pred)
    gold_n = normalize(gold)

    if pred_n == gold_n:
        return True

    for alias in aliases:
        if normalize(alias) == pred_n:
            return True

    yes_words = {"yes", "yeah", "yep", "true"}
    no_words = {"no", "nope", "nah", "false"}
    if gold_n in yes_words and pred_n in yes_words:
        return True
    if gold_n in no_words and pred_n in no_words:
        return True

    if gold_n and gold_n in pred_n:
        return True

    if pred_n and pred_n in gold_n:
        gold_toks = tokens(gold)
        pred_toks = tokens(pred)
        if len(pred_toks) >= len(gold_toks) * 0.8:
            return True

    def extract_year(s):
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s)
        return m.group(1) if m else None

    gold_year = extract_year(gold)
    pred_year = extract_year(pred)

    gold_is_date = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", gold.lower()))
    pred_is_year_only = bool(re.fullmatch(r"\s*\d{4}\s*", pred.strip()))
    if gold_is_date and pred_is_year_only:
        return "ambiguous"

    if gold_year and pred_year and gold_year == pred_year:
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
        }
        def get_month(s):
            sl = s.lower()
            for name, num in months.items():
                if name in sl:
                    return num
            m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}", s)
            if m:
                return m.group(1).zfill(2)
            return None

        gm, pm = get_month(gold), get_month(pred)
        if gm and pm and gm == pm:
            return True
        if not gm and not pm:
            return True

    nationality_map = {
        "italian": "italy", "french": "france", "german": "germany",
        "spanish": "spain", "british": "united kingdom", "english": "england",
        "american": "united states", "australian": "australia", "dutch": "netherlands",
        "polish": "poland", "swedish": "sweden", "norwegian": "norway",
        "danish": "denmark", "finnish": "finland", "belgian": "belgium",
        "swiss": "switzerland", "austrian": "austria", "canadian": "canada",
        "japanese": "japan", "chinese": "china", "russian": "russia",
        "greek": "greece", "portuguese": "portugal", "hungarian": "hungary",
        "czech": "czech republic", "romanian": "romania", "turkish": "turkey",
        "iranian": "iran", "indian": "india", "scottish": "scotland",
        "welsh": "wales", "irish": "ireland", "mexican": "mexico",
        "brazilian": "brazil", "argentinian": "argentina", "argentine": "argentina",
        "ukrainian": "ukraine", "serbian": "serbia", "croatian": "croatia",
        "slovak": "slovakia", "slovenian": "slovenia", "bulgarian": "bulgaria",
        "lithuanian": "lithuania", "latvian": "latvia", "estonian": "estonia",
        "new zealander": "new zealand", "south african": "south africa",
        "nigerian": "nigeria", "egyptian": "egypt", "kenyan": "kenya",
        "ghanaian": "ghana", "algerian": "algeria", "moroccan": "morocco",
        "tunisian": "tunisia", "korean": "south korea", "south korean": "south korea",
        "north korean": "north korea", "vietnamese": "vietnam",
        "thai": "thailand", "philippine": "philippines", "filipino": "philippines",
        "indonesian": "indonesia", "malaysian": "malaysia", "singaporean": "singapore",
        "pakistani": "pakistan", "bangladeshi": "bangladesh", "sri lankan": "sri lanka",
        "nepali": "nepal", "chilean": "chile", "colombian": "colombia",
        "peruvian": "peru", "venezuelan": "venezuela", "cuban": "cuba",
        "jamaican": "jamaica", "haitian": "haiti", "israeli": "israel",
        "lebanese": "lebanon", "jordanian": "jordan", "saudi arabian": "saudi arabia",
        "emirati": "united arab emirates", "kuwaiti": "kuwait",
        "qatari": "qatar", "bahraini": "bahrain",
    }
    reverse_nat = {v: k for k, v in nationality_map.items()}

    if gold_n in nationality_map and nationality_map[gold_n] == pred_n:
        return True
    if pred_n in nationality_map and nationality_map[pred_n] == gold_n:
        return True
    if gold_n in reverse_nat and reverse_nat[gold_n] == pred_n:
        return True
    if pred_n in reverse_nat and reverse_nat[pred_n] == gold_n:
        return True

    us_variants = {"united states", "united states of america", "usa", "us", "america"}
    uk_variants = {"united kingdom", "great britain", "britain", "uk"}
    if gold_n in us_variants and pred_n in us_variants:
        return True
    if gold_n in uk_variants and pred_n in uk_variants:
        return True

    return "ambiguous"


def main():
    records = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    results = {}
    for r in records:
        rid = r["id"]
        result = auto_judge(r)
        if result == "ambiguous":
            if rid in MANUAL:
                results[rid] = MANUAL[rid]
            else:
                print(f"WARNING: no manual decision for {rid} — defaulting to gemma_judge={r['gemma_judge']}")
                results[rid] = r["gemma_judge"]
        else:
            results[rid] = result

    # Write output
    with open(OUTPUT, "w") as f:
        for r in records:
            out = {"id": r["id"], "judge_correct_claude": results[r["id"]]}
            f.write(json.dumps(out) + "\n")

    total = len(records)
    correct = sum(results.values())
    print("\nResults:")
    print(f"  Total:   {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total*100:.1f}%")

    # Diff vs gemma
    gemma_correct = sum(1 for r in records if r["gemma_judge"])
    agree = sum(1 for r in records if results[r["id"]] == r["gemma_judge"])
    print(f"\nGemma judge: {gemma_correct}/{total} ({gemma_correct/total*100:.1f}%)")
    print(f"Agreement with gemma: {agree}/{total} ({agree/total*100:.1f}%)")

    # Disagreements
    disagree = [r for r in records if results[r["id"]] != r["gemma_judge"]]
    print(f"\nDisagreements ({len(disagree)}):")
    for r in disagree:
        print(f"  Claude={results[r['id']]} Gemma={r['gemma_judge']} | Gold: {r['gold']} | Pred: {r['pred']}")


if __name__ == "__main__":
    main()
