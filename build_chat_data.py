"""
Generate the chatbot's training data: a large, heavily-paraphrased set of
(question, answer) pairs written to data/chat.jsonl.

The whole point of this file is **paraphrase robustness**. A tiny from-scratch
model has no outside knowledge, so it can only answer what it is trained on --
but we want it to answer the same question no matter how it's worded ("who was
the first president of the USA" == "who was the 1st president of the United
States"). The way to teach that is to show it many wordings of every fact.

So instead of writing questions by hand, we describe each fact with a *slot
template* plus lists of interchangeable phrasings:

    lead-ins:   who was / who is / name / tell me / do you know ...
    ordinals:   first / 1st,   second / 2nd ...
    countries:  the usa / the united states / the us / america ...

and the generator expands the combinations (capped per fact) into many
paraphrases that all map to the same answer. Extend the knowledge base below and
re-run:

    python build_chat_data.py
    python train_chat.py
"""

import itertools
import json
import os
import random
import re

RNG = random.Random(1234)  # deterministic dataset

# ----------------------------------------------------------------------------
# Reusable slot vocabularies (the axes of paraphrase variation)
# ----------------------------------------------------------------------------
WHO = ["who was", "who is", "who's", "name", "tell me", "do you know", "can you tell me", "i want to know"]
WHAT = ["what is", "what's", "tell me", "do you know", "can you tell me", "name"]
WHICH = ["what is", "which is", "what's", "tell me", "name", "do you know"]
HOWMANY = ["how many", "tell me how many", "do you know how many"]
Q = ["?", "?", "?", ""]  # mostly with a question mark

ORDINALS = {
    1: ["first", "1st"], 2: ["second", "2nd"], 3: ["third", "3rd"], 4: ["fourth", "4th"],
    5: ["fifth", "5th"], 6: ["sixth", "6th"], 7: ["seventh", "7th"], 8: ["eighth", "8th"],
    16: ["sixteenth", "16th"], 32: ["thirty second", "32nd"], 35: ["thirty fifth", "35th"],
}

CAP_PER_FACT = 14  # max paraphrases generated per fact

pairs = []  # accumulates (question, answer)


def _clean(q):
    q = re.sub(r"\s+", " ", q).strip()
    q = q.replace(" ?", "?")
    return q


def fact(answer, template, slots, cap=CAP_PER_FACT):
    """Expand a slot template into up to `cap` paraphrased questions, each mapped
    to the same answer."""
    keys = list(slots.keys())
    all_combos = list(itertools.product(*[slots[k] for k in keys]))
    RNG.shuffle(all_combos)
    seen = set()
    n = 0
    for combo in all_combos:
        q = template
        for k, v in zip(keys, combo):
            q = q.replace("{" + k + "}", v)
        q = _clean(q)
        if q and q not in seen:
            seen.add(q)
            pairs.append((q, answer))
            n += 1
        if n >= cap:
            break


def simple(answer, questions):
    """Attach a fixed answer to an explicit list of question phrasings."""
    for q in questions:
        pairs.append((_clean(q), answer))


# ----------------------------------------------------------------------------
# Arithmetic (expanded range + several phrasings)
# ----------------------------------------------------------------------------
def arithmetic():
    # Addition / subtraction over 0..20; multiplication over 0..12 (times tables)
    # -- large products are genuinely hard for a model this small to memorize.
    ADD_SUB = range(0, 21)
    MUL = range(0, 13)
    for a in ADD_SUB:
        for b in ADD_SUB:
            s = a + b
            ans = f"{a} plus {b} equals {s}."
            simple(ans, [f"what is {a} plus {b}?", f"what is {a} + {b}?",
                         f"what's {a} plus {b}?", f"{a} plus {b}", f"add {a} and {b}",
                         f"calculate {a} + {b}", f"how much is {a} plus {b}?",
                         f"how much is {a} + {b}?", f"what is the sum of {a} and {b}?"])
            if a >= b:
                d = a - b
                ans = f"{a} minus {b} equals {d}."
                simple(ans, [f"what is {a} minus {b}?", f"what is {a} - {b}?",
                             f"{a} minus {b}", f"subtract {b} from {a}",
                             f"how much is {a} minus {b}?", f"{a} take away {b}"])
    for a in MUL:
        for b in MUL:
            p = a * b
            ans = f"{a} times {b} equals {p}."
            simple(ans, [f"what is {a} times {b}?", f"what is {a} * {b}?",
                         f"{a} times {b}", f"multiply {a} and {b}", f"{a} x {b}",
                         f"how much is {a} times {b}?", f"what is {a} multiplied by {b}?"])


# ----------------------------------------------------------------------------
# Countries: capital, currency, language, continent
# ----------------------------------------------------------------------------
# name -> (capital, currency, language, continent, [name synonyms])
COUNTRIES = {
    "france": ("Paris", "the euro", "French", "Europe", ["france"]),
    "japan": ("Tokyo", "the yen", "Japanese", "Asia", ["japan"]),
    "italy": ("Rome", "the euro", "Italian", "Europe", ["italy"]),
    "spain": ("Madrid", "the euro", "Spanish", "Europe", ["spain"]),
    "germany": ("Berlin", "the euro", "German", "Europe", ["germany"]),
    "united kingdom": ("London", "the pound", "English", "Europe",
                        ["the united kingdom", "the uk", "britain", "great britain", "england"]),
    "russia": ("Moscow", "the ruble", "Russian", "Europe", ["russia"]),
    "china": ("Beijing", "the yuan", "Chinese", "Asia", ["china"]),
    "canada": ("Ottawa", "the canadian dollar", "English", "North America", ["canada"]),
    "egypt": ("Cairo", "the egyptian pound", "Arabic", "Africa", ["egypt"]),
    "brazil": ("Brasilia", "the real", "Portuguese", "South America", ["brazil"]),
    "india": ("New Delhi", "the rupee", "Hindi", "Asia", ["india"]),
    "australia": ("Canberra", "the australian dollar", "English", "Oceania", ["australia"]),
    "mexico": ("Mexico City", "the peso", "Spanish", "North America", ["mexico"]),
    "greece": ("Athens", "the euro", "Greek", "Europe", ["greece"]),
    "portugal": ("Lisbon", "the euro", "Portuguese", "Europe", ["portugal"]),
    "netherlands": ("Amsterdam", "the euro", "Dutch", "Europe", ["the netherlands", "holland"]),
    "sweden": ("Stockholm", "the krona", "Swedish", "Europe", ["sweden"]),
    "norway": ("Oslo", "the krone", "Norwegian", "Europe", ["norway"]),
    "ireland": ("Dublin", "the euro", "English", "Europe", ["ireland"]),
    "united states": ("Washington D.C.", "the dollar", "English", "North America",
                      ["the united states", "the usa", "the us", "america", "the united states of america"]),
    "argentina": ("Buenos Aires", "the peso", "Spanish", "South America", ["argentina"]),
    "turkey": ("Ankara", "the lira", "Turkish", "Asia", ["turkey"]),
    "poland": ("Warsaw", "the zloty", "Polish", "Europe", ["poland"]),
    "austria": ("Vienna", "the euro", "German", "Europe", ["austria"]),
    "switzerland": ("Bern", "the swiss franc", "German", "Europe", ["switzerland"]),
    "belgium": ("Brussels", "the euro", "Dutch", "Europe", ["belgium"]),
    "denmark": ("Copenhagen", "the krone", "Danish", "Europe", ["denmark"]),
    "finland": ("Helsinki", "the euro", "Finnish", "Europe", ["finland"]),
    "thailand": ("Bangkok", "the baht", "Thai", "Asia", ["thailand"]),
    "vietnam": ("Hanoi", "the dong", "Vietnamese", "Asia", ["vietnam"]),
    "south korea": ("Seoul", "the won", "Korean", "Asia", ["south korea"]),
    "saudi arabia": ("Riyadh", "the riyal", "Arabic", "Asia", ["saudi arabia"]),
    "kenya": ("Nairobi", "the shilling", "Swahili", "Africa", ["kenya"]),
    "nigeria": ("Abuja", "the naira", "English", "Africa", ["nigeria"]),
    "south africa": ("Pretoria", "the rand", "English", "Africa", ["south africa"]),
    "morocco": ("Rabat", "the dirham", "Arabic", "Africa", ["morocco"]),
    "peru": ("Lima", "the sol", "Spanish", "South America", ["peru"]),
    "chile": ("Santiago", "the peso", "Spanish", "South America", ["chile"]),
    "new zealand": ("Wellington", "the new zealand dollar", "English", "Oceania", ["new zealand"]),
}


def countries():
    for name, (cap, cur, lang, cont, syns) in COUNTRIES.items():
        disp = name.title()
        fact(f"The capital of {disp} is {cap}.",
             "{lead} the capital {of} {c}{q}",
             {"lead": WHAT, "of": ["of", "city of"], "c": syns, "q": Q})
        fact(f"The capital of {disp} is {cap}.",
             "{lead} {c}'s capital{q}", {"lead": WHAT, "c": syns, "q": Q})
        simple(f"The capital of {disp} is {cap}.", [f"capital of {s}" for s in syns])
        fact(f"The currency of {disp} is {cur}.",
             "{lead} the currency {of} {c}{q}",
             {"lead": WHAT, "of": ["of", "used in"], "c": syns, "q": Q})
        fact(f"The currency of {disp} is {cur}.",
             "{lead} {c}'s currency{q}", {"lead": WHAT, "c": syns, "q": Q})
        fact(f"The main language spoken in {disp} is {lang}.",
             "{lead} language {verb} in {c}{q}",
             {"lead": ["what", "which"], "verb": ["do they speak", "do people speak", "is spoken"],
              "c": syns, "q": Q})
        fact(f"The main language spoken in {disp} is {lang}.",
             "{lead} the language {of} {c}{q}",
             {"lead": WHAT, "of": ["of", "spoken in"], "c": syns, "q": Q})
        fact(f"{disp} is in {cont}.",
             "{lead} continent is {c} {inn}{q}",
             {"lead": WHICH, "c": syns, "inn": ["in", "on", "located in"], "q": Q})


# ----------------------------------------------------------------------------
# Ordinal facts (presidents, planets) -- the paraphrase axis the user asked for
# ----------------------------------------------------------------------------
PRESIDENTS = {
    1: "George Washington", 2: "John Adams", 3: "Thomas Jefferson",
    4: "James Madison", 16: "Abraham Lincoln", 32: "Franklin Roosevelt",
    35: "John F. Kennedy",
}
USA = ["the usa", "the united states", "the us", "america", "the united states of america"]


def presidents():
    for n, who in PRESIDENTS.items():
        fact(f"The {ORDINALS[n][0]} president of the United States was {who}.",
             "{lead} the {ord} president of {c}{q}",
             {"lead": WHO, "ord": ORDINALS[n], "c": USA, "q": Q})


PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]


def planets():
    for i, planet in enumerate(PLANETS, start=1):
        fact(f"The {ORDINALS[i][0]} planet from the sun is {planet}.",
             "{lead} the {ord} planet from the sun{q}",
             {"lead": WHICH, "ord": ORDINALS[i], "q": Q})


# ----------------------------------------------------------------------------
# Chemical elements (name <-> symbol, atomic number) for the first 20
# ----------------------------------------------------------------------------
ELEMENTS = [
    ("hydrogen", "H", 1), ("helium", "He", 2), ("lithium", "Li", 3), ("beryllium", "Be", 4),
    ("boron", "B", 5), ("carbon", "C", 6), ("nitrogen", "N", 7), ("oxygen", "O", 8),
    ("fluorine", "F", 9), ("neon", "Ne", 10), ("sodium", "Na", 11), ("magnesium", "Mg", 12),
    ("aluminium", "Al", 13), ("silicon", "Si", 14), ("phosphorus", "P", 15), ("sulfur", "S", 16),
    ("chlorine", "Cl", 17), ("argon", "Ar", 18), ("potassium", "K", 19), ("calcium", "Ca", 20),
]


def elements():
    for name, sym, num in ELEMENTS:
        fact(f"The chemical symbol for {name} is {sym}.",
             "{lead} the chemical symbol for {n}{q}",
             {"lead": WHAT, "n": [name], "q": Q})
        fact(f"The chemical symbol for {name} is {sym}.",
             "{lead} {n}'s symbol{q}", {"lead": WHAT, "n": [name], "q": Q})


# ----------------------------------------------------------------------------
# General knowledge, science, geography, etc. (explicit paraphrase lists)
# ----------------------------------------------------------------------------
def general():
    simple("The sky is blue.", ["what color is the sky?", "what colour is the sky?",
                                 "what color is the sky", "tell me the color of the sky"])
    simple("Grass is green.", ["what color is grass?", "what colour is grass?", "what color is grass"])
    simple("The sun is yellow.", ["what color is the sun?", "what color is the sun"])
    simple("Snow is white.", ["what color is snow?", "what color is snow"])
    simple("There are seven days in a week.",
           ["how many days are in a week?", "how many days in a week", "days in a week",
            "how many days does a week have?"])
    simple("There are twelve months in a year.",
           ["how many months are in a year?", "months in a year", "how many months in a year"])
    simple("There are three hundred and sixty five days in a year.",
           ["how many days are in a year?", "days in a year", "how many days in a year"])
    simple("There are twenty four hours in a day.",
           ["how many hours are in a day?", "hours in a day", "how many hours in a day"])
    simple("There are sixty minutes in an hour.",
           ["how many minutes are in an hour?", "minutes in an hour"])
    simple("There are sixty seconds in a minute.",
           ["how many seconds are in a minute?", "seconds in a minute"])
    simple("There are eight planets in the solar system.",
           ["how many planets are there?", "how many planets are in the solar system?",
            "number of planets", "how many planets are in our solar system"])
    simple("The largest planet is Jupiter.",
           ["what is the largest planet?", "what is the biggest planet?", "biggest planet"])
    simple("The smallest planet is Mercury.",
           ["what is the smallest planet?", "smallest planet"])
    simple("The closest planet to the sun is Mercury.",
           ["what is the closest planet to the sun?", "closest planet to the sun",
            "what is the nearest planet to the sun?", "which planet is nearest to the sun?",
            "which planet is closest to the sun?"])
    simple("The largest ocean is the Pacific Ocean.",
           ["what is the largest ocean?", "biggest ocean", "what is the biggest ocean?"])
    simple("The tallest mountain is Mount Everest.",
           ["what is the tallest mountain?", "highest mountain", "what is the highest mountain?",
            "what is the tallest mountain in the world?", "what's the tallest mountain"])
    simple("The longest river is the Nile.",
           ["what is the longest river?", "longest river in the world", "what is the longest river in the world?"])
    simple("The largest desert is the Sahara.",
           ["what is the largest desert?", "biggest desert"])
    simple("The largest country by area is Russia.",
           ["what is the largest country?", "biggest country", "what is the biggest country in the world?"])
    simple("Water is made of hydrogen and oxygen.",
           ["what is water made of?", "what is water made from?", "what are the elements in water?"])
    simple("The chemical symbol for water is H2O.",
           ["what is the chemical symbol for water?", "chemical formula for water"])
    simple("Plants produce oxygen.",
           ["what gas do plants produce?", "what do plants produce?", "what do plants give off?"])
    simple("Plants make food through photosynthesis.",
           ["how do plants make food?", "what is the process plants use to make food?"])
    simple("Bees make honey.", ["what do bees make?", "what do bees produce?"])
    simple("The fastest land animal is the cheetah.",
           ["what is the fastest land animal?", "fastest animal on land", "what is the fastest animal?"])
    simple("The largest animal is the blue whale.",
           ["what is the largest animal?", "biggest animal", "what is the biggest animal in the world?"])
    simple("A spider has eight legs.", ["how many legs does a spider have?", "spider legs"])
    simple("An insect has six legs.", ["how many legs does an insect have?", "insect legs"])
    simple("Water freezes at zero degrees celsius.",
           ["what is the freezing point of water?", "at what temperature does water freeze?"])
    simple("Water boils at one hundred degrees celsius.",
           ["what is the boiling point of water?", "at what temperature does water boil?"])
    simple("There are seven continents.",
           ["how many continents are there?", "number of continents", "how many continents are there in the world?"])
    simple("The speed of light is about three hundred thousand kilometers per second.",
           ["what is the speed of light?", "how fast is light?"])
    simple("Sound travels slower than light.", ["what travels faster, sound or light?"])
    simple("The human body has two hundred and six bones.",
           ["how many bones are in the human body?", "how many bones does a human have?", "number of bones in the body"])
    simple("The human heart has four chambers.",
           ["how many chambers does the heart have?", "how many chambers are in the human heart?"])
    simple("The largest organ in the human body is the skin.",
           ["what is the largest organ?", "what is the biggest organ in the body?"])
    simple("Humans breathe in oxygen and breathe out carbon dioxide.",
           ["what do humans breathe?", "what gas do humans breathe out?"])
    simple("The powerhouse of the cell is the mitochondria.",
           ["what is the powerhouse of the cell?", "what part of the cell makes energy?"])
    simple("DNA carries genetic information.", ["what does dna do?", "what is dna for?"])
    simple("World War Two ended in 1945.",
           ["when did world war two end?", "when did ww2 end?", "what year did world war 2 end?"])
    simple("The first man landed on the moon in 1969.",
           ["when did humans land on the moon?", "what year did we land on the moon?",
            "when was the moon landing?"])
    simple("Romeo and Juliet was written by William Shakespeare.",
           ["who wrote romeo and juliet?", "who is the author of romeo and juliet?"])
    simple("Hamlet was written by William Shakespeare.",
           ["who wrote hamlet?", "who is the author of hamlet?"])
    simple("The theory of relativity was developed by Albert Einstein.",
           ["who developed the theory of relativity?", "who came up with relativity?"])
    simple("The telephone was invented by Alexander Graham Bell.",
           ["who invented the telephone?", "who is the inventor of the telephone?"])
    simple("The light bulb was invented by Thomas Edison.",
           ["who invented the light bulb?", "who made the light bulb?"])
    simple("A triangle has three sides.", ["how many sides does a triangle have?", "what shape has three sides?"])
    simple("A square has four sides.", ["how many sides does a square have?", "what shape has four sides?"])
    simple("A pentagon has five sides.", ["how many sides does a pentagon have?", "what shape has five sides?"])
    simple("A hexagon has six sides.", ["how many sides does a hexagon have?", "what shape has six sides?"])
    simple("The first day of the week is Sunday.", ["what is the first day of the week?"])
    simple("The smallest prime number is two.", ["what is the smallest prime number?"])
    simple("The value of pi is about 3.14.", ["what is pi?", "what is the value of pi?"])
    # opposites
    for a, b in [("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"),
                 ("day", "night"), ("left", "right"), ("happy", "sad"), ("open", "closed"),
                 ("black", "white"), ("wet", "dry")]:
        simple(f"The opposite of {a} is {b}.",
               [f"what is the opposite of {a}?", f"opposite of {a}", f"what's the opposite of {a}?"])


def small_talk_and_meta():
    simple("Hello! Ask me a question.",
           ["hello", "hi", "hey", "hi there", "hello there", "hey there", "yo", "greetings"])
    simple("Hello! How can I help?", ["good morning", "good afternoon", "good evening"])
    simple("I'm just a small program, but I'm running fine. What would you like to ask?",
           ["how are you", "how are you doing", "how's it going", "how do you feel"])
    simple("You're welcome!", ["thanks", "thank you", "thanks a lot", "thank you so much", "cheers"])
    simple("Goodbye!", ["bye", "goodbye", "see you", "see you later", "farewell"])
    simple("Glad that helped. Ask me something else!", ["ok", "okay", "cool", "nice", "great"])
    simple("I am a tiny GPT language model, built from scratch in PyTorch.",
           ["what is your name?", "what's your name?", "who are you?", "what are you?",
            "what are you exactly?", "tell me about yourself"])
    simple("I was built from scratch as an educational project.",
           ["who made you?", "who created you?", "who built you?", "who is your creator?"])
    simple("I predict one word at a time using a small transformer neural network.",
           ["how do you work?", "how were you trained?", "how do you generate text?"])
    simple("No. I am a very small model trained on a small dataset, so I only know a few things.",
           ["are you chatgpt?", "are you gpt?", "are you a real ai?", "are you smart?", "are you human?"])
    simple("I can do arithmetic and answer general knowledge questions about geography, science, and history.",
           ["what can you do?", "what do you know?", "help", "what can i ask?", "what can i ask you?"])
    simple("I am tiny, with only a few million parameters.",
           ["how many parameters do you have?", "how big are you?"])


# ----------------------------------------------------------------------------
# Sports -- a deep, paraphrased knowledge base across many sports
# ----------------------------------------------------------------------------
# In this bot, "football" means American football (the user's framing); the
# global game is "soccer". name -> dict of stable, objective facts.
SPORTS = {
    "soccer": {
        "aka": ["soccer", "association football"],
        "players": "eleven", "champ": "the World Cup", "league": "FIFA",
        "gov": "FIFA", "score": "a goal", "field": "a pitch",
        "duration": "ninety minutes", "halves": "two",
    },
    "basketball": {
        "aka": ["basketball"],
        "players": "five", "champ": "the NBA Finals", "league": "the NBA",
        "gov": "FIBA", "score": "a basket", "field": "a court",
        "duration": "forty eight minutes", "quarters": "four",
    },
    "american football": {
        "aka": ["american football", "football"],
        "players": "eleven", "champ": "the Super Bowl", "league": "the NFL",
        "gov": "the NFL", "score": "a touchdown", "field": "a field",
        "duration": "sixty minutes", "quarters": "four",
    },
    "baseball": {
        "aka": ["baseball"],
        "players": "nine", "champ": "the World Series", "league": "MLB",
        "gov": "MLB", "score": "a run", "field": "a diamond", "innings": "nine",
    },
    "ice hockey": {
        "aka": ["ice hockey", "hockey"],
        "players": "six", "champ": "the Stanley Cup", "league": "the NHL",
        "gov": "the NHL", "score": "a goal", "field": "a rink", "periods": "three",
    },
    "tennis": {
        "aka": ["tennis"],
        "players": "one", "champ": "a Grand Slam", "gov": "the ITF",
        "score": "a point", "field": "a court",
    },
    "cricket": {
        "aka": ["cricket"],
        "players": "eleven", "gov": "the ICC", "score": "a run", "field": "a pitch",
    },
    "rugby": {
        "aka": ["rugby", "rugby union"],
        "players": "fifteen", "gov": "World Rugby", "score": "a try", "field": "a pitch",
    },
    "volleyball": {
        "aka": ["volleyball"],
        "players": "six", "gov": "the FIVB", "score": "a point", "field": "a court",
    },
    "golf": {
        "aka": ["golf"],
        "players": "one", "gov": "the PGA", "field": "a course",
    },
}


def sports():
    for name, d in SPORTS.items():
        disp = name
        aka = d["aka"]
        if "players" in d:
            fact(f"A {disp} team has {d['players']} players.",
                 "{lead} players are {on} a {s} team{q}",
                 {"lead": HOWMANY, "on": ["on", "in"], "s": aka, "q": Q})
            fact(f"A {disp} team has {d['players']} players.",
                 "{lead} players {play} {s}{q}",
                 {"lead": HOWMANY, "play": ["play", "are in", "are on a team in"], "s": aka, "q": Q})
        if "champ" in d:
            fact(f"The {disp} championship is called {d['champ']}.",
                 "{lead} the {s} championship called{q}",
                 {"lead": ["what is", "what's"], "s": aka, "q": Q})
            fact(f"The {disp} championship is called {d['champ']}.",
                 "{lead} the {s} championship{q}",
                 {"lead": ["what do they call", "name"], "s": aka, "q": Q})
            simple(f"The {disp} championship is called {d['champ']}.",
                   [f"what is the biggest {a} tournament?" for a in aka])
        if "gov" in d:
            fact(f"The governing body of {disp} is {d['gov']}.",
                 "{lead} the governing body of {s}{q}",
                 {"lead": ["what is", "what's", "name", "tell me"], "s": aka, "q": Q})
        if "score" in d:
            fact(f"In {disp} you score {d['score']}.",
                 "{lead} do you score in {s}{q}",
                 {"lead": ["what", "how"], "s": aka, "q": Q})
        if "field" in d:
            fact(f"{disp.capitalize()} is played on {d['field']}.",
                 "{lead} is {s} played on{q}",
                 {"lead": ["what", "where"], "s": aka, "q": Q})

    # Reverse "which sport ..." questions and other stable facts.
    simple("A touchdown is worth six points.",
           ["how many points is a touchdown?", "how much is a touchdown worth?",
            "what is a touchdown worth?", "points for a touchdown"])
    simple("A field goal in american football is worth three points.",
           ["how many points is a field goal?", "what is a field goal worth in football?"])
    simple("A basket, or field goal, in basketball is worth two points.",
           ["how many points is a basket?", "how much is a field goal worth in basketball?"])
    simple("A three pointer in basketball is worth three points.",
           ["how many points is a three pointer?", "what is a three pointer worth?"])
    simple("A free throw in basketball is worth one point.",
           ["how many points is a free throw?", "what is a free throw worth?"])
    simple("A try in rugby is worth five points.",
           ["how many points is a try?", "what is a try worth in rugby?"])
    simple("Baseball has nine innings.",
           ["how many innings are in baseball?", "how many innings in a baseball game?",
            "number of innings in baseball"])
    simple("A baseball batter is out after three strikes.",
           ["how many strikes before you are out?", "how many strikes is an out?"])
    simple("A basketball hoop is ten feet high.",
           ["how high is a basketball hoop?", "how tall is a basketball hoop?"])
    simple("A game of golf has eighteen holes.",
           ["how many holes are in golf?", "how many holes in a round of golf?", "holes in golf"])
    simple("A hole in one is when you sink the ball in a single shot in golf.",
           ["what is a hole in one?", "what does hole in one mean?"])
    simple("A hat trick is scoring three goals in one game.",
           ["what is a hat trick?", "what does a hat trick mean?", "how many goals is a hat trick?"])
    simple("A marathon is about twenty six miles long.",
           ["how long is a marathon?", "how many miles is a marathon?", "marathon distance"])
    simple("The Olympics are held every four years.",
           ["how often are the olympics held?", "how often are the olympics?", "how often do the olympics happen?"])
    simple("There are five rings on the Olympic flag.",
           ["how many rings are on the olympic flag?", "how many olympic rings are there?"])
    simple("The first modern Olympic Games were held in Athens in 1896.",
           ["where were the first modern olympics held?", "when were the first modern olympics?"])
    simple("Wimbledon is a tennis tournament.",
           ["what sport is played at wimbledon?", "what is wimbledon?", "which sport is wimbledon?"])
    simple("The Masters is a golf tournament.",
           ["what sport is the masters?", "what is the masters?"])
    simple("The Tour de France is a cycling race.",
           ["what sport is the tour de france?", "what is the tour de france?"])
    simple("The sport played at Wimbledon on grass courts is tennis.",
           ["what sport is played on grass at wimbledon?"])
    simple("Tennis is the sport where a score of zero is called love.",
           ["in which sport is zero called love?", "what sport uses the term love?",
            "what does love mean in tennis?"])
    simple("In tennis, love means a score of zero.",
           ["what is love in tennis?", "what does love mean in tennis?"])
    simple("The sport that uses a shuttlecock is badminton.",
           ["what sport uses a shuttlecock?", "which sport has a shuttlecock?"])
    simple("The sport played on ice with a puck is ice hockey.",
           ["what sport uses a puck?", "which sport is played with a puck?"])
    simple("The sport where you score a home run is baseball.",
           ["in which sport do you score a home run?", "what sport has a home run?"])
    simple("The sport where you score a touchdown is american football.",
           ["in which sport do you score a touchdown?", "what sport has a touchdown?"])
    simple("The sport where you score a slam dunk is basketball.",
           ["in which sport is there a slam dunk?", "what sport has a slam dunk?"])
    simple("Soccer is the most popular sport in the world.",
           ["what is the most popular sport in the world?", "what is the most popular sport?"])
    simple("The soccer World Cup is held every four years.",
           ["how often is the world cup?", "how often is the soccer world cup held?"])
    simple("Boxing takes place in a ring.",
           ["where does boxing take place?", "what does boxing happen in?"])
    simple("A soccer match is ninety minutes long.",
           ["how long is a soccer game?", "how long is a soccer match?", "how many minutes in soccer?"])
    simple("A soccer match has two halves.",
           ["how many halves are in soccer?", "how many halves in a soccer match?"])
    simple("A basketball game has four quarters.",
           ["how many quarters are in basketball?", "how many quarters in a basketball game?"])
    simple("An american football game has four quarters.",
           ["how many quarters are in american football?", "how many quarters in a football game?"])
    simple("An ice hockey game has three periods.",
           ["how many periods are in hockey?", "how many periods in an ice hockey game?"])
    simple("The four tennis Grand Slams are the Australian Open, the French Open, Wimbledon, and the US Open.",
           ["what are the tennis grand slams?", "name the four grand slams", "what are the four majors in tennis?"])
    simple("Bowling has ten pins.",
           ["how many pins are in bowling?", "how many bowling pins are there?"])
    simple("A perfect game in bowling is a score of three hundred.",
           ["what is a perfect score in bowling?", "what is a perfect game in bowling?"])
    # what IS <event/league> -- reverse definitions for natural questions
    simple("The Super Bowl is the championship game of American football's NFL.",
           ["what is the super bowl?", "what's the super bowl?", "tell me about the super bowl",
            "what is the superbowl?"])
    simple("The World Series is the championship of Major League Baseball.",
           ["what is the world series?", "what's the world series?", "tell me about the world series"])
    simple("The Stanley Cup is the championship trophy of the NHL in ice hockey.",
           ["what is the stanley cup?", "what's the stanley cup?"])
    simple("The NBA Finals is the championship series of basketball's NBA.",
           ["what is the nba finals?", "what are the nba finals?"])
    simple("The World Cup is the biggest tournament in soccer, held every four years.",
           ["what is the world cup?", "what's the world cup?", "tell me about the world cup"])
    simple("The NBA is the top professional basketball league in North America.",
           ["what is the nba?", "what's the nba?"])
    simple("The NFL is the top professional american football league.",
           ["what is the nfl?", "what's the nfl?"])
    simple("MLB, or Major League Baseball, is the top professional baseball league.",
           ["what is mlb?", "what is major league baseball?"])
    simple("The NHL is the top professional ice hockey league in North America.",
           ["what is the nhl?", "what's the nhl?"])
    simple("The Premier League is the top soccer league in England.",
           ["what is the premier league?", "what's the premier league?"])
    simple("The Olympics is a global sporting event held every four years.",
           ["what are the olympics?", "what is the olympics?", "tell me about the olympics"])
    # --- more rules and terms ---
    simple("There are four downs to make a first down in american football.",
           ["how many downs are in american football?", "how many downs in football?", "how many downs do you get?"])
    simple("A first down requires gaining ten yards in american football.",
           ["how many yards for a first down?", "how many yards is a first down?"])
    simple("An american football field is one hundred yards long.",
           ["how long is a football field?", "how many yards is a football field?"])
    simple("A safety in american football is worth two points.",
           ["how many points is a safety?", "what is a safety worth?"])
    simple("There are three outs in each half of a baseball inning.",
           ["how many outs are in an inning?", "how many outs per inning?", "how many outs in baseball?"])
    simple("It takes four balls to draw a walk in baseball.",
           ["how many balls is a walk?", "how many balls for a walk?"])
    simple("A grand slam in baseball is a home run with the bases loaded, scoring four runs.",
           ["what is a grand slam in baseball?", "what is a grand slam?"])
    simple("There are four bases in baseball.",
           ["how many bases are in baseball?", "how many bases in baseball?"])
    simple("A red card sends a player off in soccer.",
           ["what does a red card mean?", "what is a red card?", "which card sends a player off?"])
    simple("A yellow card is a caution in soccer.",
           ["what does a yellow card mean?", "what is a yellow card?"])
    simple("A clean sheet in soccer is when a team concedes no goals.",
           ["what is a clean sheet?", "what does a clean sheet mean?"])
    simple("A penalty kick in soccer is taken from the penalty spot.",
           ["what is a penalty kick?", "where is a penalty kick taken from?"])
    simple("In golf, a birdie is one stroke under par.",
           ["what is a birdie in golf?", "what does birdie mean in golf?"])
    simple("In golf, an eagle is two strokes under par.",
           ["what is an eagle in golf?", "what does eagle mean in golf?"])
    simple("In golf, a bogey is one stroke over par.",
           ["what is a bogey in golf?", "what does bogey mean?"])
    simple("Par is the expected number of strokes for a hole in golf.",
           ["what is par in golf?", "what does par mean in golf?"])
    simple("In tennis, deuce means the score is forty forty.",
           ["what is deuce in tennis?", "what does deuce mean?"])
    simple("A conversion in rugby is worth two points.",
           ["how many points is a conversion?", "what is a conversion worth in rugby?"])
    simple("There are six balls in an over in cricket.",
           ["how many balls are in an over?", "how many balls in a cricket over?"])
    simple("A century in cricket is a score of one hundred runs.",
           ["what is a century in cricket?", "what does a century mean in cricket?"])
    simple("A player fouls out of an NBA game after six personal fouls.",
           ["how many fouls to foul out in the nba?", "how many fouls before you foul out?"])
    # --- more competitions and events ---
    simple("The Champions League is the top club soccer competition in Europe.",
           ["what is the champions league?", "what's the champions league?"])
    simple("The Ashes is a cricket series between England and Australia.",
           ["what is the ashes?", "what's the ashes?"])
    simple("The Ryder Cup is a golf competition between the United States and Europe.",
           ["what is the ryder cup?", "what's the ryder cup?"])
    simple("The Kentucky Derby is a horse racing event.",
           ["what sport is the kentucky derby?", "what is the kentucky derby?"])
    simple("The Indianapolis 500 is a motor racing event.",
           ["what sport is the indy 500?", "what is the indianapolis 500?", "what is the indy 500?"])
    simple("Formula One is a motor racing competition.",
           ["what sport is formula one?", "what is formula one?", "what is f1?"])
    simple("The US Open is one of the four tennis Grand Slams.",
           ["what is the us open?", "what sport is the us open?"])
    # --- more sports ---
    simple("Badminton is played with a racket and a shuttlecock.",
           ["what is badminton played with?", "what equipment is used in badminton?"])
    simple("Table tennis is played on a table with small paddles and a light ball.",
           ["what is table tennis?", "how is table tennis played?", "what is ping pong?"])
    simple("Boxing matches are divided into rounds.",
           ["how is a boxing match divided?", "what are boxing matches divided into?"])
    simple("The four main swimming strokes are freestyle, backstroke, breaststroke, and butterfly.",
           ["what are the swimming strokes?", "name the swimming strokes", "what are the four swimming strokes?"])
    simple("A sprint in athletics is a short, fast running race.",
           ["what is a sprint?", "what does sprint mean in athletics?"])
    # --- a few stable athlete records ---
    simple("Michael Phelps has won the most Olympic gold medals.",
           ["who has won the most olympic gold medals?", "who has the most olympic golds?",
            "who won the most olympic gold medals?"])
    simple("Usain Bolt holds the world record for the one hundred meter sprint.",
           ["who holds the 100 meter world record?", "who is the fastest man in the world?",
            "who holds the 100m record?"])


def main():
    arithmetic()
    countries()
    presidents()
    planets()
    elements()
    general()
    sports()
    small_talk_and_meta()

    # Deduplicate, keeping the first answer seen for a given question.
    seen_q = {}
    unique = []
    for q, a in pairs:
        if q in seen_q:
            continue
        seen_q[q] = a
        unique.append((q, a))

    os.makedirs("data", exist_ok=True)
    out_path = "data/chat.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in unique:
            f.write(json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n")

    n_arith = sum(1 for _, a in unique if "equals" in a)
    print(f"Wrote {len(unique)} unique (question, answer) pairs to {out_path}")
    print(f"  arithmetic: {n_arith} | knowledge + small talk: {len(unique) - n_arith}")
    print(f"  distinct answers: {len(set(a for _, a in unique))}")


if __name__ == "__main__":
    main()
