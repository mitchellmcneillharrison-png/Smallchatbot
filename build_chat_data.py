"""
Generate the SPORTS chatbot's training data: a large, heavily-paraphrased set of
(question, answer) pairs written to data/chat.jsonl.

This bot is a **sports** model. Apart from a little basic knowledge (arithmetic,
greetings, who it is, a few everyday facts), everything it knows is sports:
hundreds of athletes, 100+ teams, leagues, championships, rules, records, and
terminology -- across soccer, American football, basketball, baseball, ice
hockey, tennis, golf, boxing, cricket, Formula One, and more.

The core trick is **paraphrase robustness**: each fact is described with a slot
template plus interchangeable phrasings, and the generator expands the
combinations (capped per fact) so the model answers the same no matter how a
question is worded. Extend the tables below and re-run:

    python build_chat_data.py
    python train_chat.py
"""

import itertools
import json
import os
import random
import re

RNG = random.Random(1234)  # deterministic dataset

# ---- reusable slot vocabularies (the axes of paraphrase variation) ---------
WHO = ["who is", "who's", "who was", "tell me about", "do you know who"]
WHAT = ["what is", "what's", "tell me", "do you know", "can you tell me", "name"]
WHICH = ["what is", "which is", "what's", "tell me", "name", "do you know"]
HOWMANY = ["how many", "tell me how many", "do you know how many"]
Q = ["?", "?", "?", ""]  # mostly with a question mark

CAP = 10  # max paraphrases generated per templated fact

pairs = []  # accumulates (question, answer)


def _clean(q):
    q = re.sub(r"\s+", " ", q).strip()
    return q.replace(" ?", "?")


def fact(answer, template, slots, cap=CAP):
    keys = list(slots.keys())
    all_combos = list(itertools.product(*[slots[k] for k in keys]))
    RNG.shuffle(all_combos)
    seen, n = set(), 0
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
    for q in questions:
        pairs.append((_clean(q), answer))


# ============================================================================
# Basic knowledge (kept small -- this is a sports bot)
# ============================================================================
def arithmetic():
    ADD_SUB = range(0, 21)
    MUL = range(0, 13)
    for a in ADD_SUB:
        for b in ADD_SUB:
            s = a + b
            simple(f"{a} plus {b} equals {s}.",
                   [f"what is {a} plus {b}?", f"what is {a} + {b}?", f"what's {a} plus {b}?",
                    f"{a} plus {b}", f"add {a} and {b}", f"how much is {a} plus {b}?",
                    f"what is the sum of {a} and {b}?"])
            if a >= b:
                simple(f"{a} minus {b} equals {a - b}.",
                       [f"what is {a} minus {b}?", f"what is {a} - {b}?", f"{a} minus {b}",
                        f"subtract {b} from {a}", f"how much is {a} minus {b}?"])
    for a in MUL:
        for b in MUL:
            simple(f"{a} times {b} equals {a * b}.",
                   [f"what is {a} times {b}?", f"what is {a} * {b}?", f"{a} times {b}",
                    f"multiply {a} and {b}", f"{a} x {b}", f"how much is {a} times {b}?"])


def basics():
    simple("The sky is blue.", ["what color is the sky?", "what colour is the sky?"])
    simple("Grass is green.", ["what color is grass?"])
    simple("There are seven days in a week.",
           ["how many days are in a week?", "days in a week", "how many days in a week?"])
    simple("There are twelve months in a year.",
           ["how many months are in a year?", "months in a year"])
    simple("There are twenty four hours in a day.",
           ["how many hours are in a day?", "hours in a day"])
    simple("There are sixty minutes in an hour.", ["how many minutes are in an hour?"])
    for a, b in [("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"),
                 ("win", "lose"), ("left", "right")]:
        simple(f"The opposite of {a} is {b}.",
               [f"what is the opposite of {a}?", f"opposite of {a}", f"what's the opposite of {a}?"])
    simple("A triangle has three sides.", ["how many sides does a triangle have?"])
    simple("A square has four sides.", ["how many sides does a square have?"])


# ============================================================================
# Sports: rules, scoring, competitions, terminology
# ============================================================================
SPORTS = {
    "soccer": {"aka": ["soccer", "association football"], "players": "eleven",
               "champ": "the World Cup", "gov": "FIFA", "score": "a goal", "field": "a pitch"},
    "basketball": {"aka": ["basketball"], "players": "five", "champ": "the NBA Finals",
                   "league": "the NBA", "gov": "FIBA", "score": "a basket", "field": "a court"},
    "american football": {"aka": ["american football", "football"], "players": "eleven",
                          "champ": "the Super Bowl", "league": "the NFL", "score": "a touchdown",
                          "field": "a field"},
    "baseball": {"aka": ["baseball"], "players": "nine", "champ": "the World Series",
                 "league": "MLB", "score": "a run", "field": "a diamond"},
    "ice hockey": {"aka": ["ice hockey", "hockey"], "players": "six", "champ": "the Stanley Cup",
                   "league": "the NHL", "score": "a goal", "field": "a rink"},
    "tennis": {"aka": ["tennis"], "players": "one", "champ": "a Grand Slam", "score": "a point",
               "field": "a court"},
    "cricket": {"aka": ["cricket"], "players": "eleven", "score": "a run", "field": "a pitch"},
    "rugby": {"aka": ["rugby", "rugby union"], "players": "fifteen", "score": "a try", "field": "a pitch"},
    "volleyball": {"aka": ["volleyball"], "players": "six", "score": "a point", "field": "a court"},
    "golf": {"aka": ["golf"], "players": "one", "field": "a course"},
}


def sports():
    for name, d in SPORTS.items():
        aka = d["aka"]
        if "players" in d:
            fact(f"A {name} team has {d['players']} players.",
                 "{lead} players are {on} a {s} team{q}",
                 {"lead": HOWMANY, "on": ["on", "in"], "s": aka, "q": Q})
            fact(f"A {name} team has {d['players']} players.",
                 "{lead} players {play} {s}{q}",
                 {"lead": HOWMANY, "play": ["play", "are in", "are on a team in"], "s": aka, "q": Q})
        if "champ" in d:
            fact(f"The {name} championship is called {d['champ']}.",
                 "{lead} the {s} championship called{q}", {"lead": ["what is", "what's"], "s": aka, "q": Q})
            fact(f"The {name} championship is called {d['champ']}.",
                 "{lead} the {s} championship{q}", {"lead": ["what do they call", "name"], "s": aka, "q": Q})
        if "gov" in d:
            fact(f"The governing body of {name} is {d['gov']}.",
                 "{lead} the governing body of {s}{q}",
                 {"lead": ["what is", "what's", "name", "tell me"], "s": aka, "q": Q})
        if "score" in d:
            fact(f"In {name} you score {d['score']}.",
                 "{lead} do you score in {s}{q}", {"lead": ["what", "how"], "s": aka, "q": Q})
        if "field" in d:
            fact(f"{name.capitalize()} is played on {d['field']}.",
                 "{lead} is {s} played on{q}", {"lead": ["what", "where"], "s": aka, "q": Q})

    # scoring, rules, terms
    simple("A touchdown is worth six points.",
           ["how many points is a touchdown?", "how much is a touchdown worth?", "what is a touchdown worth?"])
    simple("A field goal in american football is worth three points.",
           ["how many points is a field goal?", "what is a field goal worth in football?"])
    simple("A basket in basketball is worth two points.",
           ["how many points is a basket?", "how much is a field goal worth in basketball?"])
    simple("A three pointer in basketball is worth three points.",
           ["how many points is a three pointer?", "what is a three pointer worth?"])
    simple("A free throw in basketball is worth one point.",
           ["how many points is a free throw?", "what is a free throw worth?"])
    simple("A try in rugby is worth five points.", ["how many points is a try?", "what is a try worth?"])
    simple("A conversion in rugby is worth two points.", ["how many points is a conversion?"])
    simple("Baseball has nine innings.",
           ["how many innings are in baseball?", "how many innings in a baseball game?"])
    simple("A batter is out after three strikes.",
           ["how many strikes before you are out?", "how many strikes is an out?"])
    simple("There are three outs in each half of a baseball inning.",
           ["how many outs are in an inning?", "how many outs per inning?"])
    simple("It takes four balls to draw a walk in baseball.", ["how many balls is a walk?"])
    simple("A grand slam in baseball is a home run with the bases loaded, scoring four runs.",
           ["what is a grand slam in baseball?", "what is a grand slam?"])
    simple("There are four bases in baseball.", ["how many bases are in baseball?"])
    simple("A basketball hoop is ten feet high.",
           ["how high is a basketball hoop?", "how tall is a basketball hoop?"])
    simple("A player fouls out of an NBA game after six personal fouls.",
           ["how many fouls to foul out in the nba?", "how many fouls before you foul out?"])
    simple("A game of golf has eighteen holes.",
           ["how many holes are in golf?", "how many holes in a round of golf?"])
    simple("A hole in one is sinking the ball in a single shot in golf.", ["what is a hole in one?"])
    simple("In golf, a birdie is one stroke under par.", ["what is a birdie in golf?"])
    simple("In golf, an eagle is two strokes under par.", ["what is an eagle in golf?"])
    simple("In golf, a bogey is one stroke over par.", ["what is a bogey in golf?"])
    simple("Par is the expected number of strokes for a hole in golf.", ["what is par in golf?"])
    simple("A hat trick is scoring three goals in one game.",
           ["what is a hat trick?", "how many goals is a hat trick?"])
    simple("A red card sends a player off in soccer.",
           ["what does a red card mean?", "what is a red card?", "which card sends a player off?"])
    simple("A yellow card is a caution in soccer.", ["what is a yellow card?"])
    simple("A clean sheet in soccer is when a team concedes no goals.", ["what is a clean sheet?"])
    simple("In tennis, a score of zero is called love.",
           ["what is love in tennis?", "what does love mean in tennis?"])
    simple("In tennis, deuce means the score is forty forty.", ["what is deuce in tennis?"])
    simple("The four tennis Grand Slams are the Australian Open, the French Open, Wimbledon, and the US Open.",
           ["what are the tennis grand slams?", "name the four grand slams", "what are the four majors in tennis?"])
    simple("There are six balls in an over in cricket.", ["how many balls are in an over?"])
    simple("A century in cricket is a score of one hundred runs.", ["what is a century in cricket?"])
    simple("A marathon is about twenty six miles long.",
           ["how long is a marathon?", "how many miles is a marathon?"])
    simple("The Olympics are held every four years.",
           ["how often are the olympics held?", "how often are the olympics?"])
    simple("There are five rings on the Olympic flag.", ["how many rings are on the olympic flag?"])
    simple("Bowling has ten pins.", ["how many pins are in bowling?"])
    simple("A soccer match is ninety minutes long.",
           ["how long is a soccer game?", "how long is a soccer match?"])
    simple("A basketball game has four quarters.", ["how many quarters are in basketball?"])
    simple("An ice hockey game has three periods.", ["how many periods are in hockey?"])
    simple("The four main swimming strokes are freestyle, backstroke, breaststroke, and butterfly.",
           ["what are the swimming strokes?", "name the swimming strokes"])
    # reverse "which sport ..."
    simple("The sport played with a puck on ice is ice hockey.",
           ["what sport uses a puck?", "which sport is played with a puck?"])
    simple("The sport where you score a home run is baseball.",
           ["in which sport do you score a home run?", "what sport has a home run?"])
    simple("The sport where you score a touchdown is american football.",
           ["in which sport do you score a touchdown?", "what sport has a touchdown?"])
    simple("The sport with a slam dunk is basketball.",
           ["in which sport is there a slam dunk?", "what sport has a slam dunk?"])
    simple("The sport that uses a shuttlecock is badminton.", ["what sport uses a shuttlecock?"])
    simple("Soccer is the most popular sport in the world.",
           ["what is the most popular sport in the world?", "what is the most popular sport?"])


# ============================================================================
# Leagues and competitions
# ============================================================================
# name -> (description, [question aliases])
LEAGUES = [
    ("the NBA is the top professional basketball league in North America.",
     ["what is the nba", "what's the nba", "tell me about the nba"]),
    ("the NFL is the top professional American football league.",
     ["what is the nfl", "what's the nfl"]),
    ("MLB, or Major League Baseball, is the top professional baseball league.",
     ["what is mlb", "what is major league baseball"]),
    ("the NHL is the top professional ice hockey league in North America.",
     ["what is the nhl", "what's the nhl"]),
    ("the Premier League is the top soccer league in England.",
     ["what is the premier league", "what's the premier league"]),
    ("La Liga is the top soccer league in Spain.", ["what is la liga"]),
    ("Serie A is the top soccer league in Italy.", ["what is serie a"]),
    ("the Bundesliga is the top soccer league in Germany.", ["what is the bundesliga"]),
    ("Ligue 1 is the top soccer league in France.", ["what is ligue 1"]),
    ("MLS, or Major League Soccer, is the top soccer league in the United States.",
     ["what is mls", "what is major league soccer"]),
    ("the Champions League is the top club soccer competition in Europe.",
     ["what is the champions league"]),
    ("the Super Bowl is the championship game of the NFL.", ["what is the super bowl", "what's the super bowl"]),
    ("the World Series is the championship of Major League Baseball.", ["what is the world series"]),
    ("the Stanley Cup is the championship trophy of the NHL.", ["what is the stanley cup"]),
    ("the NBA Finals is the championship series of the NBA.", ["what is the nba finals"]),
    ("the World Cup is the biggest tournament in soccer, held every four years.",
     ["what is the world cup", "what's the world cup"]),
    ("the Ashes is a cricket series between England and Australia.", ["what is the ashes"]),
    ("the Ryder Cup is a golf competition between the United States and Europe.", ["what is the ryder cup"]),
    ("Wimbledon is a tennis tournament.", ["what is wimbledon", "what sport is played at wimbledon"]),
    ("the Masters is a golf tournament.", ["what is the masters", "what sport is the masters"]),
    ("the Tour de France is a cycling race.", ["what is the tour de france", "what sport is the tour de france"]),
    ("Formula One is a motor racing competition.", ["what is formula one", "what is f1"]),
    ("the Kentucky Derby is a horse racing event.", ["what is the kentucky derby"]),
    ("the Indianapolis 500 is a motor racing event.", ["what is the indy 500", "what is the indianapolis 500"]),
]


def leagues():
    lead = ["", "what is ", "what's ", "tell me about ", "explain "]
    for desc, qs in LEAGUES:
        answer = desc[0].upper() + desc[1:]
        variants = []
        for q in qs:
            base = q.rstrip("?")
            variants.append(base + "?")
            variants.append(base)
        simple(answer, variants)


# ============================================================================
# Athletes (hundreds) -- sport -> (default role, [(name, nationality[, role])])
# ============================================================================
ATHLETES_BY_SPORT = {
    ("soccer", "soccer player"): [
        ("cristiano ronaldo", "Portuguese"), ("lionel messi", "Argentine"), ("pele", "Brazilian"),
        ("diego maradona", "Argentine"), ("neymar", "Brazilian"), ("kylian mbappe", "French"),
        ("david beckham", "English"), ("ronaldinho", "Brazilian"), ("zinedine zidane", "French"),
        ("thierry henry", "French"), ("wayne rooney", "English"), ("steven gerrard", "English"),
        ("frank lampard", "English"), ("andres iniesta", "Spanish"), ("xavi", "Spanish"),
        ("sergio ramos", "Spanish"), ("luka modric", "Croatian"), ("robert lewandowski", "Polish"),
        ("mohamed salah", "Egyptian"), ("kevin de bruyne", "Belgian"), ("eden hazard", "Belgian"),
        ("harry kane", "English"), ("erling haaland", "Norwegian"), ("luis suarez", "Uruguayan"),
        ("gareth bale", "Welsh"), ("karim benzema", "French"), ("sergio aguero", "Argentine"),
        ("gianluigi buffon", "Italian"), ("paolo maldini", "Italian"), ("andrea pirlo", "Italian"),
        ("francesco totti", "Italian"), ("ryan giggs", "Welsh"), ("iker casillas", "Spanish"),
        ("roberto carlos", "Brazilian"), ("kaka", "Brazilian"), ("didier drogba", "Ivorian"),
        ("samuel etoo", "Cameroonian"), ("johan cruyff", "Dutch"), ("marco van basten", "Dutch"),
        ("ruud gullit", "Dutch"), ("franz beckenbauer", "German"), ("miroslav klose", "German"),
        ("thomas muller", "German"), ("manuel neuer", "German"), ("toni kroos", "German"),
        ("antoine griezmann", "French"), ("paul pogba", "French"), ("virgil van dijk", "Dutch"),
        ("sadio mane", "Senegalese"), ("vinicius junior", "Brazilian"), ("george best", "Northern Irish"),
    ],
    ("basketball", "basketball player"): [
        ("michael jordan", "American"), ("lebron james", "American"), ("kobe bryant", "American"),
        ("shaquille oneal", "American"), ("magic johnson", "American"), ("larry bird", "American"),
        ("kareem abdul jabbar", "American"), ("wilt chamberlain", "American"), ("bill russell", "American"),
        ("tim duncan", "American"), ("kevin durant", "American"), ("stephen curry", "American"),
        ("kawhi leonard", "American"), ("giannis antetokounmpo", "Greek"), ("dirk nowitzki", "German"),
        ("hakeem olajuwon", "Nigerian"), ("yao ming", "Chinese"), ("allen iverson", "American"),
        ("dwyane wade", "American"), ("charles barkley", "American"), ("scottie pippen", "American"),
        ("russell westbrook", "American"), ("james harden", "American"), ("chris paul", "American"),
        ("carmelo anthony", "American"), ("damian lillard", "American"), ("luka doncic", "Slovenian"),
        ("nikola jokic", "Serbian"), ("anthony davis", "American"), ("karl malone", "American"),
        ("john stockton", "American"), ("patrick ewing", "American"), ("david robinson", "American"),
        ("dennis rodman", "American"),
    ],
    ("tennis", "tennis player"): [
        ("roger federer", "Swiss"), ("rafael nadal", "Spanish"), ("novak djokovic", "Serbian"),
        ("serena williams", "American"), ("venus williams", "American"), ("pete sampras", "American"),
        ("andre agassi", "American"), ("bjorn borg", "Swedish"), ("john mcenroe", "American"),
        ("andy murray", "British"), ("boris becker", "German"), ("steffi graf", "German"),
        ("martina navratilova", "American"), ("maria sharapova", "Russian"), ("naomi osaka", "Japanese"),
        ("stan wawrinka", "Swiss"), ("jimmy connors", "American"), ("rod laver", "Australian"),
        ("chris evert", "American"), ("carlos alcaraz", "Spanish"), ("daniil medvedev", "Russian"),
    ],
    ("american football", "American football player"): [
        ("tom brady", "American"), ("peyton manning", "American"), ("joe montana", "American"),
        ("jerry rice", "American"), ("aaron rodgers", "American"), ("patrick mahomes", "American"),
        ("brett favre", "American"), ("dan marino", "American"), ("emmitt smith", "American"),
        ("barry sanders", "American"), ("john elway", "American"), ("troy aikman", "American"),
        ("drew brees", "American"), ("ray lewis", "American"), ("deion sanders", "American"),
        ("lawrence taylor", "American"),
    ],
    ("baseball", "baseball player"): [
        ("babe ruth", "American"), ("jackie robinson", "American"), ("hank aaron", "American"),
        ("willie mays", "American"), ("barry bonds", "American"), ("derek jeter", "American"),
        ("ken griffey junior", "American"), ("mickey mantle", "American"), ("lou gehrig", "American"),
        ("ted williams", "American"), ("joe dimaggio", "American"), ("mike trout", "American"),
        ("shohei ohtani", "Japanese"), ("ichiro suzuki", "Japanese"), ("alex rodriguez", "American"),
        ("pedro martinez", "Dominican"), ("david ortiz", "Dominican"),
    ],
    ("boxing", "boxer"): [
        ("muhammad ali", "American"), ("mike tyson", "American"), ("floyd mayweather", "American"),
        ("manny pacquiao", "Filipino"), ("rocky marciano", "American"), ("joe frazier", "American"),
        ("george foreman", "American"), ("sugar ray robinson", "American"), ("canelo alvarez", "Mexican"),
        ("lennox lewis", "British"), ("evander holyfield", "American"), ("larry holmes", "American"),
        ("anthony joshua", "British"), ("tyson fury", "British"),
    ],
    ("golf", "golfer"): [
        ("tiger woods", "American"), ("jack nicklaus", "American"), ("arnold palmer", "American"),
        ("rory mcilroy", "Northern Irish"), ("phil mickelson", "American"), ("gary player", "South African"),
        ("sam snead", "American"), ("ben hogan", "American"), ("seve ballesteros", "Spanish"),
        ("jordan spieth", "American"), ("dustin johnson", "American"), ("brooks koepka", "American"),
    ],
    ("Formula One", "Formula One driver"): [
        ("lewis hamilton", "British"), ("michael schumacher", "German"), ("ayrton senna", "Brazilian"),
        ("sebastian vettel", "German"), ("max verstappen", "Dutch"), ("fernando alonso", "Spanish"),
        ("niki lauda", "Austrian"), ("alain prost", "French"), ("jenson button", "British"),
        ("kimi raikkonen", "Finnish"), ("nico rosberg", "German"), ("jackie stewart", "British"),
    ],
    ("ice hockey", "ice hockey player"): [
        ("wayne gretzky", "Canadian"), ("mario lemieux", "Canadian"), ("sidney crosby", "Canadian"),
        ("alexander ovechkin", "Russian"), ("bobby orr", "Canadian"), ("gordie howe", "Canadian"),
        ("connor mcdavid", "Canadian"), ("patrick roy", "Canadian"), ("jaromir jagr", "Czech"),
    ],
    ("cricket", "cricketer"): [
        ("sachin tendulkar", "Indian"), ("virat kohli", "Indian"), ("ms dhoni", "Indian"),
        ("brian lara", "West Indian"), ("ricky ponting", "Australian"), ("shane warne", "Australian"),
        ("don bradman", "Australian"), ("jacques kallis", "South African"), ("kumar sangakkara", "Sri Lankan"),
        ("wasim akram", "Pakistani"), ("kapil dev", "Indian"), ("ben stokes", "English"),
    ],
    ("various", None): [
        ("usain bolt", "Jamaican", "sprinter", "sprinting"),
        ("michael phelps", "American", "swimmer", "swimming"),
        ("carl lewis", "American", "sprinter", "sprinting"),
        ("jesse owens", "American", "sprinter", "sprinting"),
        ("mo farah", "British", "distance runner", "athletics"),
        ("simone biles", "American", "gymnast", "gymnastics"),
        ("nadia comaneci", "Romanian", "gymnast", "gymnastics"),
    ],
}


# A second, larger roster merged into the above at generation time.
MORE_ATHLETES_BY_SPORT = {
    ("soccer", "soccer player"): [
        ("sergio busquets", "Spanish"), ("gerard pique", "Spanish"), ("carles puyol", "Spanish"),
        ("fernando torres", "Spanish"), ("david villa", "Spanish"), ("raul", "Spanish"),
        ("cafu", "Brazilian"), ("ronaldo nazario", "Brazilian"), ("rivaldo", "Brazilian"),
        ("romario", "Brazilian"), ("marcelo", "Brazilian"), ("dani alves", "Brazilian"),
        ("thiago silva", "Brazilian"), ("bobby charlton", "English"), ("gary lineker", "English"),
        ("alan shearer", "English"), ("michael owen", "English"), ("paul scholes", "English"),
        ("rio ferdinand", "English"), ("john terry", "English"), ("dennis bergkamp", "Dutch"),
        ("patrick vieira", "French"), ("roy keane", "Irish"), ("alessandro del piero", "Italian"),
        ("roberto baggio", "Italian"), ("luis figo", "Portuguese"), ("cesc fabregas", "Spanish"),
        ("arjen robben", "Dutch"), ("mesut ozil", "German"), ("philipp lahm", "German"),
        ("bastian schweinsteiger", "German"), ("angel di maria", "Argentine"),
        ("edinson cavani", "Uruguayan"), ("james rodriguez", "Colombian"), ("alexis sanchez", "Chilean"),
        ("lev yashin", "Russian"),
    ],
    ("basketball", "basketball player"): [
        ("kevin garnett", "American"), ("ray allen", "American"), ("paul pierce", "American"),
        ("steve nash", "Canadian"), ("jason kidd", "American"), ("vince carter", "American"),
        ("tracy mcgrady", "American"), ("reggie miller", "American"), ("klay thompson", "American"),
        ("draymond green", "American"), ("kyrie irving", "American"), ("jimmy butler", "American"),
        ("joel embiid", "Cameroonian"), ("ja morant", "American"), ("zion williamson", "American"),
        ("devin booker", "American"), ("jayson tatum", "American"), ("pau gasol", "Spanish"),
        ("manu ginobili", "Argentine"), ("tony parker", "French"),
    ],
    ("tennis", "tennis player"): [
        ("andy roddick", "American"), ("lleyton hewitt", "Australian"), ("justine henin", "Belgian"),
        ("kim clijsters", "Belgian"), ("simona halep", "Romanian"), ("victoria azarenka", "Belarusian"),
        ("stefanos tsitsipas", "Greek"), ("alexander zverev", "German"), ("dominic thiem", "Austrian"),
        ("juan martin del potro", "Argentine"), ("gustavo kuerten", "Brazilian"), ("jim courier", "American"),
    ],
    ("american football", "American football player"): [
        ("randy moss", "American"), ("reggie white", "American"), ("walter payton", "American"),
        ("jim brown", "American"), ("bo jackson", "American"), ("adrian peterson", "American"),
        ("rob gronkowski", "American"), ("calvin johnson", "American"), ("jj watt", "American"),
        ("russell wilson", "American"), ("cam newton", "American"), ("eli manning", "American"),
        ("ben roethlisberger", "American"), ("michael vick", "American"),
    ],
    ("baseball", "baseball player"): [
        ("sandy koufax", "American"), ("nolan ryan", "American"), ("roger clemens", "American"),
        ("greg maddux", "American"), ("randy johnson", "American"), ("cal ripken junior", "American"),
        ("tony gwynn", "American"), ("rickey henderson", "American"), ("bob gibson", "American"),
        ("yogi berra", "American"), ("roberto clemente", "Puerto Rican"), ("albert pujols", "Dominican"),
        ("frank thomas", "American"),
    ],
    ("boxing", "boxer"): [
        ("sugar ray leonard", "American"), ("marvin hagler", "American"), ("roberto duran", "Panamanian"),
        ("julio cesar chavez", "Mexican"), ("oscar de la hoya", "American"), ("bernard hopkins", "American"),
        ("wladimir klitschko", "Ukrainian"), ("vitali klitschko", "Ukrainian"), ("deontay wilder", "American"),
        ("gennady golovkin", "Kazakh"),
    ],
    ("golf", "golfer"): [
        ("tom watson", "American"), ("lee trevino", "American"), ("nick faldo", "English"),
        ("greg norman", "Australian"), ("vijay singh", "Fijian"), ("ernie els", "South African"),
        ("justin thomas", "American"), ("collin morikawa", "American"), ("jon rahm", "Spanish"),
        ("scottie scheffler", "American"),
    ],
    ("Formula One", "Formula One driver"): [
        ("mika hakkinen", "Finnish"), ("nelson piquet", "Brazilian"), ("damon hill", "British"),
        ("nigel mansell", "British"), ("emerson fittipaldi", "Brazilian"), ("valtteri bottas", "Finnish"),
        ("daniel ricciardo", "Australian"), ("charles leclerc", "Monegasque"), ("lando norris", "British"),
    ],
    ("ice hockey", "ice hockey player"): [
        ("mark messier", "Canadian"), ("steve yzerman", "Canadian"), ("ray bourque", "Canadian"),
        ("martin brodeur", "Canadian"), ("dominik hasek", "Czech"), ("nicklas lidstrom", "Swedish"),
        ("teemu selanne", "Finnish"), ("joe sakic", "Canadian"), ("patrick kane", "American"),
        ("auston matthews", "American"),
    ],
    ("cricket", "cricketer"): [
        ("rahul dravid", "Indian"), ("rohit sharma", "Indian"), ("ab de villiers", "South African"),
        ("muttiah muralitharan", "Sri Lankan"), ("glenn mcgrath", "Australian"), ("adam gilchrist", "Australian"),
        ("viv richards", "West Indian"), ("imran khan", "Pakistani"), ("shahid afridi", "Pakistani"),
        ("jasprit bumrah", "Indian"),
    ],
    ("various", None): [
        ("katie ledecky", "American", "swimmer", "swimming"),
        ("mark spitz", "American", "swimmer", "swimming"),
        ("eliud kipchoge", "Kenyan", "marathon runner", "athletics"),
        ("haile gebrselassie", "Ethiopian", "distance runner", "athletics"),
        ("florence griffith joyner", "American", "sprinter", "sprinting"),
        ("sergey bubka", "Ukrainian", "pole vaulter", "athletics"),
    ],
}


def _merged_athletes():
    merged = {}
    for src in (ATHLETES_BY_SPORT, MORE_ATHLETES_BY_SPORT):
        for key, entries in src.items():
            merged.setdefault(key, [])
            merged[key].extend(entries)
    return merged


def athletes():
    who = ["who is", "who's", "who was", "tell me about"]
    for (sport, default_role), entries in _merged_athletes().items():
        for entry in entries:
            name, nat = entry[0], entry[1]
            role = entry[2] if len(entry) > 2 else default_role
            the_sport = entry[3] if len(entry) > 3 else sport
            disp = name.title()
            fact(f"{disp} is a famous {nat} {role}.",
                 "{lead} {name}{q}", {"lead": who, "name": [name], "q": Q})
            fact(f"{disp}'s sport is {the_sport}.",
                 "{lead} sport does {name} play{q}", {"lead": ["what", "which"], "name": [name], "q": Q})
            fact(f"{disp}'s sport is {the_sport}.",
                 "what does {name} play{q}", {"name": [name], "q": Q})


# ============================================================================
# Teams (100+) -- (full name, nickname, league, sport, city)
# ============================================================================
def _nba():
    data = [
        ("Atlanta Hawks", "Atlanta"), ("Boston Celtics", "Boston"), ("Brooklyn Nets", "Brooklyn"),
        ("Charlotte Hornets", "Charlotte"), ("Chicago Bulls", "Chicago"), ("Cleveland Cavaliers", "Cleveland"),
        ("Dallas Mavericks", "Dallas"), ("Denver Nuggets", "Denver"), ("Detroit Pistons", "Detroit"),
        ("Golden State Warriors", "San Francisco"), ("Houston Rockets", "Houston"),
        ("Indiana Pacers", "Indianapolis"), ("Los Angeles Clippers", "Los Angeles"),
        ("Los Angeles Lakers", "Los Angeles"), ("Memphis Grizzlies", "Memphis"), ("Miami Heat", "Miami"),
        ("Milwaukee Bucks", "Milwaukee"), ("Minnesota Timberwolves", "Minneapolis"),
        ("New Orleans Pelicans", "New Orleans"), ("New York Knicks", "New York"),
        ("Oklahoma City Thunder", "Oklahoma City"), ("Orlando Magic", "Orlando"),
        ("Philadelphia 76ers", "Philadelphia"), ("Phoenix Suns", "Phoenix"),
        ("Portland Trail Blazers", "Portland"), ("Sacramento Kings", "Sacramento"),
        ("San Antonio Spurs", "San Antonio"), ("Toronto Raptors", "Toronto"),
        ("Utah Jazz", "Salt Lake City"), ("Washington Wizards", "Washington"),
    ]
    return [(full, full.split()[-1], "the NBA", "basketball", city) for full, city in data]


def _nfl():
    data = [
        ("Arizona Cardinals", "Arizona"), ("Atlanta Falcons", "Atlanta"), ("Baltimore Ravens", "Baltimore"),
        ("Buffalo Bills", "Buffalo"), ("Carolina Panthers", "Charlotte"), ("Chicago Bears", "Chicago"),
        ("Cincinnati Bengals", "Cincinnati"), ("Cleveland Browns", "Cleveland"), ("Dallas Cowboys", "Dallas"),
        ("Denver Broncos", "Denver"), ("Detroit Lions", "Detroit"), ("Green Bay Packers", "Green Bay"),
        ("Houston Texans", "Houston"), ("Indianapolis Colts", "Indianapolis"),
        ("Jacksonville Jaguars", "Jacksonville"), ("Kansas City Chiefs", "Kansas City"),
        ("Las Vegas Raiders", "Las Vegas"), ("Los Angeles Chargers", "Los Angeles"),
        ("Los Angeles Rams", "Los Angeles"), ("Miami Dolphins", "Miami"),
        ("Minnesota Vikings", "Minneapolis"), ("New England Patriots", "New England"),
        ("New Orleans Saints", "New Orleans"), ("New York Giants", "New York"),
        ("New York Jets", "New York"), ("Philadelphia Eagles", "Philadelphia"),
        ("Pittsburgh Steelers", "Pittsburgh"), ("San Francisco 49ers", "San Francisco"),
        ("Seattle Seahawks", "Seattle"), ("Tampa Bay Buccaneers", "Tampa"),
        ("Tennessee Titans", "Nashville"), ("Washington Commanders", "Washington"),
    ]
    return [(full, full.split()[-1], "the NFL", "American football", city) for full, city in data]


def _mlb():
    data = [
        ("New York Yankees", "New York"), ("Boston Red Sox", "Boston"), ("Los Angeles Dodgers", "Los Angeles"),
        ("Chicago Cubs", "Chicago"), ("San Francisco Giants", "San Francisco"),
        ("St Louis Cardinals", "St Louis"), ("Houston Astros", "Houston"), ("Atlanta Braves", "Atlanta"),
        ("New York Mets", "New York"), ("Philadelphia Phillies", "Philadelphia"),
        ("Chicago White Sox", "Chicago"), ("Detroit Tigers", "Detroit"), ("Toronto Blue Jays", "Toronto"),
        ("Seattle Mariners", "Seattle"), ("Texas Rangers", "Texas"), ("Oakland Athletics", "Oakland"),
    ]
    return [(full, full.split()[-1], "MLB", "baseball", city) for full, city in data]


def _nhl():
    data = [
        ("Boston Bruins", "Boston"), ("Montreal Canadiens", "Montreal"), ("Toronto Maple Leafs", "Toronto"),
        ("Chicago Blackhawks", "Chicago"), ("Detroit Red Wings", "Detroit"), ("New York Rangers", "New York"),
        ("Pittsburgh Penguins", "Pittsburgh"), ("Edmonton Oilers", "Edmonton"), ("Tampa Bay Lightning", "Tampa"),
        ("Colorado Avalanche", "Denver"), ("Vegas Golden Knights", "Las Vegas"),
        ("Los Angeles Kings", "Los Angeles"), ("Washington Capitals", "Washington"),
        ("Philadelphia Flyers", "Philadelphia"),
    ]
    return [(full, full.split()[-1], "the NHL", "ice hockey", city) for full, city in data]


def _soccer_clubs():
    # (full, nickname, league, city)
    data = [
        ("Manchester United", "United", "the Premier League", "Manchester"),
        ("Manchester City", "City", "the Premier League", "Manchester"),
        ("Liverpool", "Liverpool", "the Premier League", "Liverpool"),
        ("Chelsea", "Chelsea", "the Premier League", "London"),
        ("Arsenal", "Arsenal", "the Premier League", "London"),
        ("Tottenham Hotspur", "Tottenham", "the Premier League", "London"),
        ("Real Madrid", "Madrid", "La Liga", "Madrid"),
        ("Barcelona", "Barcelona", "La Liga", "Barcelona"),
        ("Atletico Madrid", "Atletico", "La Liga", "Madrid"),
        ("Bayern Munich", "Bayern", "the Bundesliga", "Munich"),
        ("Borussia Dortmund", "Dortmund", "the Bundesliga", "Dortmund"),
        ("Juventus", "Juventus", "Serie A", "Turin"),
        ("AC Milan", "Milan", "Serie A", "Milan"),
        ("Inter Milan", "Inter", "Serie A", "Milan"),
        ("Napoli", "Napoli", "Serie A", "Naples"),
        ("Paris Saint Germain", "PSG", "Ligue 1", "Paris"),
        ("Ajax", "Ajax", "the Eredivisie", "Amsterdam"),
        ("Porto", "Porto", "the Primeira Liga", "Porto"),
        ("Benfica", "Benfica", "the Primeira Liga", "Lisbon"),
        ("Celtic", "Celtic", "the Scottish Premiership", "Glasgow"),
    ]
    return [(full, nick, league, "soccer", city) for full, nick, league, city in data]


def teams():
    all_teams = _nba() + _nfl() + _mlb() + _nhl() + _soccer_clubs()
    for full, nick, league, sport, city in all_teams:
        aka = list({full.lower(), nick.lower(), ("the " + nick).lower()})
        fact(f"{full} play in {league}.",
             "{lead} {t} play{inn}{q}",
             {"lead": ["what league do", "which league do", "what league are"],
              "t": aka, "inn": [" in", ""], "q": Q}, cap=6)
        fact(f"{full} are a {sport} team.",
             "{lead} sport do {t} play{q}", {"lead": ["what", "which"], "t": aka, "q": Q}, cap=6)
        fact(f"{full} are based in {city}.",
             "{lead} {t} {loc}{q}",
             {"lead": ["what city are", "where are", "where do"], "t": aka,
              "loc": ["from", "based", "play"], "q": Q}, cap=6)


# ============================================================================
# Home venues (stable, iconic stadiums) and playing positions
# ============================================================================
VENUES = [
    ("The New York Yankees play at Yankee Stadium.",
     ["where do the yankees play", "what stadium do the yankees play at", "what is the yankees home stadium"]),
    ("The Boston Red Sox play at Fenway Park.",
     ["where do the red sox play", "what is the home of the red sox"]),
    ("The Chicago Cubs play at Wrigley Field.",
     ["where do the cubs play", "what is the home of the cubs"]),
    ("The Green Bay Packers play at Lambeau Field.",
     ["where do the packers play", "what is the home of the packers"]),
    ("The Dallas Cowboys play at AT&T Stadium.",
     ["where do the cowboys play", "what stadium do the cowboys play at"]),
    ("The New York Knicks play at Madison Square Garden.",
     ["where do the knicks play", "what is the home of the knicks"]),
    ("Manchester United play at Old Trafford.",
     ["where do manchester united play their home games", "what is the home stadium of manchester united",
      "what stadium do manchester united play at"]),
    ("Liverpool play at Anfield.",
     ["what is the home of liverpool", "what stadium do liverpool play at"]),
    ("Arsenal play at the Emirates Stadium.",
     ["what is the home of arsenal", "what stadium do arsenal play at"]),
    ("Real Madrid play at the Santiago Bernabeu.",
     ["what is the home of real madrid", "what stadium do real madrid play at"]),
    ("Barcelona play at Camp Nou.",
     ["what is the home of barcelona", "what stadium do barcelona play at"]),
    ("Bayern Munich play at the Allianz Arena.",
     ["what is the home of bayern munich", "what stadium do bayern munich play at"]),
]


def venues():
    for answer, qs in VENUES:
        variants = []
        for q in qs:
            variants += [q + "?", q]
        simple(answer, variants)


# position -> (sport, description)
POSITIONS = {
    "quarterback": ("american football", "the player who leads the offense and throws the ball"),
    "wide receiver": ("american football", "a player who catches passes"),
    "running back": ("american football", "a player who runs with the ball"),
    "goalkeeper": ("soccer", "the player who guards the goal"),
    "striker": ("soccer", "a forward whose main job is to score goals"),
    "defender": ("soccer", "a player who protects the goal"),
    "midfielder": ("soccer", "a player who plays in the middle of the pitch"),
    "pitcher": ("baseball", "the player who throws the ball to the batter"),
    "catcher": ("baseball", "the player who catches behind home plate"),
    "point guard": ("basketball", "the player who runs the offense"),
    "goalie": ("ice hockey", "the player who guards the net"),
}


def positions():
    for pos, (sport, desc) in POSITIONS.items():
        fact(f"A {pos} in {sport} is {desc}.",
             "{lead} a {p}{q}",
             {"lead": ["what is", "what's", "explain", "tell me about"], "p": [pos], "q": Q})
        fact(f"A {pos} in {sport} is {desc}.",
             "what does a {p} do{q}", {"p": [pos], "q": Q})


# ============================================================================
# Small talk + identity (rebranded as a sports bot)
# ============================================================================
def small_talk_and_meta():
    simple("Hello! Ask me a sports question.",
           ["hello", "hi", "hey", "hi there", "hello there", "hey there", "yo", "greetings"])
    simple("Hello! Ask me about athletes, teams, leagues, or sports rules.",
           ["good morning", "good afternoon", "good evening"])
    simple("I'm doing great and ready to talk sports. What would you like to know?",
           ["how are you", "how are you doing", "how's it going"])
    simple("You're welcome!", ["thanks", "thank you", "thanks a lot", "cheers"])
    simple("Goodbye!", ["bye", "goodbye", "see you", "see you later"])
    simple("I am a tiny sports chatbot, a GPT language model built from scratch in PyTorch.",
           ["what is your name?", "what's your name?", "who are you?", "what are you?", "tell me about yourself"])
    simple("I was built from scratch as an educational sports chatbot.",
           ["who made you?", "who created you?", "who built you?"])
    simple("I predict one word at a time using a small transformer neural network.",
           ["how do you work?", "how were you trained?"])
    simple("No. I am a small model that only knows about sports and a few basics.",
           ["are you chatgpt?", "are you a real ai?", "are you smart?"])
    simple("I know about athletes, teams, leagues, championships, and the rules of many sports.",
           ["what can you do?", "what do you know?", "help", "what can i ask?", "what can i ask you?"])


def main():
    arithmetic()
    basics()
    sports()
    leagues()
    athletes()
    teams()
    venues()
    positions()
    small_talk_and_meta()

    seen_q, unique = {}, []
    for q, a in pairs:
        if q in seen_q:
            continue
        seen_q[q] = a
        unique.append((q, a))

    os.makedirs("data", exist_ok=True)
    with open("data/chat.jsonl", "w", encoding="utf-8") as f:
        for q, a in unique:
            f.write(json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n")

    n_arith = sum(1 for _, a in unique if "equals" in a)
    print(f"Wrote {len(unique)} unique pairs | arithmetic {n_arith} | "
          f"sports+basics {len(unique) - n_arith} | distinct answers {len(set(a for _, a in unique))}")


if __name__ == "__main__":
    main()
