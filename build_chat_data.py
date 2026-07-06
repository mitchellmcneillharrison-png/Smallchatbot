"""
Generate the chatbot's training data: a set of (question, answer) pairs written
to data/chat.jsonl.

This is the bot's entire "knowledge". Because it's a tiny from-scratch model, it
can only learn to answer things that appear here (and close rewordings) -- there
is no pretraining and no outside knowledge. So we deliberately cover a small,
self-contained world:

  * arithmetic (addition / subtraction / multiplication over small numbers),
    with several phrasings, so questions like "what is 1 plus 1?" work;
  * greetings and small talk;
  * questions about the bot itself (honest about its limits);
  * a hand-written set of general-knowledge facts, each asked several ways to
    encourage the model to generalize across phrasings.

Edit or extend the FACTS list to teach the bot new things, then re-run:
    python build_chat_data.py
    python train_chat.py
"""

import json
import os

NUM_RANGE = range(0, 11)  # operands 0..10 for arithmetic


def arithmetic_pairs():
    pairs = []
    for a in NUM_RANGE:
        for b in NUM_RANGE:
            # addition
            s = a + b
            ans = f"{a} plus {b} equals {s}."
            for q in (
                f"what is {a} plus {b}?",
                f"what is {a} + {b}?",
                f"what's {a} plus {b}?",
                f"{a} plus {b}",
                f"add {a} and {b}",
                f"calculate {a} + {b}",
            ):
                pairs.append((q, ans))

            # subtraction (keep results non-negative)
            if a >= b:
                d = a - b
                ans = f"{a} minus {b} equals {d}."
                for q in (
                    f"what is {a} minus {b}?",
                    f"what is {a} - {b}?",
                    f"{a} minus {b}",
                    f"subtract {b} from {a}",
                ):
                    pairs.append((q, ans))

            # multiplication
            p = a * b
            ans = f"{a} times {b} equals {p}."
            for q in (
                f"what is {a} times {b}?",
                f"what is {a} * {b}?",
                f"{a} times {b}",
                f"multiply {a} and {b}",
            ):
                pairs.append((q, ans))
    return pairs


# Greetings and small talk. Each entry is (list-of-question-phrasings, answer).
SMALL_TALK = [
    (["hello", "hi", "hey", "hi there", "hello there"], "Hello! Ask me a question."),
    (["good morning", "good afternoon", "good evening"], "Hello! How can I help?"),
    (["how are you", "how are you doing", "how's it going"],
     "I'm just a tiny program, but I'm running fine. What would you like to ask?"),
    (["thanks", "thank you", "thanks a lot"], "You're welcome!"),
    (["bye", "goodbye", "see you", "see you later"], "Goodbye!"),
    (["ok", "okay", "cool", "nice"], "Glad that helped. Ask me something else!"),
]

# Questions about the bot itself -- kept honest about what it is.
META = [
    (["what is your name", "what's your name", "who are you", "what are you"],
     "I am a tiny GPT language model, built from scratch in PyTorch."),
    (["who made you", "who created you", "who built you"],
     "I was built from scratch as an educational project."),
    (["how do you work", "how were you trained", "how do you generate text"],
     "I predict one word at a time using a small transformer neural network."),
    (["are you chatgpt", "are you gpt", "are you a real ai", "are you smart"],
     "No. I am a very small model trained on a tiny dataset, so I only know a few things."),
    (["what can you do", "what do you know", "help", "what can i ask"],
     "I can do simple arithmetic and answer a small set of general knowledge questions."),
    (["how many parameters do you have", "how big are you"],
     "I am tiny, with only about a million parameters."),
]

# General-knowledge facts. (list-of-question-phrasings, answer).
FACTS = [
    (["what is the capital of france", "capital of france"], "The capital of France is Paris."),
    (["what is the capital of japan", "capital of japan"], "The capital of Japan is Tokyo."),
    (["what is the capital of italy", "capital of italy"], "The capital of Italy is Rome."),
    (["what is the capital of spain", "capital of spain"], "The capital of Spain is Madrid."),
    (["what is the capital of germany", "capital of germany"], "The capital of Germany is Berlin."),
    (["what is the capital of england", "capital of england"], "The capital of England is London."),
    (["what is the capital of russia", "capital of russia"], "The capital of Russia is Moscow."),
    (["what is the capital of china", "capital of china"], "The capital of China is Beijing."),
    (["what is the capital of canada", "capital of canada"], "The capital of Canada is Ottawa."),
    (["what is the capital of egypt", "capital of egypt"], "The capital of Egypt is Cairo."),
    (["what color is the sky", "what colour is the sky"], "The sky is blue."),
    (["what color is grass", "what colour is grass"], "Grass is green."),
    (["what color is the sun"], "The sun is yellow."),
    (["what color is snow"], "Snow is white."),
    (["how many days are in a week", "days in a week"], "There are seven days in a week."),
    (["how many months are in a year", "months in a year"], "There are twelve months in a year."),
    (["how many days are in a year", "days in a year"], "There are three hundred and sixty five days in a year."),
    (["how many hours are in a day", "hours in a day"], "There are twenty four hours in a day."),
    (["how many minutes are in an hour", "minutes in an hour"], "There are sixty minutes in an hour."),
    (["how many seconds are in a minute", "seconds in a minute"], "There are sixty seconds in a minute."),
    (["how many planets are in the solar system", "how many planets are there"],
     "There are eight planets in the solar system."),
    (["what is the largest planet", "biggest planet"], "The largest planet is Jupiter."),
    (["what is the closest planet to the sun"], "The closest planet to the sun is Mercury."),
    (["what is the largest ocean", "biggest ocean"], "The largest ocean is the Pacific Ocean."),
    (["what is the tallest mountain", "highest mountain"], "The tallest mountain is Mount Everest."),
    (["what is the longest river", "longest river in the world"], "The longest river is the Nile."),
    (["what is water made of", "what is water"], "Water is made of hydrogen and oxygen."),
    (["what is the chemical symbol for water"], "The chemical symbol for water is H2O."),
    (["what gas do plants produce", "what do plants produce"], "Plants produce oxygen."),
    (["what do bees make", "what do bees produce"], "Bees make honey."),
    (["what is the fastest land animal"], "The fastest land animal is the cheetah."),
    (["what is the largest animal", "biggest animal"], "The largest animal is the blue whale."),
    (["how many legs does a spider have", "spider legs"], "A spider has eight legs."),
    (["how many legs does an insect have"], "An insect has six legs."),
    (["what is the freezing point of water"], "Water freezes at zero degrees celsius."),
    (["what is the boiling point of water"], "Water boils at one hundred degrees celsius."),
    (["how many continents are there", "number of continents"], "There are seven continents."),
    (["what is the opposite of hot"], "The opposite of hot is cold."),
    (["what is the opposite of up"], "The opposite of up is down."),
    (["what is the opposite of big"], "The opposite of big is small."),
    (["what is the opposite of fast"], "The opposite of fast is slow."),
    (["what is the first day of the week"], "The first day of the week is Sunday."),
    (["what shape has three sides", "shape with three sides"], "A triangle has three sides."),
    (["what shape has four sides", "shape with four sides"], "A square has four sides."),
    (["what language is spoken in france"], "The language spoken in France is French."),
    (["what language is spoken in japan"], "The language spoken in Japan is Japanese."),
    (["what is the currency of the united states", "currency of the usa"],
     "The currency of the United States is the dollar."),
    (["what is the speed of light"], "The speed of light is about three hundred thousand kilometers per second."),
    (["who wrote romeo and juliet"], "Romeo and Juliet was written by William Shakespeare."),
    (["what is the smallest prime number"], "The smallest prime number is two."),
]


def expand(groups):
    pairs = []
    for questions, answer in groups:
        for q in questions:
            pairs.append((q, answer))
    return pairs


def main():
    pairs = []
    pairs += arithmetic_pairs()
    pairs += expand(SMALL_TALK)
    pairs += expand(META)
    pairs += expand(FACTS)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for q, a in pairs:
        key = (q, a)
        if key not in seen:
            seen.add(key)
            unique.append((q, a))

    os.makedirs("data", exist_ok=True)
    out_path = "data/chat.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in unique:
            f.write(json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(unique)} unique (question, answer) pairs to {out_path}")
    n_arith = len(arithmetic_pairs())
    print(f"  arithmetic: {n_arith} | small talk + meta + facts: {len(unique) - n_arith}")


if __name__ == "__main__":
    main()
