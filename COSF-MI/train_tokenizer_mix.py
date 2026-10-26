"""
Train the 16k tokenizer for the data/mix pretraining set.

Unlike tokenizer_8k (TinyStories only) it keeps newlines and indentation
(needed for code), splits numbers into single digits (helps arithmetic),
and falls back to bytes instead of <unk> for rare characters.

Usage:  python train_tokenizer_mix.py
"""

import random
from pathlib import Path

import sentencepiece as spm

SRC = Path("data/mix")
SAMPLE = Path("data/mix/tokenizer_sample.txt")


def build_sample(per_source_chars=25e6, seed=0):
    rng = random.Random(seed)
    with open(SAMPLE, "w", encoding="utf-8") as out:
        for name in ["wiki", "web", "textbook", "code", "stories", "math"]:
            docs = (SRC / f"{name}_train.txt").read_text(encoding="utf-8").split("\n<|endoftext|>\n")
            rng.shuffle(docs)
            n = 0
            for d in docs:
                # one doc per line; real newlines become the "\n" user symbol below
                out.write(d.replace("\n", " ") + "\n")
                n += len(d)
                if n >= per_source_chars:
                    break
            print(f"  {name}: {n / 1e6:.0f}M chars")


def main():
    print("Building tokenizer sample...")
    build_sample()
    print("Training sentencepiece...")
    spm.SentencePieceTrainer.train(
        input=str(SAMPLE),
        model_prefix="models/tokenizer_mix16k",
        vocab_size=16000,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        user_defined_symbols=["\n", "<|endoftext|>", "    ", "\t"],
        max_sentence_length=1 << 20,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=16,
    )
    sp = spm.SentencePieceProcessor(model_file="models/tokenizer_mix16k.model")
    tests = ["def f(x):\n    return x + 1\n\nprint(f(5))",
             "The Eiffel Tower, built in 1889, is 330 metres tall.",
             "Q: What is 47 + 38?\nA: 47 + 38 = 85.",
             "Once upon a time, there was a little girl named Lily."]
    for t in tests:
        ids = sp.encode(t)
        print(f"{len(ids):3d} tokens  roundtrip={'OK' if sp.decode(ids) == t else repr(sp.decode(ids))}")
    print("Done.")


if __name__ == "__main__":
    main()
