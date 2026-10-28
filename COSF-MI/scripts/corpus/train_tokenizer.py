"""
Train AURA's clean-corpus tokenizer: SentencePiece BPE on data/corpus_v1/tok_sample.txt (build_corpus.py sample).

Same conventions as models/tokenizer_mix16k.model so the rest of the code works unchanged: byte fallback (any Unicode
round-trips), digits split one per token, <unk>=0 <s>=1 </s>=2, no pad. 24k vocab: a compromise between web text
coverage and embedding size at small model sizes (24k × 384 = 9.2M of the 23.5M sweep model).

Usage (from COSF-MI/): .venv/bin/python scripts/corpus/train_tokenizer.py --vocab 24000 --out models/tokenizer_clean24k
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/corpus_v1/tok_sample.txt")
    p.add_argument("--vocab", type=int, default=24000)
    p.add_argument("--out", default="models/tokenizer_clean24k")
    p.add_argument("--sentences", type=int, default=4_000_000, help="lines sampled for training (RAM grows with this)")
    p.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 4))
    args = p.parse_args()
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(ROOT / args.input), model_prefix=str(ROOT / args.out), vocab_size=args.vocab, model_type="bpe",
        byte_fallback=True, split_digits=True, character_coverage=0.9995, input_sentence_size=args.sentences,
        shuffle_input_sentence=True, num_threads=args.threads, max_sentence_length=16384,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, normalization_rule_name="identity", remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True, split_by_whitespace=True,
        user_defined_symbols=["\n"])  # the sample is one paragraph per line, so "\n" must be added to stay one token
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / args.out) + ".model")
    t = "You: What is 47 + 38?\nAura: 47 + 38 = 85. Fun fact: owls can't move their eyes! 🙂 中文"
    print(f"trained {args.out}.model, vocab {sp.vocab_size()}, {time.time() - t0:.0f} s")
    print("round-trip ok:", sp.decode(sp.encode(t)) == t, "|", [sp.id_to_piece(i) for i in sp.encode(t)][:30])


if __name__ == "__main__":
    main()
