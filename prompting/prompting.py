"""Classify DiCo-NLI text pairs by prompting a generative language model.

The model gets a prompt template filled with Text 1 and Text 2 and has to answer
with one of the four labels. The answers are scored on the dev set (Weighted F1,
SoftCons and HardCons) and, if --test is given, also on the test set.

Example:
    python prompting.py --model Qwen/Qwen3.5-9B --prompt prompts/0-shot.txt --dev_sample 80
"""

from transformers import pipeline
import torch
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]

# The label a pair should get when Text 1 and Text 2 are swapped (used for SoftCons).
# NEGATIVE_OTHER is not in here, so a NEGATIVE_OTHER pair never counts as consistent.
REVERSE_LABEL = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def create_arg_parser():
    """Create the command-line argument parser and return the parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_file", default="../data/dev.csv",
                        help="Development CSV (default: ../data/dev.csv)")
    parser.add_argument("--test", nargs="?", const="../data/test.csv",
                        type=Path, help="Optionally predict a test CSV (default: data/test.csv)")
    parser.add_argument("-o", "--output_file",
                        default="predictions/attempt1_Baseline.csv",
                        help="Test prediction CSV (default: predictions/attempt1_Baseline.csv)")
    parser.add_argument("-m", "--model",
                            default="Qwen/Qwen3.5-0.8B",
                            help="Model string from Huggingface (default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("-p", "--prompt", default="prompts/baseline.txt",
                        help="Prompt template with {text1} and {text2} placeholders (default: prompts/baseline.txt)")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Number of prompts generated at once (default: 32)")
    parser.add_argument("--dev_sample", type=int, default=None,
                        help="Only use this many dev rows, sampled by pair (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the dev sample (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_corpus(corpus_file):
    """Read a DiCo-NLI CSV file.

    Returns a list of dicts, one per row, with the keys instance_id, pair_id,
    text1, text2 and label.
    """
    with open(corpus_file, encoding="utf-8-sig", newline="") as in_file:
        return list(csv.DictReader(in_file))


def group_by_pair(rows):
    """Group original and reversed examples by source pair.

    Returns a dict from pair_id to the rows of that pair. Most pairs have two rows
    (Text 1/Text 2 and the swapped version), some have only one.
    """
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    return pairs


def sample_rows(rows, n, seed):
    """Shuffle whole pairs (so both directions stay together) and keep the first n rows.

    Shuffling pairs instead of single rows keeps most pairs complete, so SoftCons
    and HardCons can still be computed on the sample.
    """
    pairs = list(group_by_pair(rows).values())
    random.Random(seed).shuffle(pairs)
    # Flatten the shuffled pairs back into one list of rows
    return [row for pair in pairs for row in pair][:n]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def read_template(template_file):
    """Read a prompt template from the prompts directory."""
    return Path(template_file).read_text(encoding="utf-8").strip()


def create_prompt(row, template):
    """Fill the template with the text pair and wrap it as a chat message."""
    prompt = template.format(text1=row["text1"], text2=row["text2"])
    return [{"role": "user", "content": prompt}]


def predict_labels(pipeline, rows, template, batch_size):
    """Prompt the model in batches and return the parsed labels and raw outputs.

    Returns two lists of the same length as rows: the predicted labels and the
    text the model generated (kept for debugging).
    """
    prompts = [create_prompt(row, template) for row in rows]
    outputs = pipeline(prompts,
                       batch_size=batch_size,
                       max_new_tokens=10,          # enough for the longest label
                       do_sample=False,            # greedy decoding, so results are reproducible
                       return_full_text=False,     # only return the answer, not the prompt
                       tokenizer_encode_kwargs={"enable_thinking": False})  # no thinking for Qwen
    raw_outputs = [output[0]["generated_text"] for output in outputs]
    predictions = [parse_label(raw) for raw in raw_outputs]
    return predictions, raw_outputs


def parse_label(output_text):
    """Return the first label from LABELS that appears in the model output.

    Falls back to NEGATIVE_OTHER if the output contains none of the labels.
    """
    output_text = output_text.upper()
    for label in LABELS:
        if label in output_text:
            return label
    return "NEGATIVE_OTHER"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(rows, predictions):
    """Calculate the official Weighted F1, SoftCons, and HardCons scores.

    SoftCons: share of pairs where the prediction for the swapped version is the
    reverse of the prediction for the original (gold labels are not used).
    HardCons: share of pairs where both directions are predicted correctly.
    Both only use pairs that have two rows.
    """
    gold_labels = [row["label"] for row in rows]
    weighted_f1 = f1_score(gold_labels, predictions, labels=LABELS, average="weighted", zero_division=0)
    prediction_by_id = {row["instance_id"]: prediction for row, prediction in zip(rows, predictions)}
    reversible_pairs = [pair for pair in group_by_pair(rows).values() if len(pair) == 2]

    # Count consistent pairs (soft) and fully correct pairs (hard)
    soft_correct = 0
    hard_correct = 0
    for first, second in reversible_pairs:
        first_prediction = prediction_by_id[first["instance_id"]]
        second_prediction = prediction_by_id[second["instance_id"]]
        if REVERSE_LABEL.get(first_prediction) == second_prediction:
            soft_correct += 1
        if (first_prediction == first["label"] and second_prediction == second["label"]):
            hard_correct += 1

    # Final scores
    soft_cons = soft_correct / len(reversible_pairs)
    hard_cons = hard_correct / len(reversible_pairs)
    return weighted_f1, soft_cons, hard_cons


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------

def write_predictions(output_file, rows, predictions):
    """Write predictions in the official instance_id,label format."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["instance_id", "label"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"instance_id": row["instance_id"], "label": prediction})


def write_raw_outputs(output_file, rows, predictions, raw_outputs):
    """Write gold label, parsed label and raw model output per row (for debugging)."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["instance_id", "gold", "predicted", "raw"])
        writer.writeheader()
        for row, prediction, raw in zip(rows, predictions, raw_outputs):
            writer.writerow({"instance_id": row["instance_id"], "gold": row["label"],
                             "predicted": prediction, "raw": raw})


# ---------------------------------------------------------------------------
# Not used in this script (left over from the baseline)
# ---------------------------------------------------------------------------

LABEL2ID = {"EQUIVALENCE": 0, "FORWARD_ENTAILMENT": 1, "BACKWARD_ENTAILMENT": 2, "NEGATIVE_OTHER": 3}
ID2LABEL = {0: "EQUIVALENCE", 1: "FORWARD_ENTAILMENT", 2: "BACKWARD_ENTAILMENT", 3: "NEGATIVE_OTHER"}


def make_documents(rows):
    """Turn each text pair into premise and hypothesis-marked unigrams."""
    documents = []
    for row in rows:
        text1 = ["TEXT1_" + token.lower() for token in row["text1"].split()]
        text2 = ["TEXT2_" + token.lower() for token in row["text2"].split()]
        documents.append(text1 + text2)
    return documents


def identity(value):
    """Return already-tokenized input unchanged."""
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = create_arg_parser()
    template = read_template(args.prompt)

    # Load the model; bfloat16 where the GPU supports it (A100), else float16 (V100)
    generator = pipeline(
        task="text-generation",
        model=args.model,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    # Decoder-only models must be padded on the left when generating in batches
    generator.tokenizer.padding_side = "left"
    # Some tokenizers (e.g. Mistral) have no pad token, which batching needs
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token

    # Predict the dev set (or a sample of it)
    dev_rows = read_corpus(args.dev_file)
    if args.dev_sample:
        dev_rows = sample_rows(dev_rows, args.dev_sample, args.seed)
    dev_predictions, dev_raw = predict_labels(generator, dev_rows, template, args.batch_size)

    # Print every dev example with its gold label, prediction and raw output
    for row, prediction, raw in zip(dev_rows, dev_predictions, dev_raw):
        print(f"{row['text1']} | {row['text2']} | gold={row['label']} | pred={prediction} | raw={raw!r}")

    # Save the raw dev outputs next to the prediction file and print the scores
    raw_file = Path(args.output_file).with_suffix("")
    write_raw_outputs(f"{raw_file}_dev_raw.csv", dev_rows, dev_predictions, dev_raw)
    weighted_f1, soft_cons, hard_cons = evaluate(dev_rows, dev_predictions)
    print("Official scores on dev:")
    print(f"  Weighted F1: {100 * weighted_f1:.1f}")
    print(f"  SoftCons:    {100 * soft_cons:.1f}")
    print(f"  HardCons:    {100 * hard_cons:.1f}")

    # Do prediction on test if asked
    if args.test:
        test_rows = read_corpus(args.test)
        test_predictions, test_raw = predict_labels(generator, test_rows, template, args.batch_size)
        write_predictions(args.output_file, test_rows, test_predictions)
        write_raw_outputs(f"{raw_file}_test_raw.csv", test_rows, test_predictions, test_raw)
        print(f"Predictions written to {args.output_file}")

        weighted_f1, soft_cons, hard_cons = evaluate(test_rows, test_predictions)
        print("Official scores on test:")
        print(f"  Weighted F1: {100 * weighted_f1:.1f}")
        print(f"  SoftCons:    {100 * soft_cons:.1f}")
        print(f"  HardCons:    {100 * hard_cons:.1f}")
