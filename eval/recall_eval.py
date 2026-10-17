# SPDX-License-Identifier: MIT
# Prompt conventions adapted from CyclicFlowAttention's MIT prefix-recall tasks.
"""Zero-shot prefix recall, with the same contains metric and prompts as CyFA."""

import argparse
import json
import re
from pathlib import Path

from cyclic_flow_attention.generation import generate_batch
from cyclic_flow_attention.utils import load_model, load_tokenizer, setup_cuda

TASKS = {
    "based_fda": "hazyresearch/based-fda",
    "based_swde": "hazyresearch/based-swde-v2",
    "based_squad": "hazyresearch/based-squad",
    "based_nq_2048": "hazyresearch/based_nq_2048",
    "based_triviaqa": "hazyresearch/based_triviaqa",
    "based_drop": "hazyresearch/based_drop",
}


def format_document(task, document):
    doc = dict(document)
    if task == "based_squad":
        return doc["text"].strip(), [doc["value"]]
    if task in ("based_fda", "based_swde"):
        text, key = doc["text"], doc["key"]
        if task == "based_swde":
            text = text.split("Summary of information above...")[0]
        while text.lower().endswith((key + ":").lower()):
            text = text[: -len(key) - 1]
        if task == "based_fda":
            text = text.strip("\n").strip(".")
            if not text.endswith("."):
                text += "."
        else:
            text = text.strip("\n").strip().strip("\n").strip().strip(".") + "."
        return text + " " + key[0].upper() + key[1:] + ":", [doc["value"]]
    context, question = doc["context"].strip(), doc["question"].strip()
    if task == "based_triviaqa":
        context = re.sub(r"\[PAR\]|\[DOC\]|\[TLE\]", "", context)
    if task == "based_nq_2048":
        if context.endswith(question):
            context = context[: -len(question)]
    else:
        while context.lower().endswith(question.lower()) and question:
            context = context[: -len(question)]
    return context.strip().strip(".") + ". " + question, doc["answers"]


def prepare_document(task, document, tokenizer, context_length=2048, answer_length=48):
    """Apply the original prefix-recall harness's answer-centred context window."""
    doc = dict(document)
    context_key = (
        "text" if task in ("based_fda", "based_swde", "based_squad") else "context"
    )
    answer_keys = {
        "based_fda": ("key", "value"),
        "based_swde": ("key", "value"),
        "based_squad": ("value",),
        "based_nq_2048": ("answers",),
        "based_triviaqa": (),
        "based_drop": (),
    }[task]
    desired_length = context_length - answer_length
    if desired_length < 1:
        raise ValueError("Context length must exceed answer length")
    context = doc[context_key].strip()
    tokens = tokenizer.encode(context, add_special_tokens=True)
    answer_pos = -1
    for key in answer_keys:
        answer = doc[key]
        if isinstance(answer, list):
            answer = answer[0]
        if len(answer) <= 1:
            return None
        match = re.search(re.escape(answer), context, re.IGNORECASE)
        if match and (answer_pos == -1 or match.start() < answer_pos):
            answer_pos = match.start()
    if answer_pos < 0:
        for answer in doc.get("alt_answers", []):
            match = re.search(re.escape(answer), context, re.IGNORECASE)
            if match:
                answer_pos = match.start()
                break
    if not answer_keys:
        answer_pos = 0
    prefix = tokenizer.encode(context[:answer_pos], add_special_tokens=True)
    answer_token = len(prefix)
    if answer_pos == -1 and "tok_pos" in doc and len(prefix) > desired_length:
        answer_token = doc["tok_pos"]
    start = max(0, answer_token - desired_length // 2)
    end = min(len(tokens), answer_token + desired_length - (answer_token - start))
    doc[context_key] = tokenizer.decode(tokens[start:end])
    prompt, answers = format_document(task, doc)
    # The causal harness also reserves generation space in its final prompt.
    ids = tokenizer.encode(prompt)[-desired_length:] or [tokenizer.bos_id]
    return ids, answers


def contains_score(prediction, answers):
    return float(
        any(
            re.search(re.escape(answer), prediction, re.IGNORECASE)
            for answer in answers
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--answer-length", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--input", help="Offline JSONL records with task and task-specific fields"
    )
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    if (
        any(task not in TASKS for task in tasks)
        or min(args.context_length, args.answer_length, args.batch_size) < 1
    ):
        parser.error("Unknown task or invalid context/answer length")
    setup_cuda()
    model, directory = load_model(args.model_path)
    # The original recall CLI explicitly casts every parameter to BF16.
    model.bfloat16()
    tokenizer = load_tokenizer(directory, args.tokenizer)
    result, samples = {}, []
    for task in tasks:
        if args.input:
            records = (
                json.loads(line)
                for line in Path(args.input).read_text().split("\n")
                if line.strip()
            )
            records = (row for row in records if row["task"] == task)
        else:
            from datasets import load_dataset

            records = load_dataset(TASKS[task], split="validation", streaming=True)
        prepared_records = []
        for index, doc in enumerate(records):
            if args.limit is not None and index >= args.limit:
                break
            prepared = prepare_document(
                task, doc, tokenizer, args.context_length, args.answer_length
            )
            if prepared is not None:
                ids, answers = prepared
                prepared_records.append((index, ids, answers))
        prepared_records.sort(key=lambda row: len(row[1]), reverse=True)
        scores = []
        for begin in range(0, len(prepared_records), args.batch_size):
            batch = prepared_records[begin : begin + args.batch_size]
            outputs = generate_batch(
                model,
                [row[1] for row in batch],
                args.answer_length,
                tokenizer.eos_id,
                on_token=lambda ids: "\n" in tokenizer.decode(ids),
            )
            for (index, ids, answers), values in zip(batch, outputs):
                prediction = tokenizer.decode(values).split("\n")[0]
                score = contains_score(prediction, answers)
                scores.append(score)
                samples.append(
                    dict(
                        task=task,
                        index=index,
                        prediction=prediction,
                        answers=answers,
                        contains=score,
                    )
                )
            print(
                f"{task}: {len(scores)}/{len(prepared_records)} examples, contains={sum(scores) / len(scores):.4f}",
                flush=True,
            )
        if not scores:
            raise ValueError("No evaluation documents for " + task)
        result[task] = dict(contains=sum(scores) / len(scores), samples=len(scores))
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                results=result,
                samples=samples,
                config=vars(args),
                parameter_dtype="bfloat16",
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
