# SPDX-License-Identifier: MIT
"""Likelihood scoring with Jittor model execution and native log-softmax."""

import jittor as jt
import numpy as np


class Scorer:
    def __init__(self, model, tokenizer, max_length=2048, batch_size=1):
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length, self.batch_size = max_length, batch_size

    def token_loglikelihood(self, pairs):
        outputs = [(0.0, True)] * len(pairs)
        order = sorted(
            range(len(pairs)),
            key=lambda i: len(pairs[i][0]) + len(pairs[i][1]),
            reverse=True,
        )
        with jt.no_grad():
            for begin in range(0, len(order), self.batch_size):
                group = order[begin : begin + self.batch_size]
                prepared = []
                for index in group:
                    context, continuation = pairs[index]
                    if not continuation:
                        continue
                    if len(continuation) > self.max_length:
                        raise ValueError("Continuation exceeds evaluation context")
                    context = context or [self.tokenizer.bos_id]
                    sequence = (context + continuation)[-(self.max_length + 1) :]
                    prepared.append(
                        (
                            index,
                            sequence[:-1],
                            continuation,
                            len(sequence) - 1 - len(continuation),
                        )
                    )
                if not prepared:
                    continue
                length = min(
                    self.max_length,
                    (max(len(row[1]) for row in prepared) + 63) // 64 * 64,
                )
                ids = np.full((len(prepared), length), self.tokenizer.eos_id, np.int32)
                rows, positions, targets = [], [], []
                for row, (_, tokens, continuation, start) in enumerate(prepared):
                    ids[row, : len(tokens)] = tokens
                    rows.extend([row] * len(continuation))
                    positions.extend(range(start, start + len(continuation)))
                    targets.extend(continuation)
                hidden, _ = self.model.model(jt.array(ids))
                selected = hidden[
                    jt.array(np.array(rows, np.int32)),
                    jt.array(np.array(positions, np.int32)),
                ]
                # Only continuation positions need the vocabulary projection.
                count = len(targets)
                padding = (-count) % 64
                if padding:
                    selected = jt.concat(
                        [
                            selected,
                            jt.zeros(
                                (padding, selected.shape[-1]), dtype=selected.dtype
                            ),
                        ],
                        dim=0,
                    )
                logits = self.model.lm_head(selected)
                greedy, maxima = logits.argmax(-1)
                maxima = maxima.float32()
                normalizer = (
                    (logits.float32() - maxima.unsqueeze(-1)).exp().sum(-1).log()
                )
                chosen = logits[
                    jt.arange(count).int32(), jt.array(np.array(targets, np.int32))
                ].float32()
                scores = (chosen - maxima[:count] - normalizer[:count]).numpy()
                greedy = greedy[:count].numpy()
                offset = 0
                for index, _, continuation, _ in prepared:
                    end = offset + len(continuation)
                    outputs[index] = (
                        float(scores[offset:end].sum(dtype=np.float32)),
                        bool(np.array_equal(greedy[offset:end], continuation)),
                    )
                    offset = end
                if begin and begin % (100 * self.batch_size) == 0:
                    print(
                        f"Likelihood: {min(begin + self.batch_size, len(order))}/{len(order)} requests",
                        flush=True,
                    )
        return outputs

    def encode(self, text):
        # lm-eval 0.4.12 defaults to the checkpoint tokenizer's BOS convention.
        bos = (
            self.tokenizer.backend.id_to_token(self.tokenizer.bos_id)
            if self.tokenizer.backend
            else None
        )
        return self.tokenizer.encode(
            text, add_special_tokens=not (bos and text.startswith(bos))
        )

    def loglikelihood(self, requests):
        pairs = []
        for context, continuation in requests:
            if not context:
                tokens = self.tokenizer.encode(continuation)
                if tokens and tokens[0] == self.tokenizer.bos_id:
                    pairs.append((tokens[:1], tokens[1:]))
                else:
                    pairs.append(([self.tokenizer.bos_id], tokens))
                continue
            count = len(context) - len(context.rstrip())
            if count:
                continuation, context = (
                    context[-count:] + continuation,
                    context[:-count],
                )
            context_ids = self.encode(context)
            combined = self.encode(context + continuation)
            pairs.append((context_ids, combined[len(context_ids) :]))
        return self.token_loglikelihood(pairs)

    def rolling_loglikelihood(self, texts):
        pairs, owners = [], []
        for owner, text in enumerate(texts):
            tokens = self.encode(text)
            for start in range(0, len(tokens), self.max_length):
                context = tokens[max(0, start - self.max_length) : start] or [
                    self.tokenizer.bos_id
                ]
                pairs.append((context, tokens[start : start + self.max_length]))
                owners.append(owner)
        totals = [0.0] * len(texts)
        for owner, (score, _) in zip(owners, self.token_loglikelihood(pairs)):
            totals[owner] += score
        return totals
