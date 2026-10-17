from pathlib import Path

import jittor as jt
import numpy as np

from cyclic_flow_attention.generation import generate
from cyclic_flow_attention.utils import (
    Tokenizer,
    initialize_model,
    load_model,
    load_tokenizer,
    read_config,
    save_model,
)
from eval.scoring import Scorer

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_cache_and_likelihood(tmp_path):
    model = initialize_model(read_config(ROOT / "configs/tiny.json")).eval()
    ids = jt.array(np.arange(65, dtype=np.int32)[None] + 32)
    with jt.no_grad():
        reference = model(ids)["logits"].float32().numpy()
        prefix = model(ids[:, :64], use_cache=True)
        decoded = (
            model(ids[:, 64:], cache=prefix["cache"], use_cache=True)["logits"]
            .float32()
            .numpy()
        )
    rrms = np.linalg.norm(decoded - reference[:, -1:]) / np.linalg.norm(
        reference[:, -1:]
    )
    assert rrms < 0.02
    save_model(model, tmp_path, Tokenizer("byte"))
    loaded, directory = load_model(tmp_path)
    tokenizer = load_tokenizer(directory)
    with jt.no_grad():
        actual = loaded(ids)["logits"].float32().numpy()
    np.testing.assert_array_equal(actual, reference)
    scorer = Scorer(loaded, tokenizer, 64, 2)
    score, greedy = scorer.token_loglikelihood([([32, 33], [34, 35])])[0]
    with jt.no_grad():
        logits = loaded(jt.array([[32, 33, 34]]))["logits"].float32().numpy()[0, 1:]
    expected = (
        logits[np.arange(2), [34, 35]]
        - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1))
        - logits.max(-1)
    )
    np.testing.assert_allclose(score, expected.sum(), atol=1e-5)
    first = generate(loaded, [32, 33], 4, temperature=0.8, top_p=0.9, seed=5)
    second = generate(loaded, [32, 33], 4, temperature=0.8, top_p=0.9, seed=5)
    assert first == second and len(first) == 4


def test_padded_generation_matches_individual_prompts():
    from cyclic_flow_attention.generation import generate_batch

    model = initialize_model(
        read_config(Path(__file__).resolve().parents[1] / "configs/tiny.json")
    )
    prompts = [[10, 20, 30], list(range(4, 67))]
    expected = [generate(model, ids, max_new_tokens=8) for ids in prompts]
    actual = generate_batch(model, prompts, max_new_tokens=8)
    assert actual == expected


def test_likelihood_bos_and_continuation_boundaries(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    backend = FastTokenizer(
        WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4})
    )
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens(["<s>", "</s>"])
    backend.post_processor = TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    scorer = Scorer(
        initialize_model(read_config(ROOT / "configs/tiny.json")), Tokenizer(tmp_path)
    )
    captured = []
    scorer.token_loglikelihood = lambda pairs: (
        captured.extend(pairs) or [(0.0, True)] * len(pairs)
    )
    scorer.loglikelihood([("hello ", "world"), ("", "world"), ("<s>hello", " world")])
    assert captured == [([1, 3], [4]), ([1], [4]), ([1, 3], [4])]


def test_eval_parameter_precision():
    from eval.lm_eval import prepare_model

    model = initialize_model(read_config(ROOT / "configs/tiny.json"))
    prepare_model(model)
    params = model.state_dict()
    assert str(params["model.norm.weight"].dtype) == "bfloat16"
    assert str(params["model.layers.0.attn.o_norm.weight"].dtype) == "bfloat16"
    assert str(params["model.layers.0.attn.q_norm.weight"].dtype) == "float32"
    assert str(params["model.layers.0.attn.readout"].dtype) == "float32"
    model.bfloat16()
    assert all(str(value.dtype) == "bfloat16" for value in model.parameters())
    assert len(generate(model, [10, 20, 30], max_new_tokens=2)) == 2


def test_short_convolution_inference_cache():
    from cyclic_flow_attention.layers.cyfa import ShortConvolution

    for width in (1, 4):
        conv = ShortConvolution(64, width, bias=True).bfloat16()
        x = jt.randn((2, 19, 64)).bfloat16()
        reference, _ = conv(x)
        expected = reference.float32().numpy()
        with jt.no_grad():
            actual, state = conv(x)
            first, cache = conv(x[:, :7])
            last, cache = conv(x[:, 7:], cache)
            np.testing.assert_array_equal(actual.float32().numpy(), expected)
            np.testing.assert_array_equal(
                jt.concat([first, last], dim=1).float32().numpy(), expected
            )
            np.testing.assert_array_equal(
                cache.float32().numpy(), state.float32().numpy()
            )
            if width == 4:
                _, state = conv(x, lengths=jt.array([2, 19]))
                tail = x.float32().numpy()
                expected_state = np.stack(
                    [
                        np.concatenate([np.zeros((1, 64), np.float32), tail[0, :2]]),
                        tail[1, -3:],
                    ]
                )
                np.testing.assert_array_equal(state.float32().numpy(), expected_state)
