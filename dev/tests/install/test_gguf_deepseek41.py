"""DeepSeek-V4.1's GGUFs (general.architecture deepseek41) and their Engram
file (deepseek41-engram): the configuration, tokenizer and tensor screen the
installer derives from the headers of the IQ1_M files of
ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (revision
653a9874aa38c356233c5f3cc8c96b19a43b57eb), the Engram file it takes beside
them, and the selection of their split files at the repository's root.
Their scalar metadata, arrays and tensor table, read by HTTP range requests,
are fixtures (dev/tests/fixtures/model-configs/deepseek-v4.1-flash), and so is the
config.json of the release they were made from,
deepseek-ai/DeepSeek-V4.1-Flash-1017 (revision
1bae0e1088993b9b82de08f59de5be5a34e42170)."""

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, pre_tokenizers

from dev.tests import fixture_files
from dev.tests.installer_fixtures import (
    CONFIGS,
    CONTROL,
    DEEPSEEK,
    GGML,
    NORMAL,
    USER_DEFINED,
)
from install import assembly, gguf, local, models, upstream

FIXTURES = CONFIGS / DEEPSEEK.name.lower()
TEMPLATE = (CONFIGS.parent / "chat_templates/deepseek41.jinja").read_text()
ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
SOURCE = "1bae0e1088993b9b82de08f59de5be5a34e42170"
VOCABULARY = 129280
# DeepSeek's added tokens the tests read, at their IDs and of the types the
# real files give them: its control tokens, its roles among them, special,
# and the tags its template writes in text, user-defined and not special.
ADDED = {
    0: ("<｜begin▁of▁sentence｜>", CONTROL),
    1: ("<｜end▁of▁sentence｜>", CONTROL),
    2: ("<｜▁pad▁｜>", CONTROL),
    128799: ("<｜System｜>", CONTROL),
    128803: ("<｜User｜>", CONTROL),
    128804: ("<｜Assistant｜>", CONTROL),
    128821: ("<think>", USER_DEFINED),
    128822: ("</think>", USER_DEFINED),
    128825: ("｜DSML｜", USER_DEFINED),
}
# The keys of DeepSeek's config.json no GGUF header states: training
# settings, the router's top-k method its correction bias implies, and the
# DSpark and MTP layers the GGUFs leave out.
UNSTATED = {
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "initializer_range",
    "tie_word_embeddings",
    "topk_method",
    "use_cache",
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_noise_token_id",
    "dspark_num_experts_per_tok",
    "dspark_target_layer_ids",
}


def header_values():
    """The first file's metadata: the scalars and arrays of the fixtures, its
    booleans, which scalar_metadata keeps as 1, and a tokenizer of DeepSeek's
    129280 tokens, the added tokens the tests read at their IDs."""
    typed = json.loads((FIXTURES / "gguf-metadata.json").read_text())
    arrays = json.loads((FIXTURES / "gguf-arrays.json").read_text())
    tokens = [f"[base{index}]" for index in range(VOCABULARY)]
    types = [NORMAL] * VOCABULARY
    base = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["12", "123"]
    tokens[3 : 3 + len(base)] = base
    for index, (text, kind) in ADDED.items():
        tokens[index], types[index] = text, kind
    return (
        typed["unsigned"]
        | typed["float"]
        | typed["string"]
        | arrays
        | {
            "deepseek41.expert_weights_norm": True,
            "deepseek41.engram.present": True,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "joyai-llm",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": ["1 2", "12 3"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.padding_token_id": 1,
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.chat_template": TEMPLATE,
        }
    )


def tensor_table():
    """The real files' tensors: name, extents and type, in their order."""
    return [
        (name, extents, GGML[kind])
        for name, (kind, extents) in json.loads(
            (FIXTURES / "gguf-tensors.json").read_text()
        ).items()
    ]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class DeepSeek41GgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None, tensors=(), name="deepseek41.gguf"):
        path = fixture_files.write_gguf(
            self.root / name,
            header_values() if values is None else values,
            [(tensor, extents, kind, b"") for tensor, extents, kind in tensors],
        )
        return gguf.Metadata(path, tensors=True)

    def engram(self, values=None, tables=None):
        values = {
            "general.architecture": gguf.ENGRAM_ARCHITECTURE,
            "deepseek41.engram.source": SOURCE,
        } | (values or {})
        tables = (
            [
                ("blk.1.engram_embd.weight", [256, 384006168], GGML["IQ4_NL"]),
                ("blk.14.engram_embd.weight", [256, 384016682], GGML["IQ4_NL"]),
            ]
            if tables is None
            else tables
        )
        return self.metadata(values, tables, name="engram.gguf")

    def test_the_derived_config_is_deepseek_s_own(self):
        config = gguf.model_config(self.metadata())
        release = json.loads((FIXTURES / "config.json").read_text())
        self.assertEqual(config["model_type"], release["model_type"])
        text, expected = config["text_config"], release["text_config"]
        # Every key of the release's text config but those no header states,
        # each of the header's value (its floats at f32), and the routed
        # experts again as num_experts, by which the engine plans a GGUF MoE.
        self.assertEqual(set(expected) - set(text), UNSTATED)
        self.assertEqual(set(text) - set(expected), {"num_experts", "eos_token_id"})
        for key in set(expected) & set(text):
            value = expected[key]
            if isinstance(value, float):
                self.assertEqual(text[key], f32(value), key)
            elif key == "rope_scaling":
                self.assertEqual(
                    text[key],
                    {
                        k: f32(v) if isinstance(v, float) else v
                        for k, v in value.items()
                    },
                )
            else:
                self.assertEqual(text[key], value, key)
        self.assertEqual(text["num_experts"], expected["n_routed_experts"])
        self.assertEqual(text["eos_token_id"], release["eos_token_id"])

    def test_metadata_the_config_cannot_state_is_refused(self):
        for key, value, refusal in (
            ("deepseek41.expert_scoring_func", "sigmoid", "expert scoring function"),
            ("deepseek41.rope.scaling.type", "linear", "RoPE scaling"),
            ("deepseek41.swiglu_clamp_shexp", [7.0] * 40, "SwiGLU clamps"),
            ("deepseek41.kv_source_layer_ids", [8, 2], "kv_source_layer_ids"),
            ("deepseek41.engram.layer_ids", [1, 40], "engram.layer_ids"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    gguf.model_config(self.metadata(header_values() | {key: value}))

    def test_the_derived_tokenizer_splits_as_deepseek_s(self):
        files = gguf.tokenizer_files(self.metadata())
        backend = json.loads(files["tokenizer/tokenizer.json"])
        # DeepSeek's three splits, then byte-level BPE, without a normalizer.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(
            [part["type"] for part in backend["pre_tokenizer"]["pretokenizers"]],
            ["Split", "Split", "Split", "ByteLevel"],
        )
        tokenizer = Tokenizer.from_str(files["tokenizer/tokenizer.json"].decode())
        pieces = tokenizer.pre_tokenizer.pre_tokenize_str
        for text, spans in (
            # Digits in runs of three; a run of Han and katakana apart.
            ("12345", [(0, 3), (3, 5)]),
            ("中文テストabc", [(0, 5), (5, 8)]),
            # A punctuation mark opens the letters after it, and a run of
            # punctuation keeps its newlines.
            (".dotfile", [(0, 8)]),
            ("a, b!\n\n", [(0, 1), (1, 2), (2, 4), (4, 7)]),
        ):
            self.assertEqual([span for _, span in pieces(text)], spans, text)
        # Control tokens are special; the tags the template writes are text.
        self.assertEqual(
            tokenizer.encode("<｜User｜><think>｜DSML｜<｜end▁of▁sentence｜>").ids,
            [128803, 128821, 128825, 1],
        )
        self.assertEqual(files["tokenizer/chat_template.jinja"].decode(), TEMPLATE)

    def test_the_screen_reads_every_tensor_of_the_real_files(self):
        table = tensor_table()
        metadata = self.metadata(tensors=table)
        gguf.require_loadable(metadata)
        loaded = gguf.loaded_tensors(metadata)
        self.assertEqual(set(loaded), {name for name, _, _ in table})
        # The KV sources compress, those of a ratio of two through a pooling
        # gate; the index sources score; the Engram layers project rows.
        for layer in (2, 8, 14):
            self.assertIn(f"blk.{layer}.attn_compressor_gate.weight", loaded)
        self.assertIn("blk.20.attn_compressor_kv.weight", loaded)
        self.assertNotIn("blk.20.attn_compressor_gate.weight", loaded)
        self.assertNotIn("blk.3.attn_compressor_kv.weight", loaded)
        self.assertIn("blk.24.indexer.attn_q_b.weight", loaded)
        self.assertNotIn("blk.24.indexer.attn_k.weight", loaded)
        self.assertEqual(
            {name for name in loaded if "engram_wkv" in name},
            {"blk.1.engram_wkv.weight", "blk.14.engram_wkv.weight"},
        )
        self.assertEqual(loaded["engram_primes.weight"], {"I64"})

    def test_the_screen_refuses_what_the_loader_does_not_read(self):
        table = tensor_table()
        without_gate = [t for t in table if t[0] != "blk.2.attn_compressor_gate.weight"]
        sinks_q4 = [
            (name, extents, GGML["Q4_K"] if name == "blk.0.attn_sinks.weight" else kind)
            for name, extents, kind in table
        ]
        for tensors, refusal in (
            (without_gate, r"attn_compressor_gate\.weight missing \(1 tensor\)"),
            (sinks_q4, r"attn_sinks\.weight Q4_K \(1 tensor\)"),
        ):
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    gguf.require_loadable(self.metadata(tensors=tensors))

    def test_the_engram_file_is_a_plain_name_beside_the_target(self):
        self.assertEqual(gguf.engram_file(self.metadata()), ENGRAM)
        for name in (
            "../" + ENGRAM,
            "/tmp/" + ENGRAM,
            "sub/" + ENGRAM,
            "sub\\" + ENGRAM,
            "." + ENGRAM,
            "engram.bin",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(models.ModelError, "unsafe GGUF metadata"):
                    gguf.engram_file(
                        self.metadata(
                            header_values() | {"deepseek41.engram.file": name}
                        )
                    )

    def test_the_engram_file_must_be_the_backbone_s(self):
        header = self.metadata()
        gguf.require_engram(header, self.engram())
        for engram, refusal in (
            (
                self.engram({"general.architecture": "deepseek41"}),
                "not a deepseek41-engram",
            ),
            (self.engram({"deepseek41.engram.source": "0" * 40}), "another backbone"),
            (
                self.engram(
                    tables=[("blk.1.engram_embd.weight", [256, 8], GGML["IQ4_NL"])]
                ),
                "one table",
            ),
            (
                self.engram(
                    tables=[
                        ("blk.1.engram_embd.weight", [256, 8], GGML["IQ4_NL"]),
                        ("blk.14.engram_embd.weight", [256, 8], GGML["I64"]),
                    ]
                ),
                "one table",
            ),
        ):
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    gguf.require_engram(header, engram)

    def test_an_engram_file_is_no_target(self):
        with self.assertRaisesRegex(
            models.ModelError, "select a variant of the backbone"
        ):
            gguf.model_config(self.engram())

    def test_split_files_at_the_root_select_their_variant(self):
        def parts(variant, count=4):
            return [
                f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-{variant}-{k:05d}-of-{count:05d}.gguf"
                for k in range(1, count + 1)
            ]

        files = {
            *parts("IQ1_M"),
            *parts("IQ2_XXS"),
            *parts("IQ2_XS"),
            ENGRAM,
            "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf",
            "README.md",
        }
        for variant in ("IQ1_M", "IQ2_XXS", "IQ2_XS"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    upstream.select_gguf(files, variant), (parts(variant), False)
                )
        incomplete = files - {parts("IQ1_M")[2]}
        with self.assertRaisesRegex(models.ModelError, "incomplete"):
            upstream.select_gguf(incomplete, "IQ1_M")
        # The Engram file alone ends in its own variant; the screen then
        # refuses it as no target (test_an_engram_file_is_no_target).
        self.assertEqual(upstream.select_gguf(files, "IQ4_NL"), ([ENGRAM], True))

    def test_local_files_prepare_with_their_engram_file_and_no_draft(self):
        # The engine knows the family's configuration once it loads the
        # family; until then the installer's check stands in for its own.
        directory = self.root / "DeepSeek-V4.1-Flash-GGUF"
        directory.mkdir()
        tensors = tensor_table()
        names = [f"DeepSeek-V4.1-Flash-IQ1_M-{k:05d}-of-00002.gguf" for k in (1, 2)]
        for number, name in enumerate(names):
            split = {
                "split.count": 2,
                "split.tensors.count": len(tensors),
                "split.no": number,
            }
            self.metadata(
                (header_values() if number == 0 else {}) | split,
                tensors[number::2],
                name=f"{directory.name}/{name}",
            )
        self.engram()
        (self.root / "engram.gguf").rename(directory / ENGRAM)
        with (
            mock.patch.object(upstream, "check_model", return_value=DEEPSEEK),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for target in (directory / names[1], directory):
                with self.subTest(target=target.name):
                    record = assembly.verify(
                        local.prepare_local_model(target, models_dir=self.root / "m")
                    )
                    self.assertEqual(record["family"], "DeepSeek-V4.1-Flash")
                    self.assertEqual(
                        record["sources"]["draft"],
                        {"repo": models.NO_DRAFT, "revision": None},
                    )
                    self.assertLessEqual(
                        {*("target/" + name for name in names), "engram/" + ENGRAM},
                        set(record["files"]),
                    )
                    self.assertFalse(
                        any(n.startswith("draft/") for n in record["files"])
                    )


if __name__ == "__main__":
    unittest.main()
