import json
import unittest

import lab
from export_plugin import base_model, build, exportable, quant, speed


class ExportFlagsTest(unittest.TestCase):
    def test_device_flag_launches_export(self):
        self.assertTrue(exportable({}))
        self.assertTrue(exportable({"flags": []}))
        self.assertTrue(exportable({"flags": ["--device /dev/kfd"]}))
        self.assertTrue(exportable({"flags": ["--device /dev/kfd", "--device /dev/dri"]}))

    def test_privileged_flag_launches_stay_out(self):
        for flag in ("--ipc host", "--security-opt seccomp=unconfined", "--ulimit memlock=-1:-1",
                     "--privileged", "--pid host", "--cap-add SYS_ADMIN", "--device /dev/mem"):
            self.assertFalse(exportable({"flags": ["--device /dev/kfd", flag]}), flag)

    def test_host_programs_and_multi_machine_stay_out(self):
        self.assertFalse(exportable({"kind": "host"}))
        self.assertFalse(exportable({"machines": 2}))
        self.assertFalse(exportable({"build": "x"}))


def recipe(path):
    return json.loads((lab.RECIPES / path).read_text())


class KvFloorTest(unittest.TestCase):
    def test_floor_is_128k_over_8gb(self):
        self.assertEqual(lab.KV_FLOOR_TOKENS, 131072)
        self.assertEqual(lab.KV_FLOOR_ABOVE_VRAM_GB, 8)

    def test_template_recipe(self):
        r = recipe("nvidia/rtx-3060-12gb/gemma-4-12b.tabbyapi.64k.json")  # 64k x 1 seq on 12 GB
        self.assertLess(lab.kv_tokens(r), lab.KV_FLOOR_TOKENS)
        self.assertTrue(lab.below_kv_floor(r))
        self.assertFalse(lab.below_kv_floor({**r, "card": "rtx-3060-ti-8gb"}))  # 8 GB is not over 8 GB
        self.assertFalse(lab.below_kv_floor({**r, "set": {**r["set"], "ctx": 131072}}))
        self.assertFalse(lab.below_kv_floor({**r, "set": {**r["set"], "ctx": 65536, "seqs": 2}}))  # 2 x (64k + 1k)

    def test_plugin_block_zero_kv_counts_context(self):
        r = recipe("intel/intel-arc-b390-64gb/qwen3.6-35b-a3b.llama.cpp.128k.json")  # kvTokens 0, ctxTokens 128k
        self.assertEqual(lab.profile(r["engine"])["plugin"]["serving"]["kvTokens"], 0)
        self.assertEqual(lab.kv_tokens(r), 131072)
        self.assertFalse(lab.below_kv_floor(r))
        r = recipe("amd/rx-6800-xt-16gb/qwen3.5-9b.llama.cpp.64k.json")  # 64k on 16 GB
        self.assertTrue(lab.below_kv_floor(r))

    def test_export_holds_the_floor(self):
        cat = json.loads((lab.ROOT / "dist" / "catalog.json").read_text())
        for version in (2, 3):
            out = json.loads(build(version))
            self.assertTrue(out["hardware"])
            for card, hw in out["hardware"].items():
                vram = cat["cards"][card]["vram_gb"] or 0
                for e in hw["recipes"]:
                    kv = e["serving"]["kvTokens"] or e["serving"]["ctxTokens"]
                    if vram * e["cards"] > lab.KV_FLOOR_ABOVE_VRAM_GB:
                        self.assertGreaterEqual(kv, lab.KV_FLOOR_TOKENS, f"v{version} {card} {e['id']}")


class DetailsTest(unittest.TestCase):
    """v3's optional fields for picking a model: quant, speed and the base model."""
    QUANTS = {
        "AWQ · FP8 W8A8 (per-channel, dynamic per-token)": "FP8",
        "AWQ · INT4 W4A16 (g32, asym), BF16 linear attention": "INT4",
        "AWQ · INT8 W8A16 (g32, sym)": "INT8",
        "GPTQ · INT4 W4A16": "INT4",
        "INT4": "INT4",
        "safetensors · BF16": "BF16",
        "BF16": "BF16",
        "FP8": "FP8",
        "safetensors · FP8": "FP8",
        "ModelOpt · NVFP4": "NVFP4",
        "nvfp4": "NVFP4",
        "safetensors · NVFP4 modelopt-mixed W4A16+FP8": "NVFP4",
        "hybrid ModelOpt+AQLM · NVFP4 hot experts + AQLM 2-bit cold experts": "NVFP4",
        "EXL3 · 3 bpw, 6-bit head": "EXL3 3.0bpw",
        "EXL3 · 3.00 bpw, head 5 bits": "EXL3 3.0bpw",
        "EXL3 · 2.05 bpw": "EXL3 2.05bpw",
        "EXL3 · 6.1 bpw": "EXL3 6.1bpw",
        "EXL3 · SC 5.00bpw H6 V6": "EXL3 5.0bpw",
        "EXL3 · 3.05 bpw · MTP (experts on NVMe, 8 GB RAM tier)": "EXL3 3.05bpw",
        "EXL3 · bpw set by the launch": "EXL3",
        "EXL3 · mixed 2/3/5-bit routed experts, native base for the rest": "EXL3 mixed",
        "GGUF · Q4_K_M": "Q4_K_M",
        "GGUF Q4_K_M": "Q4_K_M",
        "Q4_K_M": "Q4_K_M",
        "Q6_K": "Q6_K",
        "IQ3_XXS": "IQ3_XXS",
        "GGUF · IQ3_S + MTP": "IQ3_S",
        "GGUF · UD-Q4_K_XL + MTP, vision (experts in RAM)": "UD-Q4_K_XL",
        "GGUF · QAT Q4_0": "QAT Q4_0",
        "GGUF QAD-Q4_0": "QAD Q4_0",
        "GGUF · IQ2-mix": "IQ2 mix",
        "GGUF · mixed K-quants": "GGUF mixed",
        "GGUF · NVIDIA 24 GB (IQ2_XXS / NVFP4 / Q8_0)": "GGUF mixed",
        "MLX · 4-bit (g64), native MTP": "MLX 4-bit",
        "MLX · oQ4e 4-bit, native MTP, vision": "MLX 4-bit",
        "NPU2 · Q4NX": "Q4NX",
    }

    def test_quant_labels(self):
        for fmt, label in self.QUANTS.items():
            self.assertEqual(quant(fmt), label, fmt)
        self.assertIsNone(quant(None))
        self.assertIsNone(quant(""))

    def test_every_catalog_format_has_a_quant(self):
        meta = json.loads((lab.ROOT / "registry" / "models.json").read_text(encoding="utf-8"))
        formats = {b["format"] for b in meta["builds"].values()}
        formats |= {e["format"] for hw in json.loads(build(3))["hardware"].values() for e in hw["recipes"]}
        for fmt in formats:
            self.assertTrue(quant(fmt), fmt)

    def test_speed_takes_the_first_usable_measured_proof(self):
        self.assertEqual(speed({"proof": [{"at": "2026-01-01", "tps": None}]}), {})
        self.assertEqual(speed({"proof": [{"tps": 9, "withdrawn": True, "at": "2026-01-01"},
                                          {"tps": 15.7, "prefill": 330, "at": "2026-10-15"}]}),
                         {"speed": {"decode": 15.7, "prefill": 330, "at": "2026-10-15", "source": "lab"}})
        self.assertEqual(speed({"proof": [{"tps": 12, "prefill": None, "at": "2026-02-02", "reported": True}]}),
                         {"speed": {"decode": 12, "at": "2026-02-02", "source": "reported"}})
        self.assertEqual(speed({"proof": [{"tps": 30.5, "at": None, "legacy": True}]}),
                         {"speed": {"decode": 30.5, "source": "legacy"}})

    def test_base_model_follows_a_build_to_its_base(self):
        meta = json.loads((lab.ROOT / "registry" / "models.json").read_text(encoding="utf-8"))
        m = base_model({"model": "qwen3.8-27b-awq-bf16-int4"}, meta)["model"]
        self.assertEqual((m["id"], m["name"]), ("qwen3.8-27b", "Qwen3.8-27B"))
        self.assertLessEqual(set(m), {"id", "name", "about", "goodFor", "released", "hf"})
        self.assertEqual(base_model({"model": "gemma-4-26b-a4b-it-awq-4bit"}, meta)["model"]["name"], "Gemma 4 26B A4B")
        self.assertEqual(base_model({"model": "no-such-model"}, meta), {})

    def test_only_v3_carries_details(self):
        v2, v3 = json.loads(build(2)), json.loads(build(3))
        for hw in v2["hardware"].values():
            self.assertNotIn("name", hw)
            for e in hw["recipes"]:
                self.assertFalse({"model", "quant", "speed"} & set(e), e["id"])
        n1x = v3["hardware"]["rtx-spark-n1x-128gb"]
        self.assertEqual(n1x["name"], "RTX Spark N1X")
        for e in n1x["recipes"]:
            self.assertTrue({"model", "quant", "speed"} <= set(e), e["id"])
            self.assertIsInstance(e["speed"]["decode"], (int, float))


if __name__ == "__main__":
    unittest.main()
