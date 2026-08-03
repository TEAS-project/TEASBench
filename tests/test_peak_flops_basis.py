import unittest

from postprocessing.moe_cost_metrics.compute_sparsity_metrics import (
    PEAK_FLOPS_BASIS,
    PEAK_FLOPS_DICT,
    get_peak_flops,
)

# Vendor with-2:4-sparsity datasheet figures (FLOP/s), per GPU. NVIDIA quotes these; the
# catalog must hold HALF of each (the dense basis), because S-MFU divides measured FLOPs by
# the peak and a sparse-basis denominator would silently halve every published MFU. A
# regression to datasheet-sparse figures is exactly the failure this test exists to catch.
NVIDIA_SPARSE = {
    "bfloat16": {
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 4500e12,
        "NVIDIA-B300-269GB": 4500e12,
    },
    "fp8": {
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-B200-183GB": 9000e12,
        "NVIDIA-B300-269GB": 9000e12,
    },
    "int8": {
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-HBM3-80GB": 3958e12,
        "NVIDIA-H200-141GB": 3958e12,
        "NVIDIA-B200-183GB": 9000e12,
        # Blackwell Ultra's INT8 path is ~29x narrower than B200's, not a typo for 9 POPS.
        "NVIDIA-B300-269GB": 307e12,
    },
    "fp4": {"NVIDIA-B200-183GB": 18000e12},
}
# The one NVIDIA row printed as `sparse | dense` rather than sparse alone, so it is NOT
# halved. Its own board total disagrees — 108 PFLOPS across 8 GPUs implies 13.5 — and halving
# the 18000 sparse figure would give 9000, quietly restoring the slower B200 rate for a faster
# part. Both wrong answers are one plausible edit away, which is why this is pinned.
NVIDIA_PRINTED_DENSE = {"fp4": {"NVIDIA-B300-269GB": 14000e12}}
# AMD names dense and with-sparsity as separate rows, so these match the published dense row
# as-is. They are not half the sparse row: AMD's sparse figures run 2.02x dense (OCP-FP8 10.1
# PFLOPS against 5.0, INT8 10.1 POPS against 5.0), and halving them is what put FP8 and INT8
# at 5050 rather than the published 5000. MXFP4 has no sparsity row at all.
AMD_DENSE = {
    "bfloat16": {"AMD-Instinct-MI355X-288GB": 2500e12},
    "fp8": {"AMD-Instinct-MI355X-288GB": 5000e12},
    "int8": {"AMD-Instinct-MI355X-288GB": 5000e12},
    "fp4": {"AMD-Instinct-MI355X-288GB": 10100e12},
}


class PeakFlopsBasisTests(unittest.TestCase):

    def test_declared_basis_is_dense(self):
        self.assertEqual(PEAK_FLOPS_BASIS, "dense")

    def test_nvidia_entries_are_half_the_sparse_datasheet(self):
        for prec, cards in NVIDIA_SPARSE.items():
            for card, sparse in cards.items():
                dense = PEAK_FLOPS_DICT[prec][card]
                self.assertAlmostEqual(
                    dense, sparse / 2, delta=1e9,
                    msg=f"{card} {prec}: expected dense {sparse/2:.3e} "
                        f"(datasheet {sparse:.3e} is with 2:4 sparsity), got {dense:.3e}",
                )

    def test_nvidia_printed_dense_entries_are_not_halved(self):
        for prec, cards in NVIDIA_PRINTED_DENSE.items():
            for card, dense in cards.items():
                self.assertEqual(PEAK_FLOPS_DICT[prec][card], dense, f"{card} {prec}")

    def test_amd_entries_match_the_dense_datasheet(self):
        for prec, cards in AMD_DENSE.items():
            for card, dense in cards.items():
                self.assertEqual(PEAK_FLOPS_DICT[prec][card], dense)

    def test_a100_sub_bf16_precisions_upcast(self):
        # Ampere has no FP8/FP4 tensor cores; those runs compute at the bf16 rate, and a
        # catalog entry above it would inflate the A100 denominator.
        bf16 = PEAK_FLOPS_DICT["bfloat16"]["NVIDIA-A100-SXM4-80GB"]
        self.assertEqual(PEAK_FLOPS_DICT["fp8"]["NVIDIA-A100-SXM4-80GB"], bf16)
        self.assertEqual(PEAK_FLOPS_DICT["fp4"]["NVIDIA-A100-SXM4-80GB"], bf16)

    def test_fp8_at_least_bf16_per_card(self):
        for card, bf16 in PEAK_FLOPS_DICT["bfloat16"].items():
            self.assertGreaterEqual(PEAK_FLOPS_DICT["fp8"][card], bf16, card)

    def test_unknown_lookups_return_zero_not_a_guess(self):
        self.assertEqual(get_peak_flops("NVIDIA-Unknown-GPU", "bfloat16"), 0)
        self.assertEqual(get_peak_flops("NVIDIA-H100-HBM3-80GB", "no-such-precision"), 0)


if __name__ == "__main__":
    unittest.main()
