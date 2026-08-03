import unicodedata
import unittest
from pathlib import Path

from postprocessing.moe_cost_metrics.compute_sparsity_metrics import (
    MEM_BW_DICT,
    PEAK_FLOPS_DICT,
    get_peak_bw,
    get_peak_flops,
)

DOC = (
    Path(__file__).resolve().parents[1]
    / "postprocessing/moe_cost_metrics/compute_sparsity_metrics.md"
)

# Vendor primary bandwidth, pinned here so a second catalog layered over the first cannot
# silently revert a corrected figure. That is not hypothetical: a later assignment in the
# module used to rewrite B200 to 8 TB/s after the table was built, which made an attempt to
# correct that number a no-op with nothing to show for it.
DOCUMENTED_BW = {
    "NVIDIA-A100-SXM4-80GB": 2.039e12,
    "NVIDIA-H100-HBM3-80GB": 3.35e12,
    "NVIDIA-H200-141GB": 4.8e12,
    "NVIDIA-B200-183GB": 7.7e12,
    "NVIDIA-B300-269GB": 7.7e12,
    "AMD-Instinct-MI355X-288GB": 8.0e12,
    "NVIDIA-GB10": 273e9,
    "Tenstorrent-Blackhole-P150b": 512e9,
}

PRECISIONS = ("bfloat16", "float16", "fp8", "int8", "fp4", "int4")
DOC_COLUMNS = ("bfloat16", "fp8", "int8", "fp4", "int4")


def _doc_rows():
    """The hardware table from the published methodology, keyed by GPU key.

    Row width is checked against the header because the cell comparison zips the two and zip
    truncates to the shorter side: a row that lost cells to a bad edit would otherwise stop
    checking the precisions past its end instead of failing.
    """
    lines = DOC.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("| GPU key"))
    width = len(lines[start].strip("|").split("|"))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != width:
            raise AssertionError(
                f"doc table row {cells[0]} holds {len(cells)} cells against the header's {width}"
            )
        rows[cells[0].strip("`")] = cells[1:7]
    return rows


def _num(cell):
    """A table cell as TFLOPS/TB per second. Footnote markers and bold are decoration."""
    cell = "".join(c for c in cell if not unicodedata.category(c).startswith("No"))
    cell = cell.replace("*", "").replace("TB/s", "").strip()
    return 0.0 if cell in {"", "—", "-", "n/a"} else float(cell)


class HardwareCatalogTests(unittest.TestCase):

    def test_bandwidth_matches_the_documented_figure(self):
        for card, bw in DOCUMENTED_BW.items():
            self.assertEqual(get_peak_bw(card), bw, card)

    def test_bandwidth_catalog_holds_nothing_undocumented(self):
        self.assertEqual(set(MEM_BW_DICT), set(DOCUMENTED_BW))

    def test_no_card_has_a_compute_peak_without_a_bandwidth(self):
        # A zero bandwidth skips the whole sidecar, so a card catalogued for FLOPS but not
        # for bandwidth publishes nothing at all rather than S-MFU alone.
        for precision, cards in PEAK_FLOPS_DICT.items():
            for card in cards:
                self.assertGreater(get_peak_bw(card), 0, f"{card} ({precision})")

    def test_every_card_resolves_in_every_precision(self):
        # Precision dispatch takes whatever the run recorded. A card missing from one dict
        # publishes a null S-MFU for those runs instead of an answer. Seeded from the union
        # rather than from bfloat16, because the omission this guards against is exactly a
        # card that is present in some precision dicts and absent from others.
        for card in {c for d in PEAK_FLOPS_DICT.values() for c in d}:
            for precision in PRECISIONS:
                self.assertGreater(get_peak_flops(card, precision), 0, f"{card} @ {precision}")

    def test_float16_mirrors_bfloat16(self):
        # The doc table has no FP16 column and the basis tests never index float16, so an
        # entry here could be any multiple of the truth with nothing failing. Every card runs
        # the two at the same rate, which makes the whole dict checkable in one line.
        self.assertEqual(PEAK_FLOPS_DICT["float16"], PEAK_FLOPS_DICT["bfloat16"])

    def test_int4_mirrors_fp4_except_on_ampere(self):
        # Weight-only 4-bit checkpoints dequantise before the matmul, so the card's FP4 rate
        # is the denominator. A100 is the exception: Ampere has a native INT4 path.
        for card in PEAK_FLOPS_DICT["fp4"]:
            if card == "NVIDIA-A100-SXM4-80GB":
                continue
            self.assertEqual(
                PEAK_FLOPS_DICT["int4"][card], PEAK_FLOPS_DICT["fp4"][card], card
            )

    def test_unknown_card_has_no_bandwidth(self):
        self.assertEqual(get_peak_bw("NVIDIA-Unknown-GPU"), 0)

    def test_doc_table_lists_every_catalogued_card(self):
        catalogued = set(MEM_BW_DICT) | {c for d in PEAK_FLOPS_DICT.values() for c in d}
        missing = catalogued - set(_doc_rows())
        self.assertEqual(missing, set(), f"catalogued in code, absent from the doc table: {missing}")

    def test_doc_table_cells_match_the_catalog(self):
        # delta only absorbs the float round-trip through the decimal cell text. It was 1e9,
        # which is exactly the width of the A100 gap between a datasheet 2.039 TB/s and the
        # rounded 2.04 the table used to print, so the check could not tell the two apart in
        # either direction. Every documented cell is exact, so nothing needs the slack.
        for card, cells in _doc_rows().items():
            self.assertAlmostEqual(
                _num(cells[0]) * 1e12, get_peak_bw(card), delta=1e6,
                msg=f"{card} bandwidth",
            )
            for cell, precision in zip(cells[1:], DOC_COLUMNS):
                self.assertAlmostEqual(
                    _num(cell) * 1e12, get_peak_flops(card, precision), delta=1e6,
                    msg=f"{card} {precision}",
                )


if __name__ == "__main__":
    unittest.main()
