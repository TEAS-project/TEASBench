import unicodedata
import unittest
from pathlib import Path

from postprocessing.agentic_cost_metrics import compute_agentic_cost
from postprocessing.moe_cost_metrics import compute_cost, compute_sparsity_metrics
from postprocessing.moe_cost_metrics import hardware_catalog
from postprocessing.moe_cost_metrics.hardware_catalog import (
    CPU_SPECS,
    GPU_HOST_CPU,
    GPU_SPECS,
    MEM_BW_DICT,
    MEM_GB_DICT,
    PEAK_FLOPS_DICT,
    get_peak_bw,
    get_peak_flops,
    gpu_tdp_w,
    host_cpu_power_w,
    mem_capacity_gb,
    peak_bw_gb_s,
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
# Doc column header -> the precision the catalog keys on. Mapped by header rather than by
# position: the table has gained a column before, and a positional slice reads the wrong
# figure into the right assertion when it does.
DOC_PRECISION_COLUMNS = {
    "BF16": "bfloat16",
    "FP8": "fp8",
    "INT8": "int8",
    "FP4": "fp4",
    "INT4": "int4",
}


def _plain(text):
    """A cell or header without its decoration: footnote markers and bold."""
    text = "".join(c for c in text if not unicodedata.category(c).startswith("No"))
    return text.replace("*", "").strip()


def _doc_rows():
    """The hardware table from the published methodology, as {GPU key: {column: cell}}.

    Row width is checked against the header because the cells are zipped onto it: a row that
    lost or gained cells to a bad edit would otherwise shift every figure one column over and
    check each against its neighbour's catalog entry.
    """
    lines = DOC.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("| GPU key"))
    headers = [_plain(c) for c in lines[start].strip("|").split("|")]
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            raise AssertionError(
                f"doc table row {cells[0]} holds {len(cells)} cells "
                f"against the header's {len(headers)}"
            )
        rows[cells[0].strip("`")] = dict(zip(headers[1:], cells[1:]))
    return rows


def _num(cell):
    """A table cell as a bare number, in whichever unit its column prints."""
    cell = _plain(cell).replace("TB/s", "").replace("GB", "").strip()
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

    def test_the_producers_share_one_catalog_object(self):
        # Identity, not equality. Equality passes on a second table that happens to agree
        # today, which is the state this file exists to prevent: a card published at one
        # figure by the sparsity sidecar and another by whoever kept a copy. A local
        # redefinition in any producer fails here rather than at the next divergence — a
        # re-pasted table is a new object even when every value in it still matches.
        #
        # The agentic producer is the fifth copy this file closed. It lives in a sibling
        # package and reaches the catalog by a relative import, so it is listed here as much
        # to hold that import working as to hold the tables single.
        for module, names in (
            (compute_sparsity_metrics, ("MEM_BW_DICT", "PEAK_FLOPS_DICT", "GPU_TYPE_MAP",
                                        "GPU_KEY_FALLBACK", "B300_KEY")),
            (compute_cost, ("GPU_SPECS", "CPU_SPECS", "GPU_HOST_CPU")),
            (compute_agentic_cost, ("GPU_SPECS", "CPU_SPECS", "GPU_HOST_CPU")),
        ):
            for name in names:
                self.assertIs(
                    getattr(module, name), getattr(hardware_catalog, name),
                    f"{module.__name__}.{name} is not the catalog's object",
                )


# The accelerator and host-CPU power figures the dashboard's Hardware catalog publishes,
# pinned for the same reason as bandwidth above: the assembler in the results repo reads
# them from this module, and its own copies drifted for two cards before it did.
DOCUMENTED_GPU_TDP_W = {
    "a100": 400, "h100": 700, "h200": 700, "b200": 1000, "b300": 1400,
    "mi355x": 1400, "gb10": 140, "blackhole-p150b": 300, "cs3": 23000,
}
# 2x host CPU TDP; b300 = 2x Xeon 8558 (330W), the other rack parts 2x Xeon 8468 (350W);
# gb10 is on-SoC; the TT p150b is a PCIe card in a 1-CPU workstation host; the CS-3's own
# TDP is a whole-system peak with no separate host.
DOCUMENTED_HOST_CPU_W = {
    "a100": 700, "h100": 700, "h200": 700, "b200": 700, "b300": 660,
    "mi355x": 450, "gb10": 0, "blackhole-p150b": 350, "cs3": 0,
}
# GB/s, as the published table prints them. Every card the catalog knows; cs3 resolves no
# canonical key and is published from the assembler's own figure.
DOCUMENTED_BW_GB_S = {
    "a100": 2039, "h100": 3350, "h200": 4800, "b200": 7700, "b300": 7700,
    "mi355x": 8000, "gb10": 273, "blackhole-p150b": 512,
}
# Nameplate capacity in GB, pinned like bandwidth and TDP above. cs3 is out for the same
# reason. These figures deliberately do NOT track the capacities inside the canonical key
# strings — B200 is keyed 183GB and carries 192, B300 is keyed 269GB and carries 288 — because
# a key is an identifier and nothing parses a capacity out of one. A test that read the key
# would pin the wrong number and make that confusion permanent.
DOCUMENTED_MEM_GB = {
    "a100": 80, "h100": 80, "h200": 141, "b200": 192, "b300": 288,
    "mi355x": 288, "gb10": 128, "blackhole-p150b": 32,
}


class PublishedSpecTests(unittest.TestCase):
    """The short-key accessors the dashboard assembler publishes through."""

    def test_gpu_tdp_matches_the_documented_figure(self):
        for key, watts in DOCUMENTED_GPU_TDP_W.items():
            self.assertEqual(gpu_tdp_w(key), watts, key)

    def test_gpu_tdp_covers_every_catalogued_part(self):
        self.assertEqual(set(GPU_SPECS), set(DOCUMENTED_GPU_TDP_W))

    def test_host_cpu_power_matches_the_documented_figure(self):
        for key, watts in DOCUMENTED_HOST_CPU_W.items():
            self.assertEqual(host_cpu_power_w(key), watts, key)

    def test_host_cpu_power_is_the_product_of_its_two_tables(self):
        # The figure is a product, so a change to either side has to show up here rather
        # than in a hand-maintained total that agrees with neither.
        for key, (num, cpu_key) in GPU_HOST_CPU.items():
            self.assertEqual(host_cpu_power_w(key), num * CPU_SPECS[cpu_key]["tdp_w"], key)

    def test_bandwidth_in_gb_s_matches_the_documented_figure(self):
        for key, gb_s in DOCUMENTED_BW_GB_S.items():
            self.assertAlmostEqual(peak_bw_gb_s(key), gb_s, delta=1e-6, msg=key)

    def test_whole_bandwidths_stay_integers(self):
        # A float here would publish 2039.0 where the hardware table has always printed
        # 2039 — a moved number in the emitted JSON for no change in the measurement.
        for key in DOCUMENTED_BW_GB_S:
            self.assertIsInstance(peak_bw_gb_s(key), int, key)

    def test_memory_capacity_matches_the_documented_figure(self):
        for key, gb in DOCUMENTED_MEM_GB.items():
            self.assertEqual(mem_capacity_gb(key), gb, key)

    def test_memory_capacity_covers_every_card_with_a_bandwidth(self):
        # Capacity joined the catalog last. A card carrying a bandwidth but no capacity
        # publishes a blank spec in the dashboard's hardware table rather than failing.
        self.assertEqual(set(MEM_GB_DICT), set(MEM_BW_DICT))

    def test_every_canonical_key_is_reachable_from_a_short_key(self):
        # A canonical key no short key maps to is a card the assembler cannot publish.
        self.assertEqual(
            set(hardware_catalog.GPU_KEY_FALLBACK.values()), set(MEM_BW_DICT)
        )

    def test_an_uncatalogued_part_raises_rather_than_reading_as_zero(self):
        # The failure these accessors exist to make loud: a card that reaches a consumer
        # without reaching this file must stop the run, not publish at zero watts or as a
        # zero-capacity part.
        for fn in (gpu_tdp_w, host_cpu_power_w, peak_bw_gb_s, mem_capacity_gb):
            with self.assertRaises(KeyError, msg=fn.__name__):
                fn("no-such-gpu")

    def test_doc_table_lists_every_catalogued_card(self):
        catalogued = (set(MEM_BW_DICT) | set(MEM_GB_DICT)
                      | {c for d in PEAK_FLOPS_DICT.values() for c in d})
        missing = catalogued - set(_doc_rows())
        self.assertEqual(missing, set(), f"catalogued in code, absent from the doc table: {missing}")

    def test_doc_table_cells_match_the_catalog(self):
        # delta only absorbs the float round-trip through the decimal cell text. It was 1e9,
        # which is exactly the width of the A100 gap between a datasheet 2.039 TB/s and the
        # rounded 2.04 the table used to print, so the check could not tell the two apart in
        # either direction. Every documented cell is exact, so nothing needs the slack.
        for card, cells in _doc_rows().items():
            self.assertAlmostEqual(
                _num(cells["HBM BW"]) * 1e12, get_peak_bw(card), delta=1e6,
                msg=f"{card} bandwidth",
            )
            self.assertEqual(
                _num(cells["Memory"]), MEM_GB_DICT[card], f"{card} memory capacity",
            )
            for header, precision in DOC_PRECISION_COLUMNS.items():
                self.assertAlmostEqual(
                    _num(cells[header]) * 1e12, get_peak_flops(card, precision), delta=1e6,
                    msg=f"{card} {precision}",
                )


if __name__ == "__main__":
    unittest.main()
