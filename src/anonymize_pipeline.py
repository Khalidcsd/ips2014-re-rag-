#!/usr/bin/env python3
"""
IPS-2014 anonymization pipeline
===============================
Turns the real extracted supplier/component data (sourced from a confidential
third-party OEM specification family) into a shareable, anonymized research
dataset.

Design principles
-----------------
1. Deterministic & reproducible: a fixed SALT means the same input always maps
   to the same pseudonym, so re-running never reshuffles identities. The salt is
   the only secret; change it and the mapping changes wholesale.
2. Pseudonymization, not deletion: identifying entities (supplier names, IDs,
   the OEM, the named contact) are replaced with stable surrogate keys. The
   real<->surrogate mapping is written to a SEPARATE key file that you keep
   private and never publish.
3. Non-identifying engineering data is preserved: the component taxonomy,
   counts, complexity ratings and category structure carry no personal/
   commercial identity and stay intact.
4. Honest provenance: any field that was modelled rather than extracted (the
   pricing layer) is tagged `synthetic=True` so it can never be read as
   ground truth pulled from the source document.

Stages
------
  load  ->  anonymize  ->  emit dataset (+ private key)

The extraction step is treated as already done (structured CSV/JSON in
--input). To run the pipeline literally "from the beginning" off the raw PDFs,
drop your PDF extractor in `stage_extract()` so it emits the same structures;
everything downstream is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_CODE = "IPS-2014"          # public name for the anonymized corpus
OEM_SURROGATE = "OEM-X"            # replaces the real OEM throughout
SALT = "ips-2014::change-me"       # the one secret; rotating it re-keys everything

# Free-text strings scrubbed wherever they appear (case-insensitive).
OEM_ALIASES = ["Daimler AG", "Daimler", "Mercedes-Benz", "Mercedes Benz", "MBN", "DBL", "BQF"]

# Regexes for direct identifiers in metadata / free text.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d()\s/-]{6,}\d")


# ---------------------------------------------------------------------------
# Surrogate key generation
# ---------------------------------------------------------------------------

@dataclass
class Pseudonymizer:
    """Deterministic, salted, prefix-namespaced surrogate keys with a recoverable map."""
    salt: str
    _maps: dict[str, dict[str, str]] = field(default_factory=dict)

    def key(self, namespace: str, value: str, width: int = 3) -> str:
        value = (value or "").strip()
        if not value:
            return ""
        ns = self._maps.setdefault(namespace, {})
        if value not in ns:
            # Stable ordering by salted hash -> unlinkable to original sort order.
            digest = hashlib.sha256(f"{self.salt}:{namespace}:{value}".encode()).hexdigest()
            idx = int(digest, 16) % (10 ** width)
            surrogate = f"{namespace}-{idx:0{width}d}"
            # Resolve the rare collision deterministically.
            while surrogate in ns.values():
                idx = (idx + 1) % (10 ** width)
                surrogate = f"{namespace}-{idx:0{width}d}"
            ns[value] = surrogate
        return ns[value]

    def mapping(self) -> dict[str, dict[str, str]]:
        return self._maps


def scrub_text(text: str) -> str:
    """Strip direct identifiers and OEM aliases from any free-text field."""
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL_REMOVED]", text)
    text = PHONE_RE.sub("[PHONE_REMOVED]", text)
    for alias in OEM_ALIASES:
        text = re.sub(re.escape(alias), OEM_SURROGATE, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Stage 1 — load (extraction already done upstream)
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stage_extract(input_dir: Path) -> dict:
    """Load the structured extraction. Swap in a PDF extractor here for raw input."""
    return {
        "suppliers": read_csv(input_dir / "BQF_9666_5_suppliers.csv"),
        "components": read_csv(input_dir / "BQF_9666_5_components.csv"),
        "analysis": json.loads((input_dir / "BQF_9666_5_analysis.json").read_text("utf-8")),
        "prices": _maybe_csv(input_dir / "BQF_9666_5_component_prices.csv"),
    }


def _maybe_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


# ---------------------------------------------------------------------------
# Stage 2 — anonymize
# ---------------------------------------------------------------------------

def anonymize(data: dict, pseudo: Pseudonymizer) -> dict:
    out: dict = {}

    # Suppliers: pseudonymize name + id, generalize location, keep category info.
    out["suppliers"] = []
    for row in data["suppliers"]:
        out["suppliers"].append({
            "supplier_key": pseudo.key("SUP", row["supplier_name"]),
            "supplier_ref": pseudo.key("REF", row.get("supplier_id", ""), width=4),
            "region": "DE",  # country-level only; drops the identifying city
            "categories": row.get("categories", ""),
            "category_count": row.get("category_count", ""),
        })

    # Components: pure engineering taxonomy -> no identity, passed through as-is.
    out["components"] = data["components"]

    # Metadata: strip the named contact and OEM identity entirely.
    src_meta = data["analysis"].get("document", {})
    out["metadata"] = {
        "dataset_code": DATASET_CODE,
        "source_document": pseudo.key("DOC", src_meta.get("document_id", ""), width=2),
        "source_family": f"{OEM_SURROGATE} mechanical-equipment specification family",
        "version_year": str(src_meta.get("version", ""))[:4],
        "pages": src_meta.get("pages"),
        "domain": "hydraulic equipment / industrial requirements",
        "responsible_contact": "[REMOVED — direct PII]",
        "copyright_note": f"Derived from copyright-restricted {OEM_SURROGATE} source; "
                          f"source not redistributable. This dataset is anonymized derivative data.",
    }

    # Statistics: counts only, non-identifying — carried through, OEM scrubbed.
    stats = data["analysis"].get("statistics", {})
    out["statistics"] = json.loads(scrub_text(json.dumps(stats, ensure_ascii=False)))
    out["category_breakdown"] = json.loads(
        scrub_text(json.dumps(data["analysis"].get("category_breakdown", {}), ensure_ascii=False))
    )

    # Pricing: kept but explicitly flagged synthetic (modelled, not extracted).
    out["prices_synthetic"] = [
        {**row, "synthetic": True} for row in data.get("prices", [])
    ]

    return out


# ---------------------------------------------------------------------------
# Stage 3 — emit
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stage_emit(anon: dict, pseudo: Pseudonymizer, out_dir: Path, key_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / f"{DATASET_CODE}_suppliers_anon.csv", anon["suppliers"])
    write_csv(out_dir / f"{DATASET_CODE}_components.csv", anon["components"])
    if anon["prices_synthetic"]:
        write_csv(out_dir / f"{DATASET_CODE}_prices_synthetic.csv", anon["prices_synthetic"])

    (out_dir / f"{DATASET_CODE}_dataset.json").write_text(
        json.dumps({
            "metadata": anon["metadata"],
            "statistics": anon["statistics"],
            "category_breakdown": anon["category_breakdown"],
            "n_suppliers": len(anon["suppliers"]),
            "n_component_subcategories": len(anon["components"]),
        }, ensure_ascii=False, indent=2), "utf-8"
    )

    # PRIVATE — keep out of any release. This is the re-identification key.
    (key_dir / f"{DATASET_CODE}_REIDENTIFICATION_KEY.json").write_text(
        json.dumps(pseudo.mapping(), ensure_ascii=False, indent=2), "utf-8"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Anonymize extracted OEM supplier data into a shareable dataset.")
    ap.add_argument("--input", type=Path, required=True, help="dir with extracted CSV/JSON")
    ap.add_argument("--out", type=Path, default=Path("dataset_anon"), help="public dataset output dir")
    ap.add_argument("--keys", type=Path, default=Path("private_keys"), help="PRIVATE key output dir")
    ap.add_argument("--salt", default=SALT, help="anonymization salt (keep secret)")
    args = ap.parse_args()

    pseudo = Pseudonymizer(salt=args.salt)
    data = stage_extract(args.input)
    anon = anonymize(data, pseudo)
    stage_emit(anon, pseudo, args.out, args.keys)

    n_sup = len(pseudo.mapping().get("SUP", {}))
    print(f"[{DATASET_CODE}] anonymized {n_sup} suppliers, "
          f"{len(anon['components'])} component subcategories.")
    print(f"  public dataset -> {args.out}/")
    print(f"  PRIVATE key    -> {args.keys}/  (do NOT publish)")


if __name__ == "__main__":
    main()
