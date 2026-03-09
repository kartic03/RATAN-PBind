"""
RATAN-PBind — Residue Attribution and Target Affinity Network for Protein Binding
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Quick start:
    from protbind import RatanPBind
    pb = RatanPBind()
    result = pb.predict("MASWKELLVQ...", target="egfr")
    explanation = pb.explain(result)
    print(explanation["natural_language"])
"""

from protbind.predictor import ProtBind

# Primary public name
RatanPBind = ProtBind

KNOWN_TARGETS = [
    "der21", "der7", "egfr", "fcrn", "fgf-r1", "hnmt",
    "human-ambp", "human-gm2a", "human-idi2", "human-insulin-receptor",
    "human-mzb1-perp1", "human-orm2", "human-pdgfr-beta", "human-phyh",
    "human-pmvk", "human-rfk", "human-serum-albumin", "human-tnfa",
    "ifnar2", "il7r", "mdm2", "nipah-glycoprotein-g", "pd-l1", "spcas9",
]

__version__ = "1.1.0"
__all__      = ["RatanPBind", "ProtBind", "KNOWN_TARGETS"]
