#!/usr/bin/env python3
"""R1.1 — fetch & verify the 24 target UniProt sequences -> data/targets/.
Portable: ROOT derived from this file's location. Any env (urllib only)."""
import os, json, time, csv, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/targets"); os.makedirs(OUT, exist_ok=True)

# slug -> (accession(s), status, note). der21 fixed (Q5XLE6 was panda Hb).
MAP = {
 "egfr":(["P00533"],"VERIFIED","human EGFR"),
 "pd-l1":(["Q9NZQ7"],"VERIFIED","CD274/PD-L1"),
 "il7r":(["P16871"],"VERIFIED","IL7R alpha"),
 "mdm2":(["Q00987"],"VERIFIED","MDM2"),
 "ifnar2":(["P48551"],"VERIFIED","IFNAR2"),
 "fcrn":(["P55899"],"VERIFIED","FCGRT/FcRn"),
 "human-insulin-receptor":(["P06213"],"VERIFIED","INSR"),
 "human-serum-albumin":(["P02768"],"VERIFIED","ALB"),
 "human-pdgfr-beta":(["P09619"],"VERIFIED","PDGFRB"),
 "fgf-r1":(["P11362"],"VERIFIED","FGFR1"),
 "human-tnfa":(["P01375"],"VERIFIED","TNF-alpha"),
 "hnmt":(["P50135"],"VERIFIED","HNMT"),
 "human-pmvk":(["Q15126"],"VERIFIED","PMVK"),
 "human-rfk":(["Q969G6"],"VERIFIED","RFK"),
 "human-phyh":(["O14832"],"VERIFIED","PHYH"),
 "human-idi2":(["Q9BXS1"],"VERIFIED","IDI2"),
 "human-ambp":(["P02760"],"VERIFIED","AMBP"),
 "human-orm2":(["P19652"],"VERIFIED","ORM2"),
 "human-gm2a":(["P17900"],"VERIFIED","GM2A"),
 "spcas9":(["Q99ZW2"],"VERIFIED","S. pyogenes Cas9 (bacterial)"),
 "nipah-glycoprotein-g":(["Q9IH62"],"VERIFIED","Nipah glycoprotein G"),
 "der7":(["P49273"],"VERIFIED","Der p 7 allergen"),
 "der21":(["Q2L7C5"],"VERIFIED","Der p 21.0101 allergen (FIXED from Q5XLE6)"),
 "human-mzb1-perp1":(["Q8WU39","Q96FX8"],"NEEDS-VERIFICATION","MZB1+PERP fusion? confirm vs Proteinbase"),
}

def fetch(acc):
    with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=25) as r:
        t = r.read().decode()
    return t.splitlines()[0][1:], "".join(t.splitlines()[1:])

rows, fasta = [], []
for slug,(accs,status,note) in MAP.items():
    seqs, hdrs = [], []
    for a in accs:
        h,s = fetch(a); seqs.append(s); hdrs.append(h); time.sleep(0.12)
    full = "".join(seqs)
    rows.append([slug, "+".join(accs), len(full), status, " | ".join(hdrs), note])
    fasta.append(f">{slug}|{'+'.join(accs)}|{status} {note}\n" +
                 "\n".join(full[i:i+60] for i in range(0,len(full),60)))
    print(f"  {slug:24s} {'+'.join(accs):16s} len={len(full):5d} {status}", flush=True)

with open(os.path.join(OUT,"target_uniprot_map.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["slug","uniprot_acc","seq_len","status","uniprot_header","note"]); w.writerows(rows)
with open(os.path.join(OUT,"target_sequences.fasta"),"w") as f:
    f.write("\n".join(fasta)+"\n")
nver=sum(1 for r in rows if r[3]=="VERIFIED")
print(f"DONE: {len(rows)} targets, {nver} verified, {len(rows)-nver} needs-verification -> {OUT}", flush=True)
