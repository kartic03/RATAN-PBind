#!/usr/bin/env python3
"""R1.3 — ESM-2 target embeddings (corrected seqs).
Saves models/target_emb_v2_whole.npy (24x1280), target_emb_v2_slugs.json,
and models/target_perres.npz (per-residue, all 24). base env (torch+fair-esm).
Portable ROOT; chunks sequences > 1020 for the ESM-2 position budget."""
import os, json, time, numpy as np, torch
import esm as fair_esm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTA = os.path.join(ROOT,"data/targets/target_sequences.fasta")
MDL = os.path.join(ROOT,"models"); os.makedirs(MDL, exist_ok=True)
MAXLEN = 1020
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def read_fasta(p):
    seqs={};name=None;buf=[]
    for ln in open(p):
        if ln.startswith(">"):
            if name: seqs[name]="".join(buf)
            name=ln[1:].split("|")[0].strip();buf=[]
        else: buf.append(ln.strip())
    if name: seqs[name]="".join(buf)
    return seqs

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={dev}; loading ESM-2 650M ...")
    model,alpha=fair_esm.pretrained.esm2_t33_650M_UR50D(); model=model.eval().to(dev)
    bc=alpha.get_batch_converter()
    @torch.no_grad()
    def perres(seq):
        reps=[]
        for s in range(0,len(seq),MAXLEN):
            c=seq[s:s+MAXLEN]; _,_,t=bc([("x",c)]); t=t.to(dev)
            o=model(t,repr_layers=[33])["representations"][33]
            reps.append(o[0,1:len(c)+1].float().cpu().numpy())
        return np.concatenate(reps,0)
    seqs=read_fasta(FASTA); slugs=list(seqs.keys())
    whole=np.zeros((len(slugs),1280),dtype=np.float32); pr_out={}
    for i,sl in enumerate(slugs):
        pr=perres(seqs[sl]); whole[i]=pr.mean(0); pr_out[sl]=pr.astype(np.float16)
        log(f"  {sl:24s} len={len(seqs[sl]):5d}")
    np.save(os.path.join(MDL,"target_emb_v2_whole.npy"), whole)
    json.dump(slugs, open(os.path.join(MDL,"target_emb_v2_slugs.json"),"w"))
    np.savez_compressed(os.path.join(MDL,"target_perres.npz"), **pr_out)
    log(f"saved target_emb_v2_whole.npy {whole.shape}, target_perres.npz, slugs")

if __name__=="__main__": main()
