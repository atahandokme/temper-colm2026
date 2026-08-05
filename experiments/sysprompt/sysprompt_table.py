#!/usr/bin/env python3
"""Full 8-model sysprompt-vs-neutralization table for MG3d Q4."""
import json, os
EM = ['anger','joy','sadness','fear','disgust','surprise']
R = "results_v3/sysprompt_reasoning"

MODELS = ["Llama-3.1-8B","Qwen-2.5-7B","Qwen-2.5-3B","Gemma-2-9B","Mistral-7B-v0.3",
          "Mistral-Sm-24B","GPT-4o-mini","DeepSeek-V3"]

def stats(samples):
    n=o=e=nu=0
    for s in samples:
        c=s['conditions']; oc=c.get('original',{}).get('correct')
        if oc is None: continue
        for em in EM:
            ek,nk=f'emo_{em}',f'neu_{em}'
            if ek not in c: continue
            n+=1; o+=int(bool(oc)); e+=int(bool(c[ek].get('correct')))
            if nk in c: nu+=int(bool(c[nk].get('correct')))
    return n, 100*o/n, 100*e/n, 100*nu/n

print(f'{"Model":<18}{"O(no)":>7}{"E(no)":>7}{"N(no)":>7}{"E(sys)":>8}{"drop_no":>9}{"drop_sys":>10}{"neu_rec%":>10}{"sys_rec%":>10}')
print("-"*90)
for m in MODELS:
    fn=f'{R}/nosys__{m}.json'; fs=f'{R}/sys__{m}.json'
    if not (os.path.exists(fn) and os.path.exists(fs)):
        print(f'{m:<18}  missing files')
        continue
    nb=json.load(open(fn)); sb=json.load(open(fs))
    nS = nb['samples'] if isinstance(nb,dict) else nb[0]['samples']
    sS = sb['samples'] if isinstance(sb,dict) else sb[0]['samples']
    _, no_o, no_e, no_n = stats(nS)
    _, sy_o, sy_e, sy_n = stats(sS)
    drop_no = no_o - no_e
    drop_sys = sy_o - sy_e
    # Neutralization recovery: how much of the no-sys emotional drop is recovered by neutralization
    neu_rec = 100*(no_n - no_e)/(no_o - no_e) if no_o > no_e else float('nan')
    # Sys recovery: how much of the no-sys emotional drop is recovered by adding the system prompt
    # Compare emotional with sys vs without -- normalized by the no-sys gap
    sys_rec = 100*(sy_e - no_e)/(no_o - no_e) if no_o > no_e else float('nan')
    print(f'{m:<18}{no_o:>7.1f}{no_e:>7.1f}{no_n:>7.1f}{sy_e:>8.1f}{drop_no:>9.2f}{drop_sys:>10.2f}{neu_rec:>9.0f}%{sys_rec:>9.0f}%')
