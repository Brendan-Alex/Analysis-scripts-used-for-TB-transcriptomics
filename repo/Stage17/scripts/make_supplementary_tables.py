"""Stage17: builds the supplementary tables (S1-S10) from the saved Stage7/10/11/13/16 outputs.

No refitting and no new analysis: every value is read from a results file. Writes supplementary_tables.md,
which is converted to .docx with pandoc."""
import pandas as pd, numpy as np, json, io, os
R="/root" # placeholder
B=os.path.expanduser("~/mnt/MTB ML")
M=pd.read_csv(f"{B}/Stage7/results/MASTER_results_table.csv")
C=pd.read_csv(f"{B}/Stage11/results/chosen_k_per_fold.csv")
S10=pd.read_csv(f"{B}/Stage10/results/all_selectors_comparison.csv")
EX=pd.read_csv(f"{B}/Stage13/results/external_auc.csv")
VR=pd.read_csv(f"{B}/Stage13/results/vs_random.csv")
DL=pd.read_csv(f"{B}/Stage16/results/delong_paired.csv")
SG=pd.read_csv(f"{B}/Stage13/results/subgroups.csv")
CN=pd.read_csv(f"{B}/Stage13/results/canary_noise.csv")
PG=pd.read_csv(f"{B}/Stage13/results/per_gene_direction.csv")

MM={'Logistic regression':'LR','Random forest':'RF','MLP (alpha fixed at 10)':'MLP',
    'singscore (rank-based, untrained)':'singscore','Directional average (untrained)':'DirAvg'}
ORDER=['LR','RF','MLP','singscore','DirAvg']
PN={'Native':'Nat(mRMR)','Berry86':'Berry86','Sweeney3':'Sweeney3','Zak16':'Zak16',
    'Kaforou_ATB_v_LTB':'Kaforou (ATB vs LTB)','Kaforou_ATB_v_OD':'Kaforou (ATB vs OD)'}
PORDER=['Nat (per-fold)','Nat(mRMR)','Berry86','Kaforou (ATB vs LTB)','Kaforou (ATB vs OD)','Sweeney3','Zak16']

def f3(x):
    return '' if pd.isna(x) else f"{x:.3f}"

rows=[]
M2=M[M['Panel']!='Berry86_noTLR7'].copy()
M2['m']=M2['Method'].map(MM); M2['p']=M2['Panel'].map(PN)
for (task,held,panel),g in M2.groupby(['Task','Held-out cohort','p'],sort=False):
    d={'Task':task,'Held-out cohort':held,'Panel':panel,'Genes':int(g['Panel genes used'].iloc[0])}
    beat=0
    for meth in ORDER:
        r=g[g['m']==meth]
        if len(r)==0: d[meth]=''; continue
        r=r.iloc[0]; d[meth]=f3(r['AUC'])
        if r['Proportion of random panels with AUC >= panel']<=0.05: beat+=1
    r0=g.iloc[0]
    d['DirAvg 95% CI']=''
    rr=g[g['m']=='DirAvg']
    if len(rr): d['DirAvg 95% CI']=f"{rr.iloc[0]['AUC 95% CI lower']:.3f}–{rr.iloc[0]['AUC 95% CI upper']:.3f}"
    d['Random median']=f3(g['Random panel AUC median'].median())
    d['Beats random']=f"{beat}/5"
    d['Note']='Derivation overlap' if bool(r0['Optimistic (any derivation overlap)']) else ('Sibling sub-series' if bool(r0['Sibling-cohort flag (GSE19439 fold: panel selection included sister sub-series GSE19444)']) else '')
    rows.append(d)

# Nat from Stage11 (consistent ranking, chosen k)
Cn=C[C['Ranking']=='consistent']
for (task,held),g in Cn.groupby(['Task','Held-out cohort'],sort=False):
    d={'Task':task,'Held-out cohort':held,'Panel':'Nat (per-fold)','Genes':''}
    ks=[]; beat=0
    for meth in ORDER:
        r=g[g['Method']==meth]
        if len(r)==0: d[meth]=''; continue
        r=r.iloc[0]; d[meth]=f3(r['AUC at chosen k']); ks.append(int(r['Chosen k (inner LOCO)']))
        if r['Proportion random >= at k']<=0.05: beat+=1
    d['Genes']=f"{min(ks)}–{max(ks)}" if ks and min(ks)!=max(ks) else (str(ks[0]) if ks else '')
    rr=g[g['Method']=='DirAvg']
    d['DirAvg 95% CI']=f"{rr.iloc[0]['CI lower']:.3f}–{rr.iloc[0]['CI upper']:.3f}" if len(rr) else ''
    d['Random median']=f3(g['Random median at k'].median())
    d['Beats random']=f"{beat}/5"
    d['Note']='Sibling sub-series' if bool(g.iloc[0]['Sibling flag']) else ''
    rows.append(d)

BENCH=pd.DataFrame(rows)
BENCH['po']=BENCH['Panel'].map({p:i for i,p in enumerate(PORDER)})
BENCH=BENCH.sort_values(['Task','Held-out cohort','po'])
COLS=['Held-out cohort','Panel','Genes']+ORDER+['DirAvg 95% CI','Random median','Beats random','Note']
BENCH[['Task']+COLS].to_csv('/tmp/bench.csv',index=False)
print("BENCH rows per task:"); print(BENCH['Task'].value_counts())
print(BENCH[BENCH['Task']=='ATB vs OD'][COLS].to_string(index=False))

out=io.StringIO()
W=out.write
def md(df):
    cols=list(df.columns)
    W('| '+' | '.join(cols)+' |\n')
    W('|'+'|'.join(['---']*len(cols))+'|\n')
    for _,r in df.iterrows():
        W('| '+' | '.join('' if pd.isna(v) else str(v) for v in r)+' |\n')
    W('\n')

NOTE_BENCH=("AUC in the held-out cohort for every panel and scoring method. Nat (per-fold) was rebuilt from scratch in each fold "
"and its size chosen by an inner leave-one-cohort-out loop separately for each method, so the gene count is "
"given as a range across the five methods; all other panels are fixed. The 95% confidence interval is given "
"for the directional average, the primary score; intervals for the other methods are in the deposited analysis "
"outputs. Random median is the median AUC of 50 size-matched random panels drawn for that fold and panel size. "
"Beats random counts the scoring methods whose AUC was reached by at most 5% of those random panels. "
"It is not the same object as the frozen Nat (final) panel evaluated in S7-S10 Tables, which was derived once on all "
"cohorts and therefore cannot be scored on any discovery cohort. "
"Derivation overlap marks cells where the published panel's own derivation cohort, or one of its sub-series, "
"is the held-out cohort; those AUCs are optimistic. Sibling sub-series marks the GSE19439 fold, whose training "
"cohorts include its sister sub-series GSE19444. Values are not pooled or averaged across folds.")

for i,(task,lab) in enumerate([('ATB vs LTB','S1'),('ATB vs OD','S2'),('LTB vs CON','S3')],1):
    W(f"**{lab} Table. {task}: cross-cohort benchmark, all panels and scoring methods.**\n\n")
    W(NOTE_BENCH+"\n\n")
    md(BENCH[BENCH['Task']==task][COLS])

# S4 Nat size selection
W("**S4 Table. Nat panel size selected in each fold by the inner leave-one-cohort-out loop.**\n\n")
W("For each fold the direction-consistency ranking was rebuilt on the training cohorts only, and k (1-20) was "
"chosen as the smallest value whose mean inner-loop AUC was within 0.01 of the best, separately for each scoring "
"method. Genes are listed for the directional average. The final column is the held-out AUC at the chosen k minus "
"the held-out AUC a fixed 20-gene panel would have given in the same fold.\n\n")
s4=[]
for _,r in Cn.iterrows():
    s4.append({'Task':r['Task'],'Held-out cohort':r['Held-out cohort'],'Method':r['Method'],
               'Chosen k':int(r['Chosen k (inner LOCO)']),
               'AUC at chosen k':f3(r['AUC at chosen k']),
               '95% CI':f"{r['CI lower']:.3f}–{r['CI upper']:.3f}",
               'Random median':f3(r['Random median at k']),
               'Prop. random ≥':f"{r['Proportion random >= at k']:.2f}",
               'Chosen − k=20':f"{r['Chosen minus k=20']:+.3f}",
               'Genes (DirAvg)':r['Genes'].replace(';',', ') if r['Method']=='DirAvg' else ''})
s4=pd.DataFrame(s4)
s4['mo']=s4['Method'].map({m:i for i,m in enumerate(ORDER)})
s4=s4.sort_values(['Task','Held-out cohort','mo']).drop(columns='mo')
md(s4)

# S5 selector ladder
W("**S5 Table. Comparison of six feature-selection criteria.**\n\n")
W("Every variant builds a 20-gene panel per fold, is rebuilt inside the fold using the training cohorts only, and is "
"scored once on the held-out cohort. Median, minimum and maximum are taken over all fold-by-method cells of the task "
"(25 for ATB vs LTB, 20 for the other two). Beats random counts the cells whose AUC was reached by at most 5% of "
"50 size-matched random panels. Note that these are fixed 20-gene panels, so the Nat row here is not identical to "
"the inner-loop-sized Nat reported in S1-S4 Tables.\n\n")
SN={'original Native (mRMR + FS)':'Nat(mRMR), as specified','mRMR_scaled':'mRMR, relevance and redundancy rescaled',
    'MI20_FS':'MI top-20 union + forward selection','MI_meanrank':'MI mean rank','consistent60_FS':'Direction consistency + forward selection',
    'consistent':'Direction consistency (Nat)'}
s5=S10.copy(); s5['Selector']=s5['Selector'].map(SN)
s5['so']=S10['Selector'].map({k:i for i,k in enumerate(['original Native (mRMR + FS)','mRMR_scaled','MI20_FS','MI_meanrank','consistent60_FS','consistent'])})
s5=s5.sort_values(['Task','so'])
s5=pd.DataFrame({'Task':s5['Task'],'Selection criterion':s5['Selector'],
                 'Median AUC':s5['AUC_median'].map(f3),'Min':s5['AUC_min'].map(f3),'Max':s5['AUC_max'].map(f3),
                 'Beats random':s5['beats'].astype(str)+'/'+s5['cells'].astype(str)})
md(s5)


# S6 frozen panels + fold recurrence
W("**S6 Table. Frozen Nat panels and the recurrence of their genes across folds.**\n\n")
W("The direction-consistency procedure was run once on all discovery cohorts of each task to produce a single frozen "
"panel for deposit and external validation. These panels were derived using every discovery cohort and so cannot be "
"evaluated by leave-one-cohort-out; their only out-of-sample evidence is the external cohorts in S7-S10 Tables. As "
"an internal consistency check, the two right-hand columns count how often each frozen gene was independently "
"recovered by the per-fold derivations reported in S1-S4 Tables: whether it entered that fold's top-20 "
"direction-consistent ranking, and whether it entered that fold's selected panel under the directional average. "
"Combined8, used in S7-S9 Tables, is the union of the ATB vs LTB and ATB vs OD panels and contains all eight genes "
"listed here for those two tasks. No panel is proposed for LTB vs CON: the procedure returns eleven genes, but four "
"of them were recovered by no fold at all.\n\n")
FS=pd.read_csv(f"{B}/Stage11/results/final_signatures.csv")
fs=FS[FS['Ranking']=='consistent'].copy()
fs['to']=fs['Task'].map({'ATB vs LTB':0,'ATB vs OD':1,'LTB vs CON':2})
fs=fs.sort_values(['to','Rank in final'])
s6a=pd.DataFrame({'Task':fs['Task'],'Panel':fs['Task'].map({'ATB vs LTB':'Nat (final), k = 3','ATB vs OD':'Nat (final), k = 5','LTB vs CON':'Not proposed (k = 11)'}),
    'Rank':fs['Rank in final'],'Gene':fs['Gene'],
    'In fold top-20 ranking':fs['In fold top-20'],
    'In fold selected panel (DirAvg)':fs['In fold chosen-k panel (DirAvg)']})
md(s6a)

# S7 external
W("**S7 Table. External validation: locked panels applied without refitting.**\n\n")
W("GSE144127 (ATB vs OD; 79 active TB vs 300 other disease) and GSE152218 (ATB vs LTB; 14 vs 27). Neither cohort "
"took part in panel derivation, model training or any earlier analysis. Models were trained on all discovery cohorts "
"of the task and scored once. Nat (final) is the frozen panel for that task, derived once on all discovery cohorts of "
"the task and applied here without modification: ANKRD22, FCGR1A and PSTPIP2 for ATB vs LTB (k = 3) and APOL2, WDFY1, "
"BATF2, POLB and GBP2 for ATB vs OD (k = 5). Combined8 is the union of those two panels, an eight-gene set applied "
"unchanged to both tasks. MI mean rank is the corresponding frozen panel from the MI mean-rank selector variant. "
"Random median and the empirical "
"p-value compare the panel with 50 size-matched random panels; empirical p is reported for the panels for which "
"random comparators were drawn.\n\n")
EN={'S11_consistent':'Nat (final)','S11_MI_meanrank':'MI mean rank','Combined8':'Combined8','Zak16':'Zak16',
    'Berry86':'Berry86','Berry86_noTLR7':None,'Sweeney3':'Sweeney3',
    'Kaforou_ATB_v_OD':'Kaforou (ATB vs OD)','Kaforou_ATB_v_LTB':'Kaforou (ATB vs LTB)'}
ex=EX.copy(); ex['p']=ex['panel'].map(EN); ex=ex[ex['p'].notna()]
vr=VR.set_index(['cohort','panel','method'])
s6=[]
for _,r in ex.iterrows():
    k=(r['cohort'],r['panel'],r['method'])
    rm=f3(vr.loc[k,'random_median']) if k in vr.index else ''
    ep=f"{vr.loc[k,'empirical_p']:.3f}" if k in vr.index else ''
    s6.append({'Cohort':r['cohort'],'Panel':r['p'],'Genes':int(r['panel_genes']),'Method':r['method'],
               'AUC':f3(r['AUC']),'95% CI':f"{r['CI_lo']:.3f}–{r['CI_hi']:.3f}",
               'Random median':rm,'Empirical p':ep})
s6=pd.DataFrame(s6); s6['mo']=s6['Method'].map({m:i for i,m in enumerate(ORDER)})
s6['pp']=s6['Panel'].map({p:i for i,p in enumerate(['Nat (final)','Combined8','MI mean rank','Berry86','Kaforou (ATB vs OD)','Kaforou (ATB vs LTB)','Sweeney3','Zak16'])})
s6=s6.sort_values(['Cohort','pp','mo']).drop(columns=['mo','pp'])
md(s6)

# S7 DeLong
W("**S8 Table. Paired DeLong comparisons in the external cohorts.**\n\n")
W("Nat against every other panel, computed on identical samples. ΔAUC is Nat minus comparator. p-values are adjusted "
"by the Holm procedure within each cohort and scoring method, over the six comparators listed. No comparison reaches "
"significance after adjustment in either cohort.\n\n")
dl=DL.copy(); dl['Comparator']=dl['comparator'].map(EN)
s7=pd.DataFrame({'Cohort':dl['cohort'],'Method':dl['method'],'Comparator':dl['Comparator'],
    'AUC Nat':dl['AUC_Nat'].map(f3),'AUC comparator':dl['AUC_comparator'].map(f3),
    'ΔAUC':dl['AUC_diff'].map(lambda v:f"{v:+.4f}"),
    '95% CI':dl.apply(lambda r:f"{r['diff_CI_lo']:+.4f} to {r['diff_CI_hi']:+.4f}",axis=1),
    'p':dl['p'].map(lambda v:f"{v:.3f}"),'p (Holm)':dl['p_Holm'].map(lambda v:f"{v:.3f}")})
s7['mo']=s7['Method'].map({m:i for i,m in enumerate(ORDER)})
s7=s7.sort_values(['Cohort','mo','Comparator']).drop(columns='mo')
md(s7)

# S8 subgroups, DirAvg
W("**S9 Table. Subgroup analyses in the external cohorts (directional average).**\n\n")
W("AUC of each panel within clinical and demographic subgroups of the validation cohorts. The active TB group is held "
"constant and the comparison group is restricted. Sample sizes are given as active TB versus comparator. The "
"GSE152218 female subgroup contains a single active TB case and no AUC is estimable.\n\n")
sg=SG[SG['method']=='DirAvg'].copy(); sg['p']=sg['panel'].map(EN); sg=sg[sg['p'].notna()]
sg['n']=sg['n_pos'].astype(str)+' vs '+sg['n_neg'].astype(str)
piv=sg.pivot_table(index=['cohort','subgroup','n'],columns='p',values='AUC',aggfunc='first')
piv=piv.reindex(columns=[c for c in ['Nat (final)','Combined8','MI mean rank','Berry86','Kaforou (ATB vs OD)','Kaforou (ATB vs LTB)','Sweeney3','Zak16'] if c in piv.columns])
piv=piv.round(3).reset_index().rename(columns={'cohort':'Cohort','subgroup':'Subgroup','n':'n (ATB vs comparator)'})
md(piv.astype(object).where(piv.notna(),'n.e.'))

# S9 direction + canary
W("**S10 Table. Direction preservation and leakage canary in the external cohorts.**\n\n")
W("Upper block: for every gene of the locked panels, the direction of change learned in the discovery cohorts, the "
"per-cohort discovery AUC, and the AUC of that single gene in the validation cohort. Lower block: AUC when the "
"validation expression matrix is replaced by noise, with labels unchanged.\n\n")
pg=PG.copy(); pg['p']=pg['panel'].map(EN)
s9=pd.DataFrame({'Cohort':pg['cohort'],'Panel':pg['p'],'Gene':pg['gene'],
    'Direction in discovery':pg['training_direction'],
    'Discovery AUC per cohort':pg['training_AUC_per_cohort'].map(lambda s:', '.join(f"{k} {v}" for k,v in json.loads(s).items())),
    'Validation AUC':pg['validation_AUC'].map(f3),
    'Direction preserved':pg['same_direction_in_validation'].map({True:'Yes',False:'No'})})
md(s9)
cn=CN.rename(columns={'cohort':'Cohort','noise_AUC_LR':'LR','noise_AUC_RF':'RF','noise_AUC_MLP':'MLP',
    'noise_AUC_singscore':'singscore','noise_AUC_DirAvg':'DirAvg'})
for c in ORDER: cn[c]=cn[c].map(f3)
md(cn)

open('/tmp/supp.md','w').write(out.getvalue())
print("written", len(out.getvalue()), "chars")
