import pandas as pd, numpy as np, sys
f=sys.argv[1]
df=pd.read_csv(f,index_col=0,dtype={0:str},low_memory=False)
df=df[[c for c in df.columns if str(c).startswith("GSM")]].apply(pd.to_numeric,errors="coerce").astype("float32")
df=df[df.notna().all(1)]
X=df.to_numpy(); Xc=(X-X.mean(0))/X.std(0); C=(Xc.T@Xc)/X.shape[0]
np.fill_diagonal(C,0)
i,j=np.where(np.triu(C)>0.995)
print(f.split('/')[-1],"samples",X.shape[1],"probes",X.shape[0],"pairs r>0.995:",len(i))
for a,b in zip(i,j):
    same=np.allclose(X[:,a],X[:,b],atol=1e-4)
    print(" ",df.columns[a],df.columns[b],round(float(C[a,b]),5),"IDENTICAL" if same else "")
print("max off-diag r:",round(float(C.max()),5))
