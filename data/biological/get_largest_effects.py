import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

eventnames = "TP53 (M),MCL1/1q (Amp),KRAS (M),TERT/5p (Amp),CDKN2A/9p (Del),MYC/8q (Amp),EGFR (M),EGFR/7p (Amp),EPHA7/6q (Del),NKX2-1/14q (Amp),BCL2/18q (Del),LATS2/13q (Del),RMI2/16p (Amp),TGFBR2/3p (Del),ARID5B/10q (Del),STK11/19p (Del),GNAS/20q (Amp),TP53/17p (Del),CDKN1B/12p (Del),B2M/15q (Del),STK11 (M),CCND1/11q (Amp),RAC2/22q (Del),RBM10 (M),RUNX/21q (Amp),KEAP1 (M),CXCR4/2q (Del),EPHA5/4q (Del),ATM (M),SMARCA4 (M),NF1 (M),BRAF (M),PIK3CA (M),SETD2 (M),FAT1 (M),CDKN2A (M),ARID1A (M),MET (M),MED12 (M),ERBB2 (M),ATRX (M),KMT2C (M),RB1 (M),KMT2D (M),PTPRT (M),ARID2 (M),SMAD4 (M),CTNNB1 (M),TERT (M),BCOR (M),APC (M),U2AF1 (M),TBX3 (M),CREBBP (M),ATR (M),MTOR (M),PAK7 (M),KDM5C (M),EPHA5 (M),PBRM1 (M),EPHA3 (M),NOTCH3 (M),KMT2A (M),PTEN (M),NOTCH1 (M),DICER1 (M),SF3B1 (M),STAG2 (M),BRCA2 (M),NTRK3 (M),MAX (M),PIK3C2G (M),DNMT3A (M),AMER1 (M),CARD11 (M),GNAS (M),DOT1L (M),DIS3 (M),ERBB4 (M),NOTCH2 (M),BTK (M),ARID1B (M),POLE (M),PDGFRB (M),PDGFRA (M),EP300 (M),ALK (M),TP63 (M),SPEN (M),KDR (M),NOTCH4 (M),NCOR1 (M),TSC2 (M),WT1 (M),PTPRS (M),BRCA1 (M),NSD1 (M),GATA3 (M),RFWD2 (M),RASA1 (M),CDK12 (M),PIK3R1 (M),JAK3 (M),ARAF (M),NFE2L2 (M),FBXW7 (M),IKZF1 (M),TET2 (M),PTPRD (M),RPTOR (M),JAK2 (M),ASXL1 (M),PTCH1 (M),DDR2 (M),KIT (M),BAP1 (M),CUL3 (M),MAP3K1 (M),FGFR4 (M),LATS1 (M),RET (M),CBL (M),ASXL2 (M),TGFBR2 (M),FLT1 (M)".split(
    ","
)

theta = np.loadtxt("./theta25.dat", delimiter=" ")
d = theta.shape[1]
theta = theta[:d]

effects = dict()
for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        if i == j:
            continue
        effects[(i, j)] = theta[j, i]

effects = {
    k: v
    for k, v in sorted(
        effects.items(), key=lambda x: np.abs(x[1]), reverse=True
    )
}

G = nx.DiGraph()

for i, (k, v) in enumerate(effects.items()):
    G.add_edge(eventnames[k[0]], eventnames[k[1]], weight=v)
    print(f"{eventnames[k[0]]}\t->\t{eventnames[k[1]]} :\t {v}")
    if i == 30:
        break

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=8)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

plt.show()
