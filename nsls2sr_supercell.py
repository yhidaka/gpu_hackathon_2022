import pylatt as latt

# === Element definition:
b1g3c01a = latt.bend(
    "b1g3c01a", L=2.62, angle=0.10471975512, e1=0.05236, e2=0.05236, K1=0.0, K2=0.0
)
b1g3c30a = latt.bend(
    "b1g3c30a", L=2.62, angle=0.10471975512, e1=0.05236, e2=0.05236, K1=0.0, K2=0.0
)
b1g5c01b = latt.bend(
    "b1g5c01b", L=2.62, angle=0.10471975512, e1=0.05236, e2=0.05236, K1=0.0, K2=0.0
)
b1g5c30b = latt.bend(
    "b1g5c30b", L=2.62, angle=0.10471975512, e1=0.05236, e2=0.05236, K1=0.0, K2=0.0
)
D0001 = latt.drif("D0001", L=4.65)
D0002 = latt.drif("D0002", L=0.166)
D0003 = latt.drif("D0003", L=0.802)
D0004 = latt.drif("D0004", L=0.184)
D0005 = latt.drif("D0005", L=0.186)
D0006 = latt.drif("D0006", L=0.166)
D0007 = latt.drif("D0007", L=0.58)
D0008 = latt.drif("D0008", L=0.9765)
D0009 = latt.drif("D0009", L=0.2015)
D0010 = latt.drif("D0010", L=0.654)
D0011 = latt.drif("D0011", L=0.184)
D0012 = latt.drif("D0012", L=0.184)
D0013 = latt.drif("D0013", L=0.504)
D0014 = latt.drif("D0014", L=0.3515)
D0015 = latt.drif("D0015", L=0.9765)
D0016 = latt.drif("D0016", L=0.591)
D0017 = latt.drif("D0017", L=0.166)
D0018 = latt.drif("D0018", L=0.5596)
D0019 = latt.drif("D0019", L=0.244)
D0020 = latt.drif("D0020", L=0.466)
D0021 = latt.drif("D0021", L=0.166)
D0022 = latt.drif("D0022", L=6.6)
D0023 = latt.drif("D0023", L=0.166)
D0024 = latt.drif("D0024", L=0.466)
D0025 = latt.drif("D0025", L=0.244)
D0026 = latt.drif("D0026", L=0.5596)
D0027 = latt.drif("D0027", L=0.166)
D0028 = latt.drif("D0028", L=0.591)
D0029 = latt.drif("D0029", L=0.9765)
D0030 = latt.drif("D0030", L=0.2015)
D0031 = latt.drif("D0031", L=0.654)
D0032 = latt.drif("D0032", L=0.184)
D0033 = latt.drif("D0033", L=0.184)
D0034 = latt.drif("D0034", L=0.504)
D0035 = latt.drif("D0035", L=0.3515)
D0036 = latt.drif("D0036", L=0.9765)
D0037 = latt.drif("D0037", L=0.58)
D0038 = latt.drif("D0038", L=0.166)
D0039 = latt.drif("D0039", L=0.186)
D0040 = latt.drif("D0040", L=0.184)
D0041 = latt.drif("D0041", L=0.802)
D0042 = latt.drif("D0042", L=0.166)
D0043 = latt.drif("D0043", L=4.65)
qh1g2c30a = latt.quad("qh1g2c30a", L=0.268, K1=-0.641957314648)
qh1g6c01b = latt.quad("qh1g6c01b", L=0.268, K1=-0.641957314648)
qh2g2c30a = latt.quad("qh2g2c30a", L=0.46, K1=1.43673057073)
qh2g6c01b = latt.quad("qh2g6c01b", L=0.46, K1=1.43673057073)
qh3g2c30a = latt.quad("qh3g2c30a", L=0.268, K1=-1.75355042529)
qh3g6c01b = latt.quad("qh3g6c01b", L=0.268, K1=-1.75355042529)
ql1g2c01a = latt.quad("ql1g2c01a", L=0.268, K1=-1.61785473561)
ql1g6c30b = latt.quad("ql1g6c30b", L=0.268, K1=-1.61785473561)
ql2g2c01a = latt.quad("ql2g2c01a", L=0.46, K1=1.76477357129)
ql2g6c30b = latt.quad("ql2g6c30b", L=0.46, K1=1.76477357129)
ql3g2c01a = latt.quad("ql3g2c01a", L=0.268, K1=-1.51868267756)
ql3g6c30b = latt.quad("ql3g6c30b", L=0.268, K1=-1.51868267756)
qm1g4c01a = latt.quad("qm1g4c01a", L=0.247, K1=-0.812234822773)
qm1g4c01b = latt.quad("qm1g4c01b", L=0.247, K1=-0.812234822773)
qm1g4c30a = latt.quad("qm1g4c30a", L=0.247, K1=-0.812234822773)
qm1g4c30b = latt.quad("qm1g4c30b", L=0.247, K1=-0.812234822773)
qm2g4c01a = latt.quad("qm2g4c01a", L=0.282, K1=1.22615465959)
qm2g4c01b = latt.quad("qm2g4c01b", L=0.282, K1=1.22615465959)
qm2g4c30a = latt.quad("qm2g4c30a", L=0.282, K1=1.22615465959)
qm2g4c30b = latt.quad("qm2g4c30b", L=0.282, K1=1.22615465959)
sh1 = latt.sext("sh1", L=0.2, K2=19.8329120997)
sh3 = latt.sext("sh3", L=0.2, K2=-5.85510841147)
sh4 = latt.sext("sh4", L=0.2, K2=-15.8209007067)
sl1 = latt.sext("sl1", L=0.2, K2=-13.2716060547)
sl2 = latt.sext("sl2", L=0.2, K2=35.6779214531)
sl3 = latt.sext("sl3", L=0.2, K2=-29.4608606061)
sm1a = latt.sext("sm1a", L=0.2, K2=-23.6806342393)
sm1b = latt.sext("sm1b", L=0.2, K2=-25.9460354618)
sm2 = latt.sext("sm2", L=0.25, K2=28.6431546915)
rfc = latt.rfca("rfc", L=0, voltage=3e6, freq=0.49968e9)

# === Beam Line sequence:
SC = [
    D0001,
    sh1,
    D0002,
    qh1g2c30a,
    D0003,
    qh2g2c30a,
    D0004,
    sh3,
    D0005,
    qh3g2c30a,
    D0006,
    sh4,
    D0007,
    b1g3c30a,
    D0008,
    qm1g4c30a,
    D0009,
    sm1a,
    D0010,
    qm2g4c30a,
    D0011,
    sm2,
    D0012,
    qm2g4c30b,
    D0013,
    sm1b,
    D0014,
    qm1g4c30b,
    D0015,
    b1g5c30b,
    D0016,
    ql3g6c30b,
    D0017,
    sl3,
    D0018,
    ql2g6c30b,
    D0019,
    sl2,
    D0020,
    ql1g6c30b,
    D0021,
    sl1,
    D0022,
    sl1,
    D0023,
    ql1g2c01a,
    D0024,
    sl2,
    D0025,
    ql2g2c01a,
    D0026,
    sl3,
    D0027,
    ql3g2c01a,
    D0028,
    b1g3c01a,
    D0029,
    qm1g4c01a,
    D0030,
    sm1a,
    D0031,
    qm2g4c01a,
    D0032,
    sm2,
    D0033,
    qm2g4c01b,
    D0034,
    sm1b,
    D0035,
    qm1g4c01b,
    D0036,
    b1g5c01b,
    D0037,
    sh4,
    D0038,
    qh3g6c01b,
    D0039,
    sh3,
    D0040,
    qh2g6c01b,
    D0041,
    qh1g6c01b,
    D0042,
    sh1,
    D0043,
]

# BL = 15*SC + [rfc]

ring = latt.cell(SC)
