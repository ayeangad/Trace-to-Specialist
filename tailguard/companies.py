"""Fixed company list for the Tail Guard pool (spec P2.1)."""

COMPANIES = [
 ("Zephyr Holdings","ZPHY"),("Quorvex Industries","QRVX"),("Brightwater Systems","BRGH"),
 ("Kestrel Group","KSTR"),("Obsidian Labs","BSDN"),("Halcyon Corp","HLCY"),
 ("Marlowe Partners","MRLW"),("Tidewell Dynamics","TDWL"),("Corvana Holdings","CRVN"),
 ("Fennick Industries","FNNC"),("Glenmoor Systems","GLNM"),("Ironbark Group","RNBR"),
 ("Juniper Labs","JNPR"),("Kaltrix Corp","KLTR"),("Lumora Partners","LMRX"),
 ("Nethercott Dynamics","NTHR"),("Orwell Holdings","RWLL"),("Pembrook Industries","PMBR"),
 ("Quillon Systems","QLLN"),("Ravenmoor Group","RVNM"),("Solvane Labs","SLVN"),
 ("Thornbury Corp","THRN"),("Umberlyn Partners","MBRL"),("Vexley Dynamics","VXLY"),
 ("Wexford Holdings","WXFR"),("Xandria Industries","XNDR"),("Yarrowind Systems","YRRW"),
 ("Zoltrane Group","ZLTR"),("Ashgrove Labs","SHGR"),("Bramblefield Corp","BRMB"),
 ("Cindervale Partners","CNDR"),("Draycott Dynamics","DRYC"),("Emberlin Holdings","MBRN"),
 ("Foxhollow Industries","FXHL"),("Greywick Systems","GRYW"),("Holloway Group","HLLW"),
 ("Ivorell Labs","VRLL"),("Jasperline Corp","JSPR"),("Krynwood Partners","KRYN"),
 ("Larkspire Dynamics","LRKS"),("Mistholme Holdings","MSTH"),("Northgale Industries","NRTH"),
 ("Oakhaven Systems","KHVN"),("Pinecrest Group","PNCR"),("Quarrystone Labs","QRRY"),
 ("Rookwood Corp","RKWD"),
]
# index ranges (0-based, end exclusive)
GROUPS = {
  "STD": range(0, 18),   # used by S0, T4, T5 (prompt-only slices)
  "T1":  range(18, 24),
  "T2":  range(24, 30),
  "T3":  range(30, 36),
  "T6":  range(36, 42),
  "D1":  range(42, 46),  # drift only
}
# split by company inside each group (positions within the group's range)
SPLIT_POS = {
  "STD": {"fit": range(0, 6), "cal": range(6, 12), "test": range(12, 18)},
  "TX":  {"fit": range(0, 2), "cal": range(2, 4), "test": range(4, 6)},   # T1,T2,T3,T6
}
