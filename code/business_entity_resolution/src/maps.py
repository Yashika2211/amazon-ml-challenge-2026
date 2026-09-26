"""Small hand-written normalization maps (allowed by the rules; documented in README).

Everything country-specific beyond these maps (state names/codes/native script, city
aliases, transliterated tokens, generic noise tokens) is mined from TRAIN positives.
"""

# ---- business names -------------------------------------------------------
LEGAL_MAP = {
    # English / India
    "pvt": "private", "pvtltd": "private limited", "prv": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmt": "limited", "ld": "limited",
    "inc": "incorporated", "incorp": "incorporated", "incorporation": "incorporated",
    "corp": "corporation", "corpn": "corporation",
    "co": "company", "cos": "company", "companies": "company", "cmpny": "company",
    "l.l.c": "llc", "pllc": "pllc", "llp": "llp", "plc": "plc", "lp": "lp", "pc": "pc",
    "opc": "opc", "huf": "huf",
    # France
    "societe": "societe", "ste": "societe", "sté": "societe", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl", "snc": "snc", "scp": "scp",
    "scop": "scop", "sca": "sca", "selarl": "selarl", "gie": "gie", "groupe": "group",
    # same legal form, single-member variant (SASU = SAS unipersonnelle, EURL = SARL unipersonnelle)
    "sasu": "sas", "eurl": "sarl", "cie": "company", "compagnie": "company", "ei": "ei",
    # French name abbreviations
    "frs": "freres", "ets": "etablissements",
}
LEGAL_TOKENS = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "pllc", "llp", "plc",
    "lp", "pc", "opc", "huf", "public", "societe", "sarl", "sas", "sasu", "sa", "sci", "eurl",
    "snc", "scp", "scop", "sca", "selarl", "gie", "ei",
}
# connector words that carry no identity
NAME_STOP = {"and", "the", "of", "&", "de", "du", "des", "la", "le", "les", "et", "d", "l"}
HONORIFIC = {"m", "s", "ms", "mr", "mrs", "sri", "shri", "shree", "smt", "dr"}

# ---- addresses ------------------------------------------------------------
STREET_MAP = {
    # English
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "bd": "blvd", "blv": "blvd",
    "highway": "hwy", "place": "pl", "circle": "cir", "parkway": "pkwy", "terrace": "ter",
    "square": "sq", "trail": "trl", "suite": "ste", "ste": "ste", "apartment": "apt",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft",
    "center": "ctr", "centre": "ctr", "expressway": "expy", "freeway": "fwy",
    # India
    "near": "nr", "opposite": "opp", "oppo": "opp", "floor": "flr",
    "building": "bldg", "bldng": "bldg", "society": "soc", "sector": "sec", "nagar": "ngr",
    "colony": "col", "district": "dist", "dt": "dist", "taluk": "tq", "taluka": "tq",
    "tehsil": "teh", "village": "vill", "vil": "vill", "post": "po", "cross": "crs",
    "main": "main", "industrial": "indl", "ind": "indl", "estate": "est", "complex": "cmplx",
    "apartments": "apt", "appartment": "apt", "appartments": "apt",
    # France
    "rue": "rue", "r": "rue", "chemin": "ch", "impasse": "imp", "allee": "all",
    "route": "rte", "faubourg": "fg", "sainte": "ste", "cedex": "", "bis": "", "ter": "",
    "quai": "qu", "cours": "crs", "residence": "res",
}
ADDR_STOP = {
    "no", "nos", "number", "num", "h", "hno", "house", "door", "dno", "plot", "flat", "shop",
    "unit", "the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l", "c", "o",
    "po", "box", "region", "suburban", "dcp", "null", "na", "nil", "none", "at", "via",
}
NULL_TOKENS = ["<null>", "null", "n/a", "none", "nil", "not available", "unknown"]
