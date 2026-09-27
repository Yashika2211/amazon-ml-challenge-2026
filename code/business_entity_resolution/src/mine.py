"""Mine alias tables from TRAIN positive pairs (no external data).

  indic       transliterated token -> Latin token (e.g. praivet -> private)
  comp_alias  address component -> canonical component (state name/code/native, city aliases)
  generic     token -> insertion rate in positives (noise tokens such as center, services)
"""
from __future__ import annotations

import json
import warnings
import os
import re
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from io_utils import WORK, load_source, load_truth
from normalize import normalize

warnings.filterwarnings("ignore", category=DeprecationWarning)

_SKEL = [("ph", "f"), ("x", "ks"), ("ck", "k"), ("c", "k"), ("q", "k"), ("w", "v"),
         ("z", "j"), ("y", "i"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"),
         ("kh", "k"), ("gh", "g"), ("m", "n")]


def skeleton(tok: str) -> str:
    """Transliteration-robust consonant skeleton (also used as a phonetic blocking key)."""
    s = tok.lower()
    for a, b in _SKEL:
        s = s.replace(a, b)
    if not s:
        return s
    s = s[0] + re.sub(r"[aeiouh]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


def _positive_frame(first_pass: dict) -> pl.DataFrame:
    truth = load_truth()
    s1 = first_pass[1].select("entity_id", pl.all().exclude("entity_id").name.prefix("a_"))
    s23 = pl.concat([first_pass[2], first_pass[3]]).select("entity_id", pl.all().exclude("entity_id").name.prefix("b_"))
    return (truth.join(s1, left_on="s1_id", right_on="entity_id")
                 .join(s23, left_on="m_id", right_on="entity_id"))


def mine_indic(pos: pl.DataFrame, min_count: int = 3) -> dict:
    sub = (pos.filter(pl.col("b_native") & ~pl.col("a_native"))
              .select("a_name_tok", "b_name_tok").unique())
    counts: dict = defaultdict(Counter)
    for a_tok, b_tok in sub.iter_rows():
        a_sk = [(w, skeleton(w)) for w in a_tok]
        used = set()
        scored = []
        for t in b_tok:
            ts = skeleton(t)
            for j, (w, ws) in enumerate(a_sk):
                scored.append((JaroWinkler.normalized_similarity(ts, ws), t, j, w))
        scored.sort(reverse=True)
        done = set()
        for sc, t, j, w in scored:
            if sc < 0.6 or t in done or j in used:
                continue
            done.add(t); used.add(j)
            counts[t][w] += 1
    out = {}
    for t, c in counts.items():
        w, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= 0.5 and w != t:
            out[t] = w
    return out


def mine_components(pos: pl.DataFrame, s1: pl.DataFrame, min_count: int = 20, region_rule: bool = False) -> dict:
    """Alias address components that systematically replace one another in positives."""
    short = lambda c: c.list.eval(pl.element().filter(
        ~pl.element().str.contains(r"[0-9]") & (pl.element().str.count_matches(" ") <= 2)))
    p = pos.select(pl.int_range(pl.len()).alias("pid"), short(pl.col("a_addr_comp")).alias("a"),
                   short(pl.col("b_addr_comp")).alias("b"))
    a = p.select("pid", "a").explode("a").drop_nulls()
    b = p.select("pid", "b").explode("b").drop_nulls()
    a_only = a.join(b, left_on=["pid", "a"], right_on=["pid", "b"], how="anti")
    b_only = b.join(a, left_on=["pid", "b"], right_on=["pid", "a"], how="anti")
    pairs = b_only.join(a_only, on="pid").group_by("b", "a").len()
    tot = b_only.group_by("b").len().rename({"len": "tot"})
    freq1 = (s1.select(pl.col("addr_comp").explode().alias("c")).group_by("c").len()
               .rename({"len": "f1"}))
    pairs = (pairs.filter(pl.col("len") >= min_count).join(tot, on="b")
                  .join(freq1.rename({"c": "a", "f1": "fa"}), on="a", how="left")
                  .join(freq1.rename({"c": "b", "f1": "fb"}), on="b", how="left")
                  .with_columns(pl.col("fa").fill_null(0), pl.col("fb").fill_null(0))
                  .filter(pl.col("fa") > pl.col("fb")))
    best: dict = {}
    for b_c, a_c, n, t, fa in pairs.select("b", "a", "len", "tot", "fa").iter_rows():
        if n / t < 0.3:
            continue
        sim = max(fuzz.ratio(b_c, a_c), fuzz.ratio(skeleton(b_c.replace(" ", "")), skeleton(a_c.replace(" ", "")))) / 100
        # spelling variant / transliteration of the same place, or a state-code style alias
        code_like = min(len(b_c), len(a_c)) <= 3 and max(len(b_c.split()), len(a_c.split())) <= 2
        ok = sim >= 0.8 or (code_like and b_c[0] == a_c[0] and n >= 300 and n / t >= 0.6 and fa >= 2000)
        # region-level synonym (e.g. departement vs region in the last address field): both sides are
        # very frequent components and substitute each other consistently
        if region_rule and n >= 1000 and n / t >= 0.6 and fa >= 0.01 * s1.height:
            ok = True
        score = n * (0.5 + sim)
        if ok and score > best.get(b_c, (0, None))[0]:
            best[b_c] = (score, a_c)
    alias = {b: v[1] for b, v in best.items()}
    for k in list(alias):  # resolve chains a->b->c
        seen = {k}
        v = alias[k]
        while v in alias and v not in seen:
            seen.add(v); v = alias[v]
        alias[k] = v
    return {k: v for k, v in alias.items() if k != v}


def mine_generic(pos: pl.DataFrame, s23_core: pl.Series, min_occ: int = 50) -> dict:
    p = pos.select(pl.int_range(pl.len()).alias("pid"), pl.col("a_core_tok").alias("a"), pl.col("b_core_tok").alias("b"))
    a = p.select("pid", "a").explode("a").drop_nulls()
    b = p.select("pid", "b").explode("b").drop_nulls()
    ins = b.join(a, left_on=["pid", "b"], right_on=["pid", "a"], how="anti").group_by("b").len().rename({"len": "ins"})
    occ = b.group_by("b").len().rename({"len": "occ"})
    g = occ.join(ins, on="b", how="left").with_columns(pl.col("ins").fill_null(0)).filter(pl.col("occ") >= min_occ)
    g = g.with_columns((pl.col("ins") / pl.col("occ")).alias("rate")).filter(pl.col("rate") >= 0.2)
    return dict(zip(g["b"].to_list(), g["rate"].round(3).to_list()))


def main() -> dict:
    first = {s: normalize(load_source("train", s)) for s in (1, 2, 3)}
    pos = _positive_frame(first)
    print("positives", pos.height)
    indic = mine_indic(pos)
    print("indic map", len(indic), list(indic.items())[:30])
    # component aliases are mined and applied per country ("tn" is Tennessee in the US, Tamil Nadu in India)
    comp = {}
    for c in pos["a_country"].unique().to_list():
        comp[c] = mine_components(pos.filter(pl.col("a_country") == c), first[1].filter(pl.col("country") == c))
        print("component aliases", c, len(comp[c]), list(comp[c].items())[:25])
    # generic tokens must be measured after indic mapping, otherwise transliterations look like insertions
    first = {s: normalize(load_source("train", s), {"indic": indic}) for s in (1, 2, 3)}
    pos = _positive_frame(first)
    generic = mine_generic(pos, None)
    top = sorted(generic.items(), key=lambda kv: -kv[1])[:40]
    print("generic", len(generic), top)
    maps = {"indic": indic, "comp_alias": comp, "generic": generic}
    with open(os.path.join(WORK, "maps.json"), "w") as f:
        json.dump(maps, f)
    return maps


def mine_pseudo(pred_path: str, maps: dict, min_occ: int = 50, country_generic: bool = True) -> dict:
    """Alias / generic-token mining for countries with no training data, from test pseudo-positives.

    Pseudo-positive: one-to-one best pair for a record with (p >= 0.9) or (address anchored:
    same first house number, address token Jaccard >= 0.7, name overlap >= 0.5, rank 1 for the
    record). Address anchoring keeps pairs whose names carry unseen noise words, so their
    insertion rates are not biased away. No labels are used.
    """
    import glob
    from prepare import load
    A = load("test", "A").select(pl.col("idx").alias("a_idx"), "country", "core_tok", "addr_comp")
    unseen = [c for c in A["country"].unique().to_list() if c not in maps["comp_alias"]]
    if not unseen:
        return maps
    A = A.filter(pl.col("country").is_in(unseen))
    B = load("test", "B").select(pl.col("idx").alias("b_idx"), "entity_id", "country", "core_tok", "addr_comp")
    B = B.filter(pl.col("country").is_in(unseen))
    # address components re-derived from the raw files WITHOUT any alias for these countries, so the
    # mining is independent of aliases that an earlier pseudo pass may already have applied
    from io_utils import load_source
    from normalize import normalize_addresses
    raw1 = load_source("test", 1).filter(pl.col("country").is_in(unseen))
    raw23 = pl.concat([load_source("test", s_) for s_ in (2, 3)]).filter(pl.col("country").is_in(unseen))
    ra = raw1.select("entity_id").with_columns(normalize_addresses(raw1, None)["addr_comp"])
    rb = raw23.select("entity_id").with_columns(normalize_addresses(raw23, None)["addr_comp"])
    A = (load("test", "A").select(pl.col("idx").alias("a_idx"), "entity_id").join(ra, on="entity_id")
         .join(A.drop("addr_comp"), on="a_idx"))
    B = B.drop("addr_comp").join(rb, on="entity_id").select("b_idx", "core_tok", "addr_comp")
    pred = pl.read_parquet(pred_path, columns=["a_idx", "b_idx", "p"]).join(A.select("a_idx", "country"), on="a_idx")
    cols = ["a_idx", "b_idx", "num_first_eq", "at_wjac", "nt_wov", "rank_b"]
    feats = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(WORK, "feat_test_*.parquet")))])
    d = pred.join(feats, on=["a_idx", "b_idx"], how="left")
    d = d.filter((pl.col("p") >= 0.9) | ((pl.col("num_first_eq") == 1) & (pl.col("at_wjac") >= 0.7)
                                          & (pl.col("nt_wov") >= 0.5) & (pl.col("rank_b") == 1)))
    d = d.sort("p", descending=True).unique("b_idx", keep="first")
    pos = (d.select("a_idx", "b_idx", pl.col("country").alias("a_country"))
             .join(A.select("a_idx", pl.col("core_tok").alias("a_core_tok"), pl.col("addr_comp").alias("a_addr_comp")), on="a_idx")
             .join(B.select("b_idx", pl.col("core_tok").alias("b_core_tok"), pl.col("addr_comp").alias("b_addr_comp")), on="b_idx"))
    # country-specific generic rates (applied only to pairs of that country, so US/India features
    # stay exactly as in training). Two label-free estimates, take the max:
    #   * insertion rate in pseudo-positives
    #   * over-representation in S2/S3 vs S1: a token seen e times more often than the S1 base
    #     rate implies an insertion share of 1 - 1/e among its S2/S3 occurrences
    Afull = load("test", "A").select("country", "core_tok")
    Bfull = load("test", "B").select("country", "core_tok")
    maps.setdefault("generic_by_country", {})
    for c in unseen:
        pc = pos.filter(pl.col("a_country") == c)
        s1c = A.filter(pl.col("country") == c)
        maps["comp_alias"][c] = mine_components(pc, s1c, region_rule=True)
        g = mine_generic(pc, None, min_occ)
        a_c = Afull.filter(pl.col("country") == c)
        b_c = Bfull.filter(pl.col("country") == c)
        fa = a_c.select(pl.col("core_tok").list.unique().explode().alias("t")).drop_nulls().group_by("t").len().rename({"len": "na"})
        fb = b_c.select(pl.col("core_tok").list.unique().explode().alias("t")).drop_nulls().group_by("t").len().rename({"len": "nb"})
        r = (fb.join(fa, on="t", how="left").with_columns(pl.col("na").fill_null(0))
               .with_columns(((pl.col("nb") / b_c.height) / ((pl.col("na") + 1) / a_c.height)).alias("r")))
        base = r.filter(pl.col("na") >= min_occ)["r"].median()
        r = r.filter(pl.col("nb") >= min_occ).with_columns((1 - base / pl.col("r")).clip(0, 1).alias("ov"))
        ov = {t: round(v, 3) for t, v in zip(r["t"].to_list(), r["ov"].to_list()) if v >= 0.2}
        gc = {**{k: v for k, v in maps["generic"].items()}}
        for k, v in list(g.items()) + list(ov.items()):
            gc[k] = max(gc.get(k, 0.0), v)
        if country_generic:  # public leaderboard: with 0.9807 vs without 0.976 (LOCO US->India had suggested the opposite)
            maps["generic_by_country"][c] = gc
        new = sorted(((k, v) for k, v in gc.items() if maps["generic"].get(k, 0) < v), key=lambda kv: -kv[1])
        print(f"[pseudo] {c}: {pc.height:,} pseudo-positives, aliases {maps['comp_alias'][c]}", flush=True)
        print(f"[pseudo] {c}: base over-representation {base:.3f}; raised generic tokens {len(new)}: {new[:40]}", flush=True)
    maps["pseudo_countries"] = unseen
    with open(os.path.join(WORK, "maps.json"), "w") as f:
        json.dump(maps, f)
    return maps


def load_maps() -> dict:
    with open(os.path.join(WORK, "maps.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
