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


def mine_components(pos: pl.DataFrame, s1: pl.DataFrame, min_count: int = 20) -> dict:
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
    best = (pairs.sort("len", descending=True).group_by("b").first()
                 .join(tot, on="b").filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= 0.5))
                 .join(freq1.rename({"c": "a", "f1": "fa"}), on="a", how="left")
                 .join(freq1.rename({"c": "b", "f1": "fb"}), on="b", how="left")
                 .with_columns(pl.col("fa").fill_null(0), pl.col("fb").fill_null(0))
                 .filter(pl.col("fa") > pl.col("fb")))
    alias = {}
    for b_c, a_c, n, t, fa in best.select("b", "a", "len", "tot", "fa").iter_rows():
        sim = JaroWinkler.normalized_similarity(skeleton(b_c.replace(" ", "")), skeleton(a_c.replace(" ", "")))
        # spelling variant / transliteration of the same place, or a state-code style alias
        # (same initial, very frequent canonical form, consistent replacement)
        code_like = min(len(b_c), len(a_c)) <= 3 and max(len(b_c.split()), len(a_c.split())) <= 2
        if sim >= 0.8 or (code_like and b_c[0] == a_c[0] and n >= 300 and n / t >= 0.6 and fa >= 2000):
            alias[b_c] = a_c
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


def load_maps() -> dict:
    with open(os.path.join(WORK, "maps.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
