"""Record normalization (names + addresses), vectorized with polars.

Produces, per record:
  name_tok    list[str]  full normalized name tokens (legal forms canonical)
  core_tok    list[str]  name tokens minus legal forms / stopwords / honorifics
  alt_tok     list[str]  core tokens of the DBA/AKA part (empty when none)
  name_norm   str        " ".join(name_tok)
  core        str        " ".join(core_tok)
  compact     str        core without spaces (catches domain / hashtag names)
  legal       str        sorted legal-form tokens
  native      bool       name originally in a non-Latin script
  addr_tok    list[str]  normalized address tokens (bag)
  addr_comp   list[str]  normalized comma components (after mined aliases)
  addr_num    list[str]  number tokens without leading zeros
  addr        str        " ".join(addr_tok)
  addr_empty  bool
"""
from __future__ import annotations

import unicodedata

import polars as pl
from anyascii import anyascii

from maps import ADDR_STOP, HONORIFIC, LEGAL_MAP, LEGAL_TOKENS, NAME_STOP, STREET_MAP

NON_LATIN = r"[ऀ-෿]"  # Devanagari .. Sinhala (all Indic blocks)
DBA_RE = r"^(.*?)\s+(?:d\s*\.?\s*b\s*\.?\s*a\.?|a\s*\.?\s*k\s*\.?\s*a\.?|t\s*/\s*a|trading as|doing business as)\s+(.+)$"
CONF_FROM = ["0", "1", "3", "4", "5", "8"]
CONF_TO = ["o", "l", "e", "a", "s", "b"]


def _ascii_fold(s: pl.Series) -> pl.Series:
    """NFKC + anyascii on the non-ASCII values only (unique values, then mapped back)."""
    mask = s.str.contains(r"[^\x00-\x7F]")
    if not mask.any():
        return s
    uniq = s.filter(mask).unique()
    mapping = {u: anyascii(unicodedata.normalize("NFKC", u)) for u in uniq.to_list()}
    return s.replace(mapping)


def _tok_map(expr: pl.Expr, mapping: dict) -> pl.Expr:
    return expr.list.eval(pl.element().replace(mapping))


def normalize_names(df: pl.DataFrame, indic_map: dict | None = None) -> pl.DataFrame:
    raw = df["business_name"]
    native = raw.str.contains(NON_LATIN)
    folded = _ascii_fold(raw).str.to_lowercase()
    d = pl.DataFrame({"n": folded, "native": native})
    n = pl.col("n")
    d = d.with_columns(
        n.str.replace_all(r"^[\s\-<>\.#*~_|=+,:;!\"'`()\[\]]+", "")
         .str.replace(r"^m\s*/\s*s\.?\s*", "")
         .str.replace(r"^(https?://)?(www\.)?", "")
         .str.replace(r"\.(com|co\.in|in|net|org|co|fr|us|biz|info)$", "")
         .str.replace_all(r"[&+]", " and ")
    )
    d = d.with_columns(n.str.extract_groups(DBA_RE).alias("g"))
    d = d.with_columns(
        pl.when(pl.col("g").struct.field("1").is_not_null()).then(pl.col("g").struct.field("1")).otherwise(n).alias("main"),
        pl.col("g").struct.field("2").fill_null("").alias("alt"),
    ).drop("g")

    def tokens(col: str) -> pl.Expr:
        t0 = (pl.col(col).str.replace_all(r"'s\b", "s").str.replace_all(r"[^a-z0-9]+", " ")
             .str.strip_chars().str.split(" ").list.eval(pl.element().filter(pl.element() != "")))
        # mined transliteration dictionary applies only to names that had native script
        t = pl.when(pl.col("native")).then(_tok_map(t0, indic_map)).otherwise(t0) if indic_map else t0
        # digit/letter confusion on mixed tokens that are not ordinals (3rd, 4th ...)
        t = t.list.eval(
            pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9]")
                    & ~pl.element().str.contains(r"^[0-9]+(st|nd|rd|th)$"))
            .then(pl.element().str.replace_many(CONF_FROM, CONF_TO))
            .otherwise(pl.element()))
        t = _tok_map(t, LEGAL_MAP)
        # legal map may yield two words ("private limited")
        t = t.list.join(" ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        return t.list.unique(maintain_order=True)

    drop_core = list(LEGAL_TOKENS | NAME_STOP)
    d = d.with_columns(tokens("main").alias("name_tok"), tokens("alt").alias("alt_all"))
    d = d.with_columns(
        pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(drop_core))).alias("core_tok"),
        pl.col("alt_all").list.eval(pl.element().filter(~pl.element().is_in(drop_core))).alias("alt_tok"),
        pl.col("name_tok").list.eval(pl.element().filter(pl.element().is_in(list(LEGAL_TOKENS))))
          .list.sort().list.join(" ").alias("legal"),
    )
    # honorifics only at the start of the core ("shri", "m s")
    d = d.with_columns(
        pl.when(pl.col("core_tok").list.len() > 1)
          .then(pl.col("core_tok").list.eval(pl.element().filter(
              ~(pl.element().is_in(list(HONORIFIC)) & (pl.int_range(pl.len()) == 0)))))
          .otherwise(pl.col("core_tok")).alias("core_tok"))
    d = d.with_columns(
        pl.col("name_tok").list.join(" ").alias("name_norm"),
        pl.col("core_tok").list.join(" ").alias("core"),
        pl.col("core_tok").list.join("").alias("compact"),
    )
    return d.select("name_tok", "core_tok", "alt_tok", "name_norm", "core", "compact", "legal", "native")


def normalize_addresses(df: pl.DataFrame, comp_alias: dict | None = None,
                        tok_alias: dict | None = None) -> pl.DataFrame:
    folded = _ascii_fold(df["business_address"]).str.to_lowercase()
    d = pl.DataFrame({"a": folded})
    a = pl.col("a")
    d = d.with_columns(
        a.str.replace_all(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnot available\b", " ")
         .str.replace_all(r"(\d)\s*-\s*(\d)", "$1 $2"))
    comps = (pl.col("a").str.split(",").list.eval(
        pl.element().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
        .str.split(" ").list.eval(pl.element().filter(pl.element() != "").replace(STREET_MAP)
                                  .str.replace(r"^0+([0-9])", "$1")
                                  .filter((pl.element() != "") & ~pl.element().is_in(list(ADDR_STOP))))
        .list.join(" ")).list.eval(pl.element().filter(pl.element() != "")))
    d = d.with_columns(comps.alias("addr_comp"))
    if comp_alias:
        d = d.with_columns(pl.col("addr_comp").list.eval(pl.element().replace(comp_alias)))
    d = d.with_columns(pl.col("addr_comp").list.unique(maintain_order=True))
    tok = pl.col("addr_comp").list.join(" ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    if tok_alias:
        tok = tok.list.eval(pl.element().replace(tok_alias))
    d = d.with_columns(tok.list.unique(maintain_order=True).alias("addr_tok"))
    d = d.with_columns(
        pl.col("a").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
          .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).alias("addr_num"),
        pl.col("addr_tok").list.join(" ").alias("addr"),
    )
    d = d.with_columns((pl.col("addr_tok").list.len() == 0).alias("addr_empty"))
    return d.select("addr_tok", "addr_comp", "addr_num", "addr", "addr_empty")


def normalize(df: pl.DataFrame, maps: dict | None = None) -> pl.DataFrame:
    maps = maps or {}
    names = normalize_names(df, maps.get("indic"))
    addrs = normalize_addresses(df, maps.get("comp_alias"), maps.get("tok_alias"))
    return pl.concat([df.select("entity_id", "country"), names, addrs], how="horizontal")
