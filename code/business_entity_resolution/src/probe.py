import sys, json, glob
sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from evaluate import decide
from prepare import load
cfg = json.load(open("work/m3_decision.json"))
te = pl.read_parquet("work/m3_test.parquet", columns=["a_idx", "b_idx", "p"])
cols = ["a_idx", "b_idx", "n_tset", "num_na", "num_nb", "num_first_logdiff"]
f = pl.concat([pl.read_parquet(x, columns=cols) for x in sorted(glob.glob("work/feat_test_*.parquet"))])
te = te.join(f, on=["a_idx", "b_idx"], how="left")
bucket = (pl.col("n_tset") >= 0.95) & (pl.col("num_na") > 0) & (pl.col("num_nb") > 0) & (pl.col("num_first_logdiff") > 0)
print("bucket pairs", te.filter(bucket).height, "uncertain", te.filter(bucket & (pl.col("p") > 0.1) & (pl.col("p") < 0.9)).height)
A = load("test", "A").select(pl.col("idx").alias("a_idx"), pl.col("entity_id").alias("source1_entity_id"))
Bid = load("test", "B").select(pl.col("idx").alias("b_idx"), pl.col("entity_id").alias("bid"))
for name, expr in (("down", pl.col("p") * 0.6), ("up", 1 - (1 - pl.col("p")) * 0.6)):
    d = te.with_columns(pl.when(bucket).then(expr).otherwise(pl.col("p")).alias("p")).select("a_idx", "b_idx", "p")
    ch = decide(d, cfg).join(Bid, on="b_idx")
    m = ch.group_by("a_idx").agg(pl.col("bid").sort().str.join(",").alias("matched_entity_ids"))
    res = A.join(m, on="a_idx", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    res.select("source1_entity_id", "matched_entity_ids").write_csv(f"output/leaderboard/probe_{name}_matching_results.tsv", separator="\t", quote_style="never")
    print(name, "pairs", ch.height, "non-empty S1", m.height)
base = decide(te.select("a_idx", "b_idx", "p"), cfg)
print("base pairs", base.height)
