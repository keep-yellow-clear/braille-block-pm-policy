"""법정동 단위 주차구역 공백: 점자블록 견인이 많은 동에 주차구역이 있는가."""
import pandas as pd

from config import HOTSPOT_MIN_BRAILLE


def dong_gap(tows, parking):
    b = tows[tows["braille"]]
    key = ["gu", "dong"]
    g = b.groupby(key).size().rename("점자블록 견인").to_frame()
    per_addr = b.groupby(key + ["addr"]).size()
    g["2건+ 지번 수"] = (per_addr >= 2).groupby(key).sum()
    g["2건+ 지번의 점자블록 견인"] = per_addr[per_addr >= 2].groupby(key).sum()
    g["전체 견인"] = tows.groupby(key).size()
    g["주차구역(보완 전)"] = parking.groupby(["gu", "dong_lot"]).size().rename_axis(key)
    g["주차구역(보완 후)"] = parking.groupby(["gu", "dong"]).size().rename_axis(key)
    g = g.fillna(0).astype(int).sort_values("점자블록 견인", ascending=False)
    g.index.names = ["구", "법정동"]

    hot = g[g["점자블록 견인"] >= HOTSPOT_MIN_BRAILLE]
    summary = {"점자블록 견인 동 수": len(g),
               f"견인 {HOTSPOT_MIN_BRAILLE}건+ 동 수": len(hot),
               f"견인 {HOTSPOT_MIN_BRAILLE}건+ 동의 견인 비중": hot["점자블록 견인"].sum() / g["점자블록 견인"].sum(),
               "그중 주차구역 0곳(보완 전)": int((hot["주차구역(보완 전)"] == 0).sum()),
               "그중 주차구역 0곳(보완 후)": int((hot["주차구역(보완 후)"] == 0).sum())}
    return g, summary
