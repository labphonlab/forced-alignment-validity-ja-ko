#!/usr/bin/env python3
"""原稿docx中の主要数値を、出典データと突き合わせる（2026-09-08 新設 / 09-09 改訂）。

修復前の数値が原稿に残る事故が2度起きたため、機械的に点検できるようにした。
2026-09-09 の改訂で原稿は Seoul Corpus を中核に再構成され、NIKL の聴取判定は
§4.4 の補足に退いた。点検対象もそれに合わせる。基盤は次の3つ。

  時間軸  seoul_boundary.measure()      … 人手 phoneme 層 vs MFA 出力
  範疇軸  seoul_compare.load()          … 人手発音形 vs MFA が選んだ変異形
  結合    seoul_two_axes                … 同一トークン上での両軸

使い方: python3 verify_manuscript_numbers.py [docx]
"""
import re
import statistics as st
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seoul_boundary as SB                 # noqa: E402
import seoul_compare as SC                  # noqa: E402

DEFAULT = ROOT / "docs/genkou_src/ishihara_ham_JAKLE2026-10_発表原稿.docx"

# 紙幅の都合で落としうる記述。欠けていても不一致とはしない。
OPTIONAL = {"20 ms以内", "境界点の総数", "同方向の話者数", "差", "隣接差"}


def docx_text(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode()
    return re.sub(r"<[^>]+>", "", x)


def run_numbers(run):
    """1つのランについて、原稿に載る数値を返す。"""
    SC.RUN = SB.RUN = ROOT / "seoul_corpus" / run
    rows = SC.load()
    meas = {m["utt"]: m for m in SB.measure("aligned")}
    bnd = {u: m["devs"] for u, m in meas.items()}
    devs = [d for ds in bnd.values() for d in ds]
    joined = [(r, bnd[r["utt"]]) for r in rows if r["utt"] in bnd]
    ok = [(r, d) for r, d in joined if (r["mfa"] == "applied") == bool(r["human"])]
    ng = [(r, d) for r, d in joined if (r["mfa"] == "applied") != bool(r["human"])]

    n = len(rows)
    ag = sum(1 for r in rows if (r["mfa"] == "applied") == bool(r["human"])) / n
    hp = sum(r["human"] for r in rows) / n
    fn = sum(1 for r in rows if r["human"] == 1 and r["mfa"] == "notapplied")
    fp = sum(1 for r in rows if r["human"] == 0 and r["mfa"] == "applied")

    return dict(
        rows=rows, devs=devs, joined=joined, ok=ok, ng=ng, meas=meas,
        n=n, ag=ag, hp=hp, fn=fn, fp=fp)


def expected():
    m = run_numbers(".mfa_seoul_v2")
    rows, devs, joined = m["rows"], m["devs"], m["joined"]
    meas = m["meas"]
    ok, ng = m["ok"], m["ng"]
    n, ag, hp, fn, fp = m["n"], m["ag"], m["hp"], m["fn"], m["fp"]
    from seoul_metrics import metrics, boot_ci, cells
    M = metrics(rows)
    ci = boot_ci(rows, lambda x: metrics(x)["diff"])
    spk = sorted({r["speaker"] for r in rows})
    same = sum(1 for sp in spk
               if cells([r for r in rows if r["speaker"] == sp])[1]
               > cells([r for r in rows if r["speaker"] == sp])[2])
    e = {
        "照合件数": f"{n:,}",
        "全体一致率": f"{ag:.1%}",
        "人手適用率": f"{hp:.1%}",
        "ベースラインとの差": f"{abs(100*(ag-max(hp,1-hp))):.1f}",
        "見落とし件数": f"{fn:,}",
        "不一致総数": f"{fn+fp:,}",
        "境界の平均": f"{st.mean(devs)*1000:.2f} ms",
        "境界の中央値": f"{st.median(devs)*1000:.2f} ms",
        "20 ms以内": f"{sum(1 for x in devs if x <= .020)/len(devs):.1%}",
        "境界点の総数": f"{len(devs):,}",
        "両軸トークン数": f"{len(joined):,}",
        "一致側の中央偏差": f"{st.median([st.median(d) for _, d in ok])*1000:.2f} ms",
        "不一致側の中央偏差": f"{st.median([st.median(d) for _, d in ng])*1000:.2f} ms",
        "話者数": f"{len(spk)}",
        "均衡正解率": f"{M['bal']:.1%}",
        "MCC": f"{M['mcc']:.3f}",
        "同方向の話者数": f"{same}話者" if same != len(spk) else f"{same}話者すべて",
        "ブートストラップ下限": f"{ci[0]:.1f}",
        "ブートストラップ上限": f"{ci[1]:.1f}",
    }

    # 2軸の差の区間（査読対応）と、対応率の群間比較
    import statistics as st2
    from seoul_two_axes import boot_median_diff
    J = [(r, meas[r["utt"]]) for r in rows if r["utt"] in meas]
    for field, tag in (("devs", ""),):
        use = [(r, m) for r, m in J if m[field]]
        okg = [m for r, m in use if (r["mfa"] == "applied") == bool(r["human"])]
        ngg = [m for r, m in use if (r["mfa"] == "applied") != bool(r["human"])]
        a = [st2.median(m[field]) for m in okg]
        b = [st2.median(m[field]) for m in ngg]
        lo, hi = boot_median_diff(
            [(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]), m[field])
             for r, m in use], field)
        e[f"{tag}差"] = f"{(st2.median(b)-st2.median(a))*1000:+.2f} ms".replace("+", "")
        e[f"{tag}CI下限"] = f"{lo:.2f}"
        e[f"{tag}CI上限"] = f"{hi:.2f}"
    # 見落としの割合
    e["見落とし割合"] = f"{fn/(fn+fp):.1%}"

    # 標的分節が対応から落ちえない候補（2仮説の音素類列が一致）での過程隣接分析と AUC
    import csv as _csv
    from phone_classes import IPA as _IPA
    from seoul_two_axes import auc_ci
    vp = {v["token"]: v for v in
          _csv.DictReader((SC.RUN / "variant_pronunciations.csv").open(encoding="utf-8"))}

    def _cls(q):
        return [_IPA.get(x) for x in str(q).split()]
    safe_rows = [r for r in rows if r["token"] in vp
                 and None not in _cls(vp[r["token"]]["citation"])
                 and _cls(vp[r["token"]]["citation"]) == _cls(vp[r["token"]]["sandhi"])]
    e["同一類候補数"] = f"{len(safe_rows):,}候補"
    safe = [(r, meas[r["utt"]]) for r in safe_rows if r["utt"] in meas]
    use = [(r, m) for r, m in safe if m["devs_target"]]
    ok_ = [st2.median(m["devs_target"]) for r, m in use if (r["mfa"] == "applied") == bool(r["human"])]
    ng_ = [st2.median(m["devs_target"]) for r, m in use if (r["mfa"] == "applied") != bool(r["human"])]
    lo, hi = boot_median_diff([(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]),
                                m["devs_target"]) for r, m in use], "devs_target")
    e["同一類隣接差"] = f"{(st2.median(ng_) - st2.median(ok_)) * 1000:+.2f} ms"
    e["同一類隣接CI"] = f"{lo:+.2f}〜{hi:+.2f}"
    au, alo, ahi = auc_ci([(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]),
                            st2.median(m["devs_target"])) for r, m in use])
    e["AUC(同一類隣接)"] = f"AUC {au:.2f}"
    e["AUC区間"] = f"{alo:.2f}〜{ahi:.2f}"
    au0, _, _ = auc_ci([(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]),
                         st2.median(m["devs"])) for r, m in J if m["devs"]])
    e["AUC(全標本)"] = f"全標本では{au0:.2f}"

    # 表1（過程別）: 各行の値が原稿にあるか
    JA = {"liaison": "連音化", "tensification": "濃音化", "aspiration": "激音化",
          "nasalization_obstruent": "鼻音化（阻害音由来）", "liquidization": "流音化",
          "nasalization_liquid": "鼻音化（流音由来）"}
    for cat, name in JA.items():
        mm = metrics([r for r in rows if r["cat"] == cat])
        e[f"表1 {name} MCC"] = f"{mm['mcc']:.3f}"
        e[f"表1 {name} 過小対過大"] = f"{mm['fn']:,}対{mm['fp']:,}"
    e["表1 全体 過小対過大"] = f"{fn:,}対{fp:,}"

    okc = [m["cov"] for r, m in J if (r["mfa"] == "applied") == bool(r["human"])]
    ngc = [m["cov"] for r, m in J if (r["mfa"] == "applied") != bool(r["human"])]
    e["対応率(一致)"] = f"{st2.mean(okc):.1%}"
    e["対応率(不一致)"] = f"{st2.mean(ngc):.1%}"
    return e


def main():
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    # 原稿は全角マイナス(U+2212)・全角ハイフンを使うので、比較前に正規化する
    txt = (docx_text(doc).replace("\u3000", " ").replace("\u00a0", " ")
           .replace("\u2212", "-").replace("\uff0d", "-").replace("\u2010", "-"))
    print(f"点検対象: {doc.name}\n")
    bad = 0
    for label, val in expected().items():
        hit = val in txt or val.replace(" ", "") in txt.replace(" ", "")
        if hit:
            print(f"  OK  {label:<18} {val}")
        elif label in OPTIONAL:
            print(f"  --  {label:<18} {val}（本文に記述なし・紙幅都合で可）")
        else:
            print(f"  NG  {label:<18} 期待値 {val}")
            bad += 1
    print(f"\n{bad} 件が原稿と食い違う（要確認）" if bad else "\nすべて一致した。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
