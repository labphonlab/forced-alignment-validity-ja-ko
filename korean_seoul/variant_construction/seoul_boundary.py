#!/usr/bin/env python3
"""Seoul Corpus 上で境界の時間軸を測る（2026-09-09 新設 / 09-10 全面改訂）。

**2026-09-10 の改訂（査読指摘への対応・重要）**
初版は「人手の各境界について MFA 出力の最も近い境界までの距離」を取っていた。
これは危険である。MFA が別の変異形を選んで**分節数の違う音素列**を出した場合、
対応していない境界でも近傍に何かがあれば小さい値が出る。しかも分節数が増える側に
系統的に有利に働くため、**「変異形を誤った語でも境界は正確」という中心的結果が
指標そのものから生じている**可能性があった。

そこで既定を対応付け方式に変えた。
  1. 人手ラベル(Seoul ローマ字)と MFA 出力(IPA)を共通の粗い音素類へ写す
     （喉頭素性は潰す。そこは範疇軸の争点であり対応付けの手がかりにしない）
  2. 編集距離でアライメントする
  3. **クラスが一致した対の境界だけ**を比較する
最近傍方式も感度分析のために残してある（method="nearest"）。

測定は**対象語区間に限る**。この実行は対象語だけを辞書に載せるので周囲は spn になり、
発話全体で測ると 83 ms という無意味な値になる。
"""
import csv
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phone_classes import HAND, IPA, align, is_phone   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get("SEOUL_CORPUS_DIR", "seoul_corpus/raw")).expanduser()
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
RUN = ROOT / "seoul_corpus" / (_args[0] if _args else ".mfa_seoul")
PAD = 0.15


def tiers_utf16(path):
    t = path.read_bytes().decode("utf-16")
    out = {}
    for b in re.split(r"item\s*\[\d+\]:", t)[1:]:
        m = re.search(r'name\s*=\s*"([^"]*)"', b)
        if m:
            out.setdefault(m.group(1), []).append(re.findall(
                r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', b))
    return out


def tiers_utf8(path):
    t = path.read_text(encoding="utf-8")
    out = {}
    for b in re.split(r"item\s*\[\d+\]:", t)[1:]:
        m = re.search(r'name = "([^"]*)"', b)
        if m:
            out[m.group(1)] = re.findall(
                r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', b)
    return out


def syl_phone_index(prono_hangul: str, syl_index: int):
    """発音形（ハングル）で、第 syl_index 音節の末尾までに何音素あるかを返す。

    対象の音韻過程が働くのは第 syl_index 音節と次の音節の境目なので、
    この値が**その境目の音素インデックス**になる。Seoul の表記では
    初声ㅇは音素を持たず、子音群の終声（ㄺ 等）は1音素として数える。
    """
    n = 0
    for k, ch in enumerate(prono_hangul):
        c = ord(ch) - 0xAC00
        if not (0 <= c < 11172):
            return None
        on, co = c // 588, c % 28
        n += (0 if on == 11 else 1) + 1 + (0 if co == 0 else 1)
        if k == syl_index:
            return n
    return None


def measure(method="aligned"):
    """語ごとの測定結果を dict の列で返す。

    devs        対応した全分節の境界偏差
    devs_target 音韻過程が働く音節境界に隣接する分節だけの境界偏差
    cov         人手分節のうち対応がついた割合
    """
    cands = list(csv.DictReader((RUN / "pilot_candidates.csv").open(encoding="utf-8")))
    cache = {}
    out, skipped = [], 0
    for c in cands:
        fname = c["utterance_id"].rsplit("_", 1)[0]
        src = CORPUS / f"{fname}.TextGrid"
        al = RUN / "aligned" / c["speaker_id"] / f"{c['utterance_id']}.TextGrid"
        if not src.exists() or not al.exists():
            skipped += 1
            continue
        if fname not in cache:
            cache[fname] = tiers_utf16(src)
        T = cache[fname]
        dur = float(c["duration_sec"])
        cand = [u for u in T.get("utt.ortho.", [[]])[0] if u[2].strip() == c["transcript"]]
        u = next((u for u in cand
                  if abs((float(u[1]) + PAD - max(0.0, float(u[0]) - PAD)) - dur) < 0.01),
                 cand[0] if len(cand) == 1 else None)
        if u is None:
            skipped += 1
            continue
        off = max(0.0, float(u[0]) - PAD)

        A = tiers_utf8(al)
        ph = [(float(a), float(b), t) for a, b, t in A.get("phones", [])]
        w = next((x for x in A.get("words", [])
                  if x[2] == c["token"]
                  and any(is_phone(a[2]) and float(x[0]) - 1e-6 <= a[0]
                          and a[1] <= float(x[1]) + 1e-6 for a in ph)), None)
        if w is None:
            skipped += 1
            continue
        ws, we = float(w[0]), float(w[1])

        # MFA 側：対象語の中の実音素だけ（絶対時刻に戻す）
        mfa = [(off + a, off + b, t) for a, b, t in ph
               if is_phone(t) and ws - 1e-6 <= a and b <= we + 1e-6]
        # 人手側：同区間に中心がある実音素だけ。<SIL> <LAUGH-…> <NOISE> は落とす
        lo, hi = off + ws, off + we
        hand = [(float(a), float(b), t) for a, b, t in T.get("phoneme", [[]])[0]
                if is_phone(t) and lo - 0.03 <= (float(a) + float(b)) / 2 <= hi + 0.03]
        if len(mfa) < 2 or len(hand) < 2:
            skipped += 1
            continue

        if method == "nearest":
            # 旧方式（感度分析用）。対応関係を問わず最も近い境界を拾う。
            pts = sorted({t for a, b, _ in mfa for t in (a, b)})
            hb = sorted({t for a, b, _ in hand for t in (a, b)})
            d = [min(abs(h - x) for x in pts) for h in hb]
            out.append(dict(utt=c["utterance_id"], cat=c["change_type"],
                            devs=d, devs_target=[], cov=1.0))
            continue

        ha = [HAND.get(t) for _, _, t in hand]
        ma = [IPA.get(t) for _, _, t in mfa]
        if any(x is None for x in ha) or any(x is None for x in ma):
            skipped += 1          # 未知記号がある語は落とす（黙って混ぜない）
            continue
        pairs = align(ha, ma)
        if not pairs:
            skipped += 1
            continue
        k = syl_phone_index(c["prono"], int(c["syllable_boundary"]))
        d, dt = [], []
        for i, j in pairs:
            e = (abs(hand[i][0] - mfa[j][0]), abs(hand[i][1] - mfa[j][1]))
            d += e
            # 過程が働く音節境界に接する2分節（前音節の末尾と次音節の頭）
            if k is not None and i in (k - 1, k):
                dt += e
        out.append(dict(utt=c["utterance_id"], cat=c["change_type"],
                        devs=d, devs_target=dt, cov=len(pairs) / len(hand)))
    print(f"（照合不能で除外 {skipped}）", file=sys.stderr)
    return out


def main():
    for method in ("aligned", "nearest"):
        res = measure(method)
        devs = [d for r in res for d in r["devs"]]
        n = len(devs)
        cov = st.mean([r["cov"] for r in res])
        label = "対応付け方式（既定）" if method == "aligned" else "最近傍方式（旧・感度分析）"
        print(f"\n【{label}】 境界 {n:,} 点 / 対象語 {len(res):,}"
              + (f" / 人手分節の対応率 {cov:.1%}" if method == "aligned" else ""))
        print(f"  平均 {st.mean(devs)*1000:6.2f} ms   中央値 {st.median(devs)*1000:6.2f} ms", end="")
        for th in (0.020, 0.050):
            print(f"   {int(th*1000)}ms以内 {sum(1 for x in devs if x <= th)/n:5.1%}", end="")
        print()
        per = [(r["cat"], st.median(r["devs"])) for r in res]
        for cat in sorted({t for t, _ in per}):
            v = [d for t, d in per if t == cat]
            print(f"    {cat:<24} n={len(v):>5}  {st.median(v)*1000:6.2f} ms")
    print("\n※ 人手転記者間の境界偏差は 9.04 ms、McAuliffe et al.(2026) の報告値は 14.78 ms。")


if __name__ == "__main__":
    main()
