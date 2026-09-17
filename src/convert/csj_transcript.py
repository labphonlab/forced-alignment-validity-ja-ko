# SCOPE: shared
"""CSJ転記テキストのタグ除去。MFA入力用の平文と、タグ位置情報を返す。

出典: 『日本語話し言葉コーパス』転記テキストの仕様 Version 1.0（小磯ほか）5節・表6〜表10。

なぜ正規表現の逐次置換では駄目か
--------------------------------
§5.2 が **入れ子** を明示的に許している（`(L (? 市外, 市内) 通話)`、`(K たち (F いー) ばな;橘)`）。
同一範囲の重複付与もある（`(M (A エス;Ｓ))`）。優先順位は
`A,K < ?単 < W < B < F,D,D2 < R < O < M < X < L < 笑`（右ほど外側）。
したがって**括弧を再帰的に解析する**必要があり、逐次置換は壊れる。

どの表記を入力にするか
----------------------
**基本形（漢字仮名交じり）を使う。発音形（カタカナ）は使わない。**
japanese_mfa辞書は標準表記でキーされており、発音形のカタカナは事実上すべて未知語に
なることを辞書照会で確認済み（docs/decisions-log.md「CSJのMFA入力は基本形を使う」）。

タグの出現先（表6 付与対象欄）:
    基本形のみ : (A) (K)
    発音形のみ : (W) (B) (笑) (泣) (咳) (L)
    両方       : (F) (D) (D2) (?) (M) (O) (R) (X)

算用数字は基本形の `(A 読み;表記)` に読みが入る（`(A 千九百九十五;１９９５)`）。
**セミコロンの左が読み**なので第1要素を採ればよい。JSUT側の `expand_numerals()` と
同じ漢数字の形に揃う。

⚠ (F) の完全抽出には発音形も要る
--------------------------------
§5「タグ(W)」の但し書き:

    タグ (F), <FV> は原則, 基本形, 発音形の両方に出現するが, (W) の左項（実際の発音部）に
    (F), <FV> が生じた場合, (W) の右項, 及び基本形にこれらのタグは現れない.
    そのため, (F), <FV> を完全に抽出する為には, 基本形（だけ）ではなく発音形を参照する必要がある.

**促音のF/D末尾除外判定（mapping/decisions.md「促音の閉鎖区間の帰属」付随決定2）は
(F) の網羅的な把握を前提とするため、基本形だけでは取りこぼす。**
`parse()` は基本形・発音形の双方を受け取れるようにし、(F)/(D) の位置は両方から集める。

使い方
------
    from csj_transcript import parse
    r = parse(basic="(F あの) 千九百九十五年", phonetic="(F アノ) センキュウヒャクキュウジューゴネン")
    r.text        # -> "あの 千九百九十五年"  ... タグを除去した平文（MFA入力）
    r.spans       # -> [TagSpan(tag='F', start=0, end=3, ...), ...]  出力text上の位置
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 文字範囲を指定するタグ（表6・上段）。括弧つきで出現する。
RANGE_TAGS = {
    "F", "D", "D2", "?", "M", "O", "R", "X", "A", "K", "W", "B", "笑", "泣", "咳", "L",
}

# セミコロンで2要素を持つタグ。左右どちらを採るかがタグごとに違う。
#   A: 読み;表記      -> 読み（左）。算用数字が漢数字の読みになる
#   K: 読み;表記      -> 読み（左）
#   W: 実際の発音;正  -> 発音形にのみ出現。基本形を使う限り現れない
#   B: 誤読み;正読み  -> 同上
SEMICOLON_TAKE_LEFT = {"A", "K", "W", "B"}

# 記号型タグ（表6・下段）。括弧ではなく <> で囲まれる。
SYMBOL_TAG_RE = re.compile(r"<(FV|VN|H|Q|笑|咳|息|P(?::[0-9.\-]+)?)>")

# 促音のF/D末尾除外判定に必要なタグ（mapping/decisions.md）
TRACKED_TAGS = {"F", "D", "D2"}


@dataclass
class TagSpan:
    """除去後テキスト上でのタグの範囲。"""

    tag: str
    start: int      # 出力text上の開始位置（含む）
    end: int        # 出力text上の終了位置（含まない）
    source: str     # "basic" | "phonetic"
    depth: int = 0  # 入れ子の深さ（0が最外）

    @property
    def is_tracked(self) -> bool:
        return self.tag in TRACKED_TAGS


@dataclass
class ParseResult:
    text: str
    spans: list[TagSpan] = field(default_factory=list)
    symbols: list[tuple[str, int]] = field(default_factory=list)  # (記号名, text上の位置)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_masked_region(self) -> bool:
        """(R) を含むか。**含む転記単位(IPU)は音声全体が白色雑音のため丸ごと除外する。**

        transcription.pdf §5「タグ(R)」:
        「音声ファイルに対しても、本タグの対象となる転記単位の全体を白色雑音で置換する
          処理を施している。つまり、転記単位の一部でも本タグの範囲であれば、
          その転記単位全体が白色雑音で置換される。」
        """
        return any(s.tag == "R" for s in self.spans)

    def tracked_spans(self) -> list[TagSpan]:
        """(F)/(D)/(D2) のみ。促音の除外判定に使う。"""
        return [s for s in self.spans if s.is_tracked]

    def ends_with_tracked(self, pos: int) -> str | None:
        """位置 pos が (F)/(D)/(D2) の**末尾**にあたるならそのタグ名を返す。

        `mapping/decisions.md`「促音の閉鎖区間の帰属」付随決定2:
        転記タグF/D末尾に出現する促音は先行母音と融合され母音境界が失われるため除外する。
        """
        for s in self.tracked_spans():
            if s.end - 1 == pos:
                return s.tag
        return None


def _split_tag_head(body: str) -> tuple[str | None, str]:
    """タグ本体の先頭からタグ名を切り出す。`F あの` -> ("F", "あの")。

    タグ名でなければ (None, body) を返す（ただの括弧書き）。
    """
    # 伏せ字化されると **タグ名直後の空白も × に置換される** ため、
    # `(R×××××)` のように空白なしで × が続く形があり得る（transcription.pdf p.10 実例）。
    m = re.match(r"^(D2|[FD?MORXAKWB]|笑|泣|咳|L)(?=[\s×]|$)", body)
    if not m:
        # 値なしの (?) は本体が空
        if body.strip() == "?":
            return "?", ""
        return None, body
    return m.group(1), body[m.end():].lstrip()


def _parse_segment(s: str, out: list[str], spans: list[TagSpan],
                   symbols: list[tuple[str, int]], warnings: list[str],
                   source: str, depth: int) -> None:
    """再帰下降で1区間を解析し、平文を out に追記する。"""
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]

        if ch == "(":
            # 対応する閉じ括弧を深さを数えて探す
            level = 0
            j = i
            while j < n:
                if s[j] == "(":
                    level += 1
                elif s[j] == ")":
                    level -= 1
                    if level == 0:
                        break
                j += 1
            if j >= n:
                warnings.append(f"閉じ括弧が見つからない: {s[i:i+20]!r}")
                out.append(s[i:])
                return

            body = s[i + 1 : j]
            tag, rest = _split_tag_head(body)
            start = sum(len(x) for x in out)

            if tag is None:
                # タグではない括弧。中身をそのまま展開する
                _parse_segment(body, out, spans, symbols, warnings, source, depth + 1)
            elif tag in SEMICOLON_TAKE_LEFT and ";" in rest:
                # (A 読み;表記) -> 読み（左）を採る。読み側に入れ子がありうる
                left = rest.split(";", 1)[0]
                _parse_segment(left, out, spans, symbols, warnings, source, depth + 1)
            elif tag == "?":
                # (? 候補1, 候補2) -> 第1候補。(?) は値なしで空
                if rest.strip():
                    first = re.split(r"[,，、]", rest, maxsplit=1)[0]
                    _parse_segment(first, out, spans, symbols, warnings, source, depth + 1)
            elif tag == "R":
                # (R) の範囲は伏せ字化され、**音声も白色雑音で置換されている**。
                # アラインメント不能なので本文を落とす（span は start==end で残す）。
                # さらに §5「タグ(R)」は「転記単位の一部でも本タグの範囲であれば、
                # その転記単位全体が白色雑音で置換される」と規定しており、
                # **(R) を含むIPUは丸ごと評価対象外にする必要がある**（呼び出し側の責務）。
                pass
            else:
                _parse_segment(rest, out, spans, symbols, warnings, source, depth + 1)

            end = sum(len(x) for x in out)
            if tag is not None:
                spans.append(TagSpan(tag=tag, start=start, end=end, source=source, depth=depth))
            i = j + 1
            continue

        m = SYMBOL_TAG_RE.match(s, i)
        if m:
            # 記号型タグ。文字は生まないので位置だけ記録する
            name = m.group(1).split(":", 1)[0]
            symbols.append((name, sum(len(x) for x in out)))
            i = m.end()
            continue

        if ch == ")":
            warnings.append(f"対応しない閉じ括弧: 位置 {i}")
            i += 1
            continue

        out.append(ch)
        i += 1


def strip_tags(text: str, source: str = "basic") -> ParseResult:
    """1つの表記（基本形または発音形）からタグを除去する。"""
    out: list[str] = []
    spans: list[TagSpan] = []
    symbols: list[tuple[str, int]] = []
    warnings: list[str] = []
    _parse_segment(text, out, spans, symbols, warnings, source, 0)
    return ParseResult(text="".join(out), spans=spans, symbols=symbols, warnings=warnings)


def parse(basic: str, phonetic: str | None = None) -> ParseResult:
    """基本形からMFA入力用の平文を作り、(F)/(D) の位置を基本形・発音形の双方から集める。

    発音形を渡す理由は本モジュールの docstring を参照（(W) 左項に生じた (F) は
    基本形に現れないため、基本形だけでは (F) を取りこぼす）。
    発音形側のタグ位置は**発音形テキスト上の位置**であり、基本形の位置とは対応しない。
    `TagSpan.source` で区別すること。
    """
    result = strip_tags(basic, source="basic")
    if phonetic:
        ph = strip_tags(phonetic, source="phonetic")
        # 基本形に現れないタグ（主に (W) 左項内の (F)）を補う
        basic_tags = {s.tag for s in result.spans}
        for s in ph.spans:
            if s.is_tracked and s.tag not in basic_tags:
                result.spans.append(s)
        result.warnings.extend(ph.warnings)
    return result


if __name__ == "__main__":
    # 仕様書の実例で検証する（transcription.pdf 5節・図1）
    cases = [
        ("(F あの) 千九百九十五年", "基本形のフィラー"),
        ("(A 千九百九十五;１９９５)年の", "算用数字→読み（左）を採る"),
        ("(A 二十六，三五;２６．３５)", "読みにカンマを含む"),
        ("(K たち (F いー) ばな;橘)", "入れ子: K の内側に F"),
        ("(M (A エス;Ｓ))", "同一範囲: M の内側に A"),
        ("(L (? 市外, 市内) 通話)", "入れ子: L の内側に ?複数候補"),
        ("(? 字数)の", "?値1つ"),
        ("(? 次数, 実数)", "?値複数→第1候補"),
        ("(?) で", "?値なし"),
        ("(D 形) 形式の", "言い差し"),
        ("ソレデ<H>、スィ<H>ゴイ", "記号型タグ <H>"),
        ("ハン<P:00333.068-00333.442>トシ<H>", "<P> は時刻つき"),
        ("(W ギーツ;ギジュツ)", "W: 発音形のみだが念のため"),
    ]
    print(f"{'入力':<38} {'出力':<22} タグ")
    print("-" * 84)
    for src, note in cases:
        r = strip_tags(src)
        tags = ",".join(f"{s.tag}[{s.start}:{s.end}]" for s in r.spans)
        syms = ",".join(f"<{n}>@{p}" for n, p in r.symbols)
        print(f"{src:<38} {r.text:<22} {tags} {syms}")
        for w in r.warnings:
            print(f"    [warn] {w}")
