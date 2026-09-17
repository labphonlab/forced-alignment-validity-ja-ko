# トラブルシューティング

実行時にはまった箇所と対処を記録する。同じ問題で二度止まらないようにするのが目的。
設計判断は `docs/decisions-log.md`、音素対応の判断は `mapping/decisions.md` に書く。

---

## 評価（sequence_align）

### JSUT一致度が omission 98% に縮退する（2026-07-25 修正）

- **症状**: JSUT basic5000 パイプラインを完走させると、`jsut_alignment.tsv` の類型が
  omission 98.31% / displacement 1.63% という縮退結果になる（本来は displacement が大半）。
  参照・仮説の時刻自体はほぼ一致しているのに一致が立たない。
- **原因**: `sequence_align.py` の `ref_windows()` が IPU 窓の時間範囲を**先頭単位の span**で
  固定していた。CSJ は全単位が IPU 時刻（`ipu_start`/`ipu_end`）を持つので問題ないが、
  **JSUT は発話単位でポーズ区切りが無く `ipu_start`/`ipu_end` が空**のため、フォールバックで
  先頭単位の `t_start`/`t_end`（＝最初の1音素）が窓になり、窓が1音素幅に縮退。`assign_hyp` が
  その窓に重なる仮説単位（＝最初の1音素）しか拾わず、残り全音素が omission になっていた。
- **修正**: 窓の範囲を「その窓に属する全単位の min(start)/max(end)」から張るように変更。
  IPU 時刻がある場合（CSJ）は全単位が同一 IPU 時刻を持つので min/max はその IPU 範囲に一致し
  **従来と完全に同一**（`csj_alignment.tsv` を再生成して 1,420,594 行がバイト一致することを確認）。
  IPU 時刻が無い場合（JSUT）は発話全体が窓になる。
- **修正後**: JSUT 一致度は displacement 92.56% / substitution 0.92% / omission 6.25% /
  insertion 0.28% と健全化（metric_type=concordance。精度ではなくアライナ間一致度）。
- **影響範囲**: JSUT 経路のみ。CSJ の主分析結果（論文の数値）は不変。

---

## MFA

### `sqlite3.DatabaseError: database disk image is malformed`

**症状**: `mfa align` が次のエラーで停止する。

```
sqlalchemy.exc.DatabaseError: (sqlite3.DatabaseError) database disk image is malformed
[SQL: UPDATE pronunciation SET "disambiguation" = b."disambiguation" FROM temp_pronunciation AS b ...]
```

**原因**: MFAは `~/Documents/MFA/` 配下にコーパスごとのSQLiteデータベースとjoblibキャッシュを
作る。実行が途中で異常終了すると、このデータベースが不整合な状態で残り、次回以降の実行が
すべて失敗する。`--clean` オプションを付けても回復しない（`--clean` は出力側を消すだけで、
破損したDBは消えない）。

**対処**: キャッシュを削除してから再実行する。

```bash
rm -rf "$HOME/Documents/MFA/corpus" "$HOME/Documents/MFA/joblib_cache"
```

`pretrained_models/` は消さないこと（音響モデルと辞書の再ダウンロードになる）。

**予防**: 実行を中断する場合は、可能なら `Ctrl-C` で正常終了させる。強制終了した後は
上記の削除を習慣づける。

---

### `ImportError: Please install Japanese support via conda install -c conda-forge spacy sudachipy sudachidict-core`

**症状**: `mfa align` が日本語コーパスに対してトークナイズ段階で停止する。

**原因**: japanese_mfa は形態素解析器を必要とするが、`montreal-forced-aligner` の
conda パッケージには含まれていない。

**対処**:

```bash
conda activate mfa
mamba install -y -c conda-forge spacy sudachipy sudachidict-core
```

導入したバージョンは `docs/versions.md` に記録済み（2026-07-20 時点で
spacy 3.8.14 / sudachipy 0.6.11 / sudachidict-core 20260116）。
`environment.yml` も再エクスポートすること。

---

### アラインメント結果に `spn` が現れる

**症状**: MFA出力の phones tier に `spn`（spoken noise）が入り、その区間に対応する
参照側の音素がすべて omission として計上される。

**原因**: 入力テキストに**MFAが読みに展開できない表記**（算用数字、記号、未知語）が
含まれている。MFAはそれを未知語として1トークンの `spn` に潰す。

```
入力: システィナ礼拝堂は、１４７３年に、…
参照: … s e N y o N hy a k u n a n a j u u s a N n e N …   （20音素）
MFA : … spn …                                             （1トークン）
```

**対処**: **算用数字を漢数字に展開してから**MFAに渡す。`src/align/run_mfa.py` が
既定で行う（`expand_numerals()`、`１４７３` → `千四百七十三`）。

**⚠ かな読みを入力にするのは誤り。** 一度その方針を採ったが実測で否定された。
japanese_mfa辞書は**標準表記（漢字・カタカナ）でキーされている**ため、全ひらがなを
与えると内容語の大半が未知語になり、`spn` がかえって増える（41件に増加、omissionは倍増）。
`しすてぃな` `まれーしあ` `ばちかん` はいずれも辞書に0件、`水` `礼拝堂` は存在する。

CSJでは発音形（`SUW/@PhoneticTranscription`）を使う想定だが、**それがMFA辞書のキーと
整合するかは未検証**である（同じ罠がありうる）。フェーズ3で検証すること。

**検出方法**: 変換後の中間表現で `spn` を数える。ゼロでなければ入力正規化を疑う。

```bash
awk -F'\t' 'NR>1 && $12=="spn" {print $1}' results/mfa_*_units.tsv | sort -u
```

---

## パイプライン

### 中間表現の列がずれて見える

`column -t` は連続する空フィールドを詰めて表示するため、`ipu_id` などが空の行では
列がずれて見える。**データの問題ではない。** 生のフィールド数を確認すること。

```bash
awk -F'\t' 'NR<=3{print NR": "NF" fields"}' results/xxx.tsv
```

---

### `UnmappedLabelError` が出る

**これは設計どおりの挙動である。** 写像表（`mapping/*.tsv`）にないラベルを黙って
通さないようにしてある。規約差がMFAの誤りとして静かに計上される事故を防ぐため。

対処は、当該ラベルを `mapping/` に追記するか、`mapping/decisions.md` で除外を決めること。
**変換スクリプト側に恒等写像のフォールバックを置いてはならない**（jsut-labelはローマ字、
MFAはIPAで、単独音素も対応が異なるため素通しは必ず誤る）。
