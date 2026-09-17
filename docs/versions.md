# バージョン固定記録

CLAUDE.md §2 に基づく再現性のための記録。**変更時は理由とともに追記し、既存の記録を上書きしない。**

---

## 2026-07-20 初回確定

### MFA本体

| 項目 | 値 |
|---|---|
| montreal-forced-aligner | **3.4.1** |
| インストール経路 | conda-forge（`mamba create -n mfa -c conda-forge montreal-forced-aligner`） |
| conda環境名 | `mfa` |
| kalpy | 0.10.4 |
| kaldi | 5.5.1172 |
| Python | 3.14.6 |
| プラットフォーム | macOS (Darwin 25.5.0), arm64 |

### japanese_mfa 音響モデル

| 項目 | 値 |
|---|---|
| ファイル | `~/Documents/MFA/pretrained_models/acoustic/japanese_mfa.zip` |
| **SHA-256** | `85928ffb1024486872a677a92e1fa94d2f997462d0b84c73a58aa9bb0e35179a` |
| サイズ | 92,191,596 bytes |
| モデル内 Version | 3.0 |
| Train date | 2024-02-08 02:27:14.171746 |
| Architecture | gmm-hmm |
| Phone type | triphone |
| Features | LDA あり |

### japanese_mfa 発音辞書

| 項目 | 値 |
|---|---|
| ファイル | `~/Documents/MFA/pretrained_models/dictionary/japanese_mfa.dict` |
| **SHA-256** | `4a0c66760576e4b7f3748f3169e7d5217c0135f857f0b5d34637c93a85fd1c91` |
| サイズ | 21,264,022 bytes |
| エントリ数 | 544,160 |

> **監査結果との同一性が確認済み。** `mapping/decisions.md`「長音の分割方針」付随決定3で実施した
> `V V` 表記揺れ監査（該当3,017エントリ / 3,055箇所）は、上記と同一SHA-256のファイルに対して
> 行われている。監査結果（`results/vv_audit*.tsv`）はこのモデル・辞書の組に対して有効。
> 辞書を更新した場合は監査をやり直すこと。

### 主要Pythonパッケージ

| パッケージ | バージョン |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| librosa | 0.11.0 |
| praatio | 6.2.2 |
| praat-parselmouth | 0.4.7 |
| textgrid | 1.6.1 |
| scikit-learn | 1.9.0 |
| sqlalchemy | 2.0.51 |
| postgresql | 18.4 |
| pynini | 2.1.7 |
| openfst | 1.8.4 |
| hdbscan | 0.8.44 |

完全な環境定義は `environment.yml`（`conda env export -n mfa --no-builds` の出力、243行）に固定。
再現は `conda env create -f environment.yml` で行う。

### R

| 項目 | 値 |
|---|---|
| R | 4.6.1 (2026-06-24) |
| renv | 1.2.3 |

パッケージのバージョンは `renv.lock` に固定。

### 参照した外部リソース（バージョン固定対象外だが記録）

| リソース | 版 |
|---|---|
| CSJ 分節音ラベリング仕様書 (`segment.pdf`) | Version 1.1 (2011-10-10) |
| CSJ XML文書仕様書 (`xml.pdf`) | Version 1.2 |
| CSJ 音響・言語モデル仕様書 (`asr.pdf`) | Ver.1.0 (2004-03-23) |
| MFA phone set ドキュメント | mfa-models リポジトリ main ブランチ（2026-07-20 参照） |
| jsut-label | v0.0.4 (2021-09-02、最新リリース) |

---

## 2026-07-20 追記: 日本語トークナイザの追加

japanese_mfa の使用には日本語トークナイザが必須であることが実行時に判明した
（未導入だと `mfa align` が ImportError で停止する）。conda-forge から追加導入した。

| パッケージ | バージョン |
|---|---|
| spacy | 3.8.14 |
| sudachipy | 0.6.11 |
| sudachidict-core | 20260116 |

`environment.yml` は追加後に再エクスポート済み。

---

## 変更履歴

（変更が生じたらここに追記する。形式: `## YYYY-MM-DD 変更内容` ＋ 変更理由）
