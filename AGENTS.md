# keiba-predict

芝レースの単勝確率を推定し、Stage3 で市場オッズと統合して ROI を見る。詳細は `README.md`。

コマンドはリポジトリルートから実行する。Python 3.10+。依存は `requirements.txt` のみ。

## 境界

- Stage1 の特徴に、今走のオッズ・人気・着順・走破時間・上がり・通過順・今走スピード指数を入れない。除外列は `train/column_select.py`。
- 近走・expanding 成績・スピード指数は、当該レースより前の情報だけ。`ffill` で自分の結果や未来走を埋めない。
- スピード指数は距離 × 場id × 馬場の expanding z-score。後年を足しても過去行の値は変えない。窓ごとに fit し直さない。
- Stage2 はベース予測シェアの非負重み条件付きロジット（`train/race_logit.py`）。3着二値や `min(3p, 1)` に戻さない。
- Stage3 は `log p_model` と `log p_market` の条件付きロジットだけ。longshot の power と Isotonic は戻さない。重みは 0 以上。
- 賭けは同一レースで排反ケリー。最適化に失敗したレースは賭けない。レース露出の上限は 5%。閾値はテスト年で選ばない。
- レース NN（`train/race_nn.py`）の中身は、頼まれない限り変えない。出力はメタの一入力のまま。

## TDD

機能変更・バグ修正・契約の明確化は、実装より先にテストで契約を固定する。

1. 守る契約を一文で書く（リークなら「何を見てはいけないか」）。
2. 対応する `tests/test_*.py` に失敗するテストを足す。ファイル先頭の docstring も更新する。
3. `pytest tests/ -v` で RED を確認する。
4. `datasets/` または `train/` を最小限直す。
5. 全テストが GREEN になってからリファクタする（契約を変えない限りテストは触らない）。

例外: 年次 CSV と払戻が要る学習パイプラインの数値確認は、合成テストが通ったあとで行う。

## 評価とテスト
変更後は pytest tests/ -v を必ず回す

意思決定の順は、テスト年の `delta_nll`（市場 NLL − 結合 NLL）、その次に単勝 ROI。3着 AUC は主指標にしない。`delta_nll` が 0 以下の窓の ROI を市場優位の証拠にしない。

```bash
pytest tests/ -v
```

テストは合成データだけを使う。`datasets/raw_data` を読まない。各 `tests/test_*.py` の docstring に、壊れると何が困るか書く。

学習パイプラインの数値は、年次 CSV と払戻が揃ってから確認する。生 CSV はコミットしない。

Stage の説明が実装とずれたら `README.md` を合わせる。
