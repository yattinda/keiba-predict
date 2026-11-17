"""契約: raw CSV の文字列パース結果が、人間が表計算できる値と一致する。"""

import os

import numpy as np
import pandas as pd
import pytest

from data_process import class_mapping, load_raw, parse_margin, parse_race_time_seconds


class TestRaceTimeSeconds:
    def test_colon_format(self):
        s = pd.Series(["1:34.5", "0:59.0"])
        out = parse_race_time_seconds(s)
        assert out.iloc[0] == pytest.approx(94.5)
        assert out.iloc[1] == pytest.approx(59.0)

    def test_garbage_becomes_nan(self):
        out = parse_race_time_seconds(pd.Series(["", "abc"]))
        assert out.isna().all()


class TestMargin:
  @pytest.mark.parametrize(
      "text,expected",
      [
          ("ハナ", 0.1),
          ("クビ", 0.3),
          ("1.1/2", 1.5),
          ("---", np.nan),
      ],
  )
  def test_known_tokens(self, text, expected):
      out = parse_margin(pd.Series([text])).iloc[0]
      if np.isnan(expected):
          assert np.isnan(out)
      else:
          assert out == pytest.approx(expected)


class TestClassMapping:
    def test_g1_is_ten(self):
        assert class_mapping("G1") == 10

    def test_maiden_is_one(self):
        assert class_mapping("未勝利") == 1

    def test_partial_match_one_win_not_newcomer(self):
        assert class_mapping("1勝クラス") == 4
        assert class_mapping("新馬") == 3


class TestLoadRaw:
    def test_reads_shift_jis_and_drops_bad_rows(self, tmp_path, minimal_raw_row):
        """有効行だけ残し、日付は Timestamp になる。"""
        raw_dir = tmp_path / "raw_data"
        raw_dir.mkdir()
        rows = [
            minimal_raw_row(),
            minimal_raw_row(着順="", 馬="欠損"),
            minimal_raw_row(日付="bad", 馬="日付不良"),
        ]
        path = raw_dir / "2020.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="SHIFT-JIS")

        dirname = str(tmp_path)
        out = load_raw(dirname, 2020, 2020)
        assert len(out) == 1
        assert pd.api.types.is_datetime64_any_dtype(out["日付"])
        assert int(out["着順"].iloc[0]) == 1

    def test_missing_year_file_is_skipped(self, tmp_path, minimal_raw_row):
        raw_dir = tmp_path / "raw_data"
        raw_dir.mkdir()
        pd.DataFrame([minimal_raw_row()]).to_csv(
            raw_dir / "2020.csv", index=False, encoding="SHIFT-JIS"
        )
        out = load_raw(str(tmp_path), 2019, 2021)
        assert len(out) == 1

    def test_no_files_exits(self, tmp_path):
        os.makedirs(tmp_path / "raw_data", exist_ok=True)
        with pytest.raises(SystemExit):
            load_raw(str(tmp_path), 2020, 2020)