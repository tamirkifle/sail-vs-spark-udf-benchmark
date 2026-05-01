"""Tests for execution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from sail_vs_spark.execution.common import write_output


def test_write_output_counts_parquet_without_spark_reread(tmp_path):
    class _FakeSparkSession:
        class _Read:
            def parquet(self, path):
                raise AssertionError("write_output should not re-read through Spark")

        read = _Read()

    class _FakeWriter:
        def __init__(self, root: Path):
            self.root = root

        def mode(self, mode: str):
            return self

        def parquet(self, output_parquet: str):
            out_dir = Path(output_parquet)
            out_dir.mkdir(parents=True, exist_ok=True)
            table = pa.table({"x": [1, 2, 3]})
            pq.write_table(table, out_dir / "part-00000.parquet")

    class _FakeDF:
        def __init__(self, root: Path):
            self.write = _FakeWriter(root)
            self.sparkSession = _FakeSparkSession()
            self.count_calls = 0

        def count(self):
            self.count_calls += 1
            return 99

    df = _FakeDF(tmp_path)
    out_dir = tmp_path / "output"
    n = write_output(df, str(out_dir))

    assert n == 3
    assert df.count_calls == 0
