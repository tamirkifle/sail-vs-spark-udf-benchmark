"""Concrete execution backends for configs A-D."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator

from .common import (
    build_initialized_workload,
    compact_cfg,
    read_prompts_df,
    select_struct_output,
    workload_columns,
    workload_schema_sql,
    workload_struct_type,
    write_output,
)

try:
    from sail_vs_spark.engines.sail_session import _WORKER_ENV as _ENV
except ImportError:
    import os as _os

    _ENV: dict[str, str] = {
        k: _os.environ[k]
        for k in (
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "CUDA_VISIBLE_DEVICES",
        )
        if k in _os.environ
    }

TRACE_DIR = "/tmp/sail_traces"


def _trace_path() -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    return f"{TRACE_DIR}/trace_{os.getpid()}.jsonl"


def _new_timer(label: str):
    from sail_vs_spark.profiling.boundary_timer import BoundaryTimer

    return BoundaryTimer(label, enable_tracing=True)


def _save_timer_trace(timer: Any) -> None:
    timer.save_trace(_trace_path())


@dataclass(frozen=True)
class ExecutionBackend:
    execution_code: str

    def run_workload(
        self,
        spark: Any,
        workload: str,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str | None,
    ) -> int:
        runner = getattr(self, f"run_{workload}", None)
        if runner is None:
            raise ValueError(
                f"workload {workload!r} not implemented for config {self.execution_code}"
            )
        return runner(spark, parquet_path, cfg, output_parquet)


class SparkRowBackend(ExecutionBackend):
    def run_w0(
        self,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str | None,
    ) -> int:
        from pyspark.sql.functions import udf
        from pyspark.sql.types import LongType

        depth = int(cfg.get("workloads", {}).get("w0_chained", {}).get("depth", 1))
        trace_label = f"config_{self.execution_code.lower()}_w0"

        @udf(returnType=LongType())
        def stage(x):
            if not hasattr(stage, "_timer"):
                stage._timer = _new_timer(trace_label)
            with stage._timer.measure("DATA_TRANSFER_IN"):
                value = int(x)
            with stage._timer.measure("UDF_ROW_EXECUTION"):
                out = value + 1
            with stage._timer.measure("DATA_TRANSFER_OUT"):
                result = int(out)
            _save_timer_trace(stage._timer)
            return result

        df = read_prompts_df(spark, parquet_path)
        out = df
        for _ in range(depth):
            out = out.withColumn("prompt_id", stage("prompt_id"))
        return write_output(out, output_parquet)

    def _run_struct_apply(
        self,
        workload: str,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str,
    ) -> int:
        from pyspark.sql.functions import udf

        schema = workload_struct_type(workload, cfg)
        closure_cfg = compact_cfg(cfg)
        trace_label = f"config_{self.execution_code.lower()}_{workload}"

        @udf(returnType=schema)
        def apply_workload(pid, text):
            if not hasattr(apply_workload, "_timer"):
                apply_workload._timer = _new_timer(trace_label)
            with apply_workload._timer.measure("DATA_TRANSFER_IN"):
                pid_value = int(pid)
                text_value = str(text)
            with apply_workload._timer.measure("UDF_ROW_EXECUTION"):
                wl = build_initialized_workload(workload, closure_cfg)
                raw_out = wl.apply(pid_value, text_value)
            with apply_workload._timer.measure("DATA_TRANSFER_OUT"):
                out = tuple(raw_out)
            _save_timer_trace(apply_workload._timer)
            return out

        df = read_prompts_df(spark, parquet_path)
        out = select_struct_output(
            df.withColumn("_r", apply_workload("prompt_id", "prompt_text")),
            "_r",
            workload,
            cfg,
        )
        return write_output(out, output_parquet)

    def run_w1(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_struct_apply("w1", spark, parquet_path, cfg, output_parquet)

    def run_w2(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        closure_cfg = compact_cfg(cfg)
        trace_label = f"config_{self.execution_code.lower()}_w2"

        @udf(returnType=StringType())
        def gen_one(text):
            if not hasattr(gen_one, "_timer"):
                gen_one._timer = _new_timer(trace_label)
            with gen_one._timer.measure("DATA_TRANSFER_IN"):
                text_value = str(text)

            with gen_one._timer.measure("UDF_ROW_EXECUTION"):
                wl = build_initialized_workload("w2", closure_cfg)
                _, resp = wl.apply(0, text_value)

            with gen_one._timer.measure("DATA_TRANSFER_OUT"):
                result = str(resp)
            _save_timer_trace(gen_one._timer)
            return result

        df = read_prompts_df(spark, parquet_path)
        out = df.withColumn("response", gen_one("prompt_text")).select("prompt_id", "response")
        return write_output(out, output_parquet)

    def run_w3(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_struct_apply("w3", spark, parquet_path, cfg, output_parquet)

    def run_w4(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_struct_apply("w4", spark, parquet_path, cfg, output_parquet)


class SparkPandasBackend(ExecutionBackend):
    def run_w0(
        self,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str | None,
    ) -> int:
        import pandas as pd
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import LongType

        depth = int(cfg.get("workloads", {}).get("w0_chained", {}).get("depth", 1))
        trace_label = f"config_{self.execution_code.lower()}_w0"

        @pandas_udf(LongType())
        def stage(s):
            if not hasattr(stage, "_timer"):
                stage._timer = _new_timer(trace_label)
            with stage._timer.measure("DATA_TRANSFER_IN"):
                values = s.astype("int64", copy=False)
            with stage._timer.measure("UDF_BATCH_EXECUTION"):
                out = values + 1
            with stage._timer.measure("DATA_TRANSFER_OUT"):
                result = out.astype("int64", copy=False)
            _save_timer_trace(stage._timer)
            return result

        df = read_prompts_df(spark, parquet_path)
        out = df
        for _ in range(depth):
            out = out.withColumn("prompt_id", stage("prompt_id"))
        return write_output(out, output_parquet)

    def _run_batch_apply(
        self,
        workload: str,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str,
        *,
        trace_label: str | None = None,
    ) -> int:
        import pandas as pd
        from pyspark.sql.functions import pandas_udf

        schema = workload_struct_type(workload, cfg)
        closure_cfg = compact_cfg(cfg)
        timer_label = trace_label or f"config_{self.execution_code.lower()}_{workload}"

        @pandas_udf(schema)
        def apply_workload(pid_s, text_s):
            timer = _new_timer(timer_label)
            with timer.measure("DATA_TRANSFER_IN"):
                ids = pid_s.tolist()
                texts = text_s.tolist()
            with timer.measure("UDF_BATCH_EXECUTION"):
                wl = build_initialized_workload(workload, closure_cfg)
                out = wl.apply_batch(ids, texts)
            with timer.measure("DATA_TRANSFER_OUT"):
                result = pd.DataFrame(out)
            _save_timer_trace(timer)
            return result

        df = read_prompts_df(spark, parquet_path)
        out = select_struct_output(
            df.withColumn("_r", apply_workload("prompt_id", "prompt_text")),
            "_r",
            workload,
            cfg,
        )
        return write_output(out, output_parquet)

    def run_w1(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply("w1", spark, parquet_path, cfg, output_parquet)

    def run_w2(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        import pandas as pd
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import StringType

        closure_cfg = compact_cfg(cfg)
        trace_label = f"config_{self.execution_code.lower()}_w2"

        @pandas_udf(StringType())
        def gen_batch(text_s):
            timer = _new_timer(trace_label)
            with timer.measure("DATA_TRANSFER_IN"):
                texts = text_s.tolist()
            with timer.measure("UDF_BATCH_EXECUTION"):
                wl = build_initialized_workload("w2", closure_cfg)
                out = wl.apply_batch(range(len(texts)), texts)
            with timer.measure("DATA_TRANSFER_OUT"):
                result = pd.Series(out["response"])
            _save_timer_trace(timer)
            return result

        df = read_prompts_df(spark, parquet_path)
        out = df.withColumn("response", gen_batch("prompt_text")).select("prompt_id", "response")
        return write_output(out, output_parquet)

    def run_w3(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply(
            "w3",
            spark,
            parquet_path,
            cfg,
            output_parquet,
            trace_label=f"config_{self.execution_code.lower()}_w3",
        )

    def run_w4(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply("w4", spark, parquet_path, cfg, output_parquet)


class SailArrowBackend(ExecutionBackend):
    def run_w0(
        self,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str | None,
    ) -> int:
        import pyarrow as pa

        depth = int(cfg.get("workloads", {}).get("w0_chained", {}).get("depth", 1))
        trace_label = f"config_{self.execution_code.lower()}_w0"

        def stage(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
            timer = _new_timer(trace_label)
            for batch in batch_iter:
                with timer.measure("DATA_TRANSFER_IN"):
                    ids = batch.column("prompt_id")
                    vals = batch.column("value")
                with timer.measure("UDF_BATCH_EXECUTION"):
                    try:
                        import pyarrow.compute as pc

                        bumped = pc.add(vals, 1)
                    except Exception:
                        bumped = pa.array([int(x) + 1 for x in vals.to_pylist()], type=pa.int64())
                with timer.measure("DATA_TRANSFER_OUT"):
                    out_batch = pa.RecordBatch.from_arrays([ids, bumped], names=["prompt_id", "value"])
                _save_timer_trace(timer)
                yield out_batch

        df = read_prompts_df(spark, parquet_path).selectExpr("prompt_id", "prompt_id as value")
        out = df
        for _ in range(depth):
            out = out.mapInArrow(stage, "prompt_id long, value long")
        return write_output(out, output_parquet)

    def _run_batch_apply(
        self,
        workload: str,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str,
        *,
        trace_label: str | None = None,
    ) -> int:
        import pyarrow as pa

        schema = workload_schema_sql(workload, cfg)
        columns = workload_columns(workload, cfg)
        closure_cfg = compact_cfg(cfg)
        timer_label = trace_label or f"config_{self.execution_code.lower()}_{workload}"

        def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
            os.environ.update(_ENV)
            timer = _new_timer(timer_label)
            wl = build_initialized_workload(workload, closure_cfg)
            for batch in batch_iter:
                with timer.measure("DATA_TRANSFER_IN"):
                    ids = batch.column("prompt_id").to_pylist()
                    texts = batch.column("prompt_text").to_pylist()
                with timer.measure("UDF_BATCH_EXECUTION"):
                    out = wl.apply_batch(ids, texts)
                with timer.measure("DATA_TRANSFER_OUT"):
                    out_batch = pa.RecordBatch.from_arrays(
                        [
                            pa.array(out[name], type=_arrow_type(pa, dtype))
                            for name, dtype in columns
                        ],
                        names=[name for name, _ in columns],
                    )
                _save_timer_trace(timer)
                yield out_batch

        df = read_prompts_df(spark, parquet_path)
        out = df.mapInArrow(process, schema)
        return write_output(out, output_parquet)

    def run_w1(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply("w1", spark, parquet_path, cfg, output_parquet)

    def run_w2(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply("w2", spark, parquet_path, cfg, output_parquet)

    def run_w3(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply(
            "w3",
            spark,
            parquet_path,
            cfg,
            output_parquet,
            trace_label=f"config_{self.execution_code.lower()}_w3",
        )

    def run_w4(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_batch_apply("w4", spark, parquet_path, cfg, output_parquet)


class SailUdtfBackend(ExecutionBackend):
    def run_w0(
        self,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str | None,
    ) -> int:
        from pyspark.sql.functions import udtf

        depth = int(cfg.get("workloads", {}).get("w0_chained", {}).get("depth", 1))
        trace_label = f"config_{self.execution_code.lower()}_w0"

        @udtf(returnType="prompt_id long, value long")
        class TrivialUDTF:
            def __init__(self):
                self._timer = _new_timer(trace_label)

            def eval(self, pid: int, val: int):
                with self._timer.measure("DATA_TRANSFER_IN"):
                    pid_value = int(pid)
                    value = int(val)
                with self._timer.measure("UDF_ROW_EXECUTION"):
                    out = (pid_value, value + 1)
                with self._timer.measure("DATA_TRANSFER_OUT"):
                    result = (int(out[0]), int(out[1]))
                _save_timer_trace(self._timer)
                yield result

        spark.udtf.register("w0_stage", TrivialUDTF)

        df = read_prompts_df(spark, parquet_path)
        df.createOrReplaceTempView("prompts")
        sql = "SELECT p.prompt_id, p.prompt_id as value FROM prompts p"
        for _ in range(depth):
            sql = f"SELECT u.prompt_id, u.value FROM ({sql}) t, LATERAL w0_stage(t.prompt_id, t.value) u"
        out = spark.sql(sql)
        return write_output(out, output_parquet)

    def _run_buffered_udtf(
        self,
        workload: str,
        spark: Any,
        parquet_path: str,
        cfg: dict[str, Any],
        output_parquet: str,
        *,
        trace_label: str | None = None,
    ) -> int:
        from pyspark.sql.functions import udtf

        schema = workload_schema_sql(workload, cfg)
        columns = [name for name, _ in workload_columns(workload, cfg)]
        closure_cfg = compact_cfg(cfg)
        fn_name = f"{self.execution_code.lower()}_{workload}"
        timer_label = trace_label or f"config_{self.execution_code.lower()}_{workload}"

        @udtf(returnType=schema)
        class BufferedUDTF:
            def __init__(self):
                self._wl = None
                self._buffer: list[tuple[int, str]] = []
                self._timer = _new_timer(timer_label)

            def eval(self, prompt_id: int, prompt_text: str):
                if self._wl is None:
                    import os

                    os.environ.update(_ENV)
                    self._wl = build_initialized_workload(workload, closure_cfg)
                with self._timer.measure("DATA_TRANSFER_IN"):
                    row = (int(prompt_id), str(prompt_text))
                self._buffer.append(row)

            def terminate(self):
                if not self._buffer:
                    return
                ids = [row[0] for row in self._buffer]
                texts = [row[1] for row in self._buffer]
                with self._timer.measure("UDF_BATCH_EXECUTION"):
                    out = self._wl.apply_batch(ids, texts)
                with self._timer.measure("DATA_TRANSFER_OUT"):
                    rows = [
                        tuple(out[name][idx] for name in columns)
                        for idx in range(len(out[columns[0]]))
                    ]
                _save_timer_trace(self._timer)
                for row in rows:
                    yield row

        spark.udtf.register(fn_name, BufferedUDTF)
        df = read_prompts_df(spark, parquet_path)
        df.createOrReplaceTempView("prompts")
        out = spark.sql(f"SELECT u.* FROM prompts, LATERAL {fn_name}(prompt_id, prompt_text) u")
        return write_output(out, output_parquet)

    def run_w1(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_buffered_udtf("w1", spark, parquet_path, cfg, output_parquet)

    def run_w2(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_buffered_udtf(
            "w2",
            spark,
            parquet_path,
            cfg,
            output_parquet,
            trace_label=f"config_{self.execution_code.lower()}_w2",
        )

    def run_w3(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_buffered_udtf("w3", spark, parquet_path, cfg, output_parquet)

    def run_w4(self, spark: Any, parquet_path: str, cfg: dict[str, Any], output_parquet: str) -> int:
        return self._run_buffered_udtf("w4", spark, parquet_path, cfg, output_parquet)


def _arrow_type(pa: Any, dtype: str):
    mapping = {
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "string": pa.string(),
    }
    return mapping[dtype]
