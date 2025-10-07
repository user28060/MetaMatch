# golden_cli.py
from __future__ import annotations
from pathlib import Path
import click
import pandas as pd

from golden_tools import golden_matrix_s1xs2, golden_matrix_s1xs1, _non_index_columns
from pipeline import read_csv_any


@click.command(context_settings={"show_default": True})
@click.option("--source-csv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--target-csv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--golden-json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--s1xs1/--s1xs2", default=False, help="Export S1×S1 (true) or S1×S2 (false).")
@click.option("--out-csv", required=True, type=click.Path(dir_okay=False))
def main(source_csv: str, target_csv: str, golden_json: str, s1xs1: bool, out_csv: str) -> None:
    """Build and export a 0/1 golden matrix for inspection."""
    df_src = read_csv_any(source_csv)
    df_tgt = read_csv_any(target_csv)
    src_attrs = _non_index_columns(df_src)
    tgt_attrs = _non_index_columns(df_tgt)
    G = golden_matrix_s1xs1(golden_json, src_attrs, tgt_attrs) if s1xs1 else golden_matrix_s1xs2(golden_json, src_attrs, tgt_attrs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    G.to_csv(out_csv, index=True)
    click.echo(f"[OK] Golden matrix saved to: {out_csv}")


if __name__ == "__main__":
    main()
