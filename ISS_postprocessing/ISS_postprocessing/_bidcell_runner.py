"""Small subprocess entry point used by the BIDCell wrapper."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="BIDCell YAML/JSON configuration file")
    args = parser.parse_args()

    try:
        from bidcell import BIDCellModel
    except ImportError as exc:
        raise SystemExit(
            "BIDCell is not installed in this Python environment. "
            "Install it with `python -m pip install bidcell`."
        ) from exc

    BIDCellModel(args.config).run_pipeline()


if __name__ == "__main__":
    main()
