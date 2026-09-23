"""Check participant-package files without running or inspecting the environment."""

import argparse
import csv
from pathlib import Path


REQUIRED_FILES = (
    "local_eval.py",
    "make_submission.py",
    "agent_template.py",
    "environment.py",
    "mock_environment.py",
    "scoring_core.py",
    "customer_profile.csv",
    "data/change_tariff.csv",
    "data/traffic.csv",
    "data/arpu_monthly.csv",
    "data/dict_tariff.csv",
    "tariff_dictionary.csv",
    "feature_dictionary.csv",
)
PROFILE_COLUMNS = {
    "predicted_arpu", "arpu_segment", "data_segment", "call_segment"
}


def check_package(root: Path) -> int:
    """Return 0 for an available package, 1 for missing files or profile columns."""
    failures = []
    print("Папка пакета:", root)
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            failures.append(name)
            print("НЕТ:", name)
        elif path.stat().st_size == 0:
            failures.append(name)
            print("ПУСТОЙ ФАЙЛ:", name)
        else:
            print("ЕСТЬ:", name)

    profile = root / "customer_profile.csv"
    if profile.is_file() and profile.stat().st_size:
        try:
            with profile.open(encoding="utf-8-sig", newline="") as source:
                sample = source.read(8192)
                source.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                headers = set(next(csv.reader(source, dialect), []))
            missing_columns = PROFILE_COLUMNS - headers
            if missing_columns:
                failures.append("profile columns")
                print("НЕТ КОЛОНОК ПРОФИЛЯ:", ", ".join(sorted(missing_columns)))
        except (OSError, UnicodeError, csv.Error) as exc:
            failures.append("profile format")
            print("Не удалось прочитать заголовок customer_profile.csv:", exc)

    if failures:
        print("\nПакет неполный или отличается от ТЗ. Нужны исходные файлы организаторов.")
        print("Эта проверка не заменяет python local_eval.py.")
        return 1
    print("\nФайлы найдены; обязательные колонки профиля доступны.")
    print("Это проверка комплекта, а не подтверждение работоспособности агента.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Папка, в которую распакован официальный пакет участника",
    )
    return check_package(parser.parse_args().root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
