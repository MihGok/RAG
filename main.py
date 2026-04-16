"""
main.py
────────
Точка входа. Запускает интерактивный пайплайн поиска и скачивания курсов.

Использование:
    python main.py
    python main.py --courses 10   # скачать топ-10 курсов (по умолчанию 5)
"""

import sys
import argparse
import loading_workflow as workflow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск и скачивание курсов со Stepik"
    )
    parser.add_argument(
        "--courses",
        type=int,
        default=5,
        metavar="N",
        help="Количество топ-курсов для скачивания (по умолчанию: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    session_dir = workflow.run_pipeline(max_courses=args.courses)

    if session_dir:
        print(f"\nСессия завершена: {session_dir}")
    else:
        print("\nПайплайн завершился с ошибкой.")
        sys.exit(1)


if __name__ == "__main__":
    main()