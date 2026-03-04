#!/usr/bin/env python
"""
Script para rodar todos os testes do projeto.
Execute com: python run_tests.py
"""
import subprocess
import sys


def main():
    """Execute os testes utilizando unittest."""
    print("=" * 70)
    print("Executando testes do AeroRL-Optimizer")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "test",
        "-p",
        "test_*.py",
        "-v",
    ]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
