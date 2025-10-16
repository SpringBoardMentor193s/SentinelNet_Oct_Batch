from __future__ import annotations
import sys
import re

#!/usr/bin/env python3
"""
ex.py - Palindrome utilities and simple CLI.

Usage:
    python ex.py               # interactive prompt
    python ex.py text1 text2  # check each argument
"""

ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(s: str) -> str:
        """Return lowercase alphanumeric-only version of s."""
        return "".join(ALNUM_RE.findall(s)).lower()


def is_palindrome(s: str) -> bool:
        """Check if s is a palindrome (ignores case and non-alphanumerics)."""
        n = normalize(s)
        return n == n[::-1]


def is_palindrome_number(value) -> bool:
        """Check numeric palindrome for ints (negatives are not palindromes)."""
        try:
                n = int(value)
        except (ValueError, TypeError):
                raise ValueError("Value is not an integer")
        if n < 0:
                return False
        s = str(n)
        return s == s[::-1]


def _check_and_print(token: str) -> None:
        """Print results for token (string + optional numeric check)."""
        p = is_palindrome(token)
        print(f"'{token}': palindrome (text) -> {p}")
        try:
                pn = is_palindrome_number(token)
                print(f"         palindrome (int)  -> {pn}")
        except ValueError:
                pass


def main(argv: list[str] | None = None) -> int:
        if argv is None:
                argv = sys.argv[1:]
        if not argv:
                try:
                        s = input("Enter text to check palindrome (empty to exit): ").strip()
                except EOFError:
                        return 0
                if not s:
                        return 0
                _check_and_print(s)
                return 0

        for tok in argv:
                _check_and_print(tok)
        return 0


if __name__ == "__main__":
        raise SystemExit(main())