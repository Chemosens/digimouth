"""Configuration pytest de digimouth_lib/digimouth/tests.

Rend le paquet importable même s'il n'est pas installé (pip install -e .) : on ajoute le
dossier parent (digimouth_lib/) au sys.path.
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
