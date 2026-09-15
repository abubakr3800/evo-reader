"""
evostudy — read a DIALux evo (.evo) project and redraw the lighting study.

    from evostudy import load_study
    study = load_study("Project10.evo")
    print(study.grids[0].metrics())
"""

__version__ = "1.0.0"

from .archive import EvoArchive
from .model import CalcGrid, Luminaire, Polygon, Room, Study
from .pipeline import load_study

__all__ = [
    "EvoArchive", "Study", "Room", "Luminaire", "Polygon", "CalcGrid",
    "load_study", "__version__",
]
