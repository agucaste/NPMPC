import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    DirectCollocation,
    DirectTranscription,
    LinearSystem,
    MathematicalProgram,
    Solve,
    eq,
)