from .file import File, _DimensionScale
from .variable import Variable

try:
	import pyfive
except Exception:  # pragma: no cover - optional dependency
	pyfive = None

if pyfive is not None:
	# Let external callers (e.g. cfdm/cf-python dispatch) treat ppfive
	# files as pyfive-like file handles.
	pyfive.File.register(File)
	pyfive.Dataset.register(Variable)
	pyfive.Dataset.register(_DimensionScale)

__all__ = ["File", "Variable"]
