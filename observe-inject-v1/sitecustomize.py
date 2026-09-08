
# Preserve the existing startup customization before installing observation.
import sys
from importlib.machinery import PathFinder
from pathlib import Path
_own_dir = Path(__file__).resolve().parent
_paths = [p for p in sys.path if Path(p).resolve() != _own_dir]
_spec = PathFinder.find_spec("sitecustomize", _paths)
if _spec is not None and _spec.loader is not None:
    _code = _spec.loader.get_code("sitecustomize")
    if _code is not None:
        exec(_code, dict(__name__="sitecustomize", __file__=_spec.origin))

# Generated optional observer bootstrap. Failure must not prevent Python startup.
def _install_optional_observer():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent
    if not (root / "enabled").is_file():
        return
    from timing_probe import install
    install(json.loads((root / "config.json").read_text(encoding="utf-8")))

try:
    _install_optional_observer()
except Exception:
    pass
