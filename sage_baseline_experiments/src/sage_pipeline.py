"""
Bridge to the real SAGE training and evaluation code.

Task 3 needs `ycbv_training/train_registry_multiview.py` and
`ycbv_training/evaluate_on_ycbv.py` from the parent repository. Importing
them is not quite a plain import, for two reasons:

1. `train_registry_multiview.py` uses package-relative imports
   (`from ..registry import Registry`), so it needs a *parent package* to
   sit inside. The repository has no `__init__.py` anywhere, so importing
   it as a top-level module fails with "attempted relative import beyond
   top-level package".

2. The modules it pulls in then use flat imports (`from superquadric
   import ...`), so the repository root also has to be on `sys.path`.

`load_sage_modules()` sets both up. It registers a synthetic parent
package pointing at the repository root, which means this works no matter
what the checked-out folder is called - cloning the repo under any
directory name is fine.

Nothing in the parent repository is modified. The pieces reused are the
author's own:

    train_registry_multiview.discover_video_classes
    train_registry_multiview._aggregate_and_fit
    train_registry_multiview.AXISYMMETRIC_WORDS
    evaluate_on_ycbv._init_worker / _eval_one_frame
    registry.Registry
"""

import importlib
import sys
import types
from pathlib import Path

# sage_baseline_experiments/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The repository root, holding registry.py, superquadric.py, ycbv_training/
# and this project as a subfolder.
REPO_ROOT = PROJECT_ROOT.parent

# Name for the synthetic parent package. Deliberately not the directory
# name, so a clone under any folder name behaves identically.
_PACKAGE = "_sage_repo"


class SagePipelineUnavailable(RuntimeError):
    """
    Raised when the SAGE training/eval code cannot be imported.

    Carries an actionable message rather than a bare ImportError, since
    the usual cause is a missing optional dependency or a checkout that
    does not include the parent repository.
    """


def _register_parent_package():
    """
    Make the repository root importable as a package.

    This is what lets `from ..registry import Registry` inside
    `ycbv_training/` resolve, without adding `__init__.py` files to the
    parent repository or depending on its folder name.
    """

    if _PACKAGE not in sys.modules:

        package = types.ModuleType(_PACKAGE)

        # A package with a __path__ but no __init__.py behaves like a
        # namespace package, which is all the relative imports need.
        package.__path__ = [str(REPO_ROOT)]

        sys.modules[_PACKAGE] = package

    # Flat imports inside the parent repo's own modules
    # (`from superquadric import ...`) need the root on sys.path too.
    root = str(REPO_ROOT)

    if root not in sys.path:
        sys.path.insert(0, root)


def check_available():
    """
    Report whether the SAGE pipeline can be driven from here.

    Returns
    -------
    (bool, str)
        Whether it is importable, and a human-readable reason when not.
    """

    training_script = (
        REPO_ROOT
        / "ycbv_training"
        / "train_registry_multiview.py"
    )

    if not training_script.exists():
        return False, (
            f"Not found: {training_script}\n"
            f"Task 3's SAGE half needs the parent SAGE repository. "
            f"This project is expected to live inside it, as "
            f"<repo>/sage_baseline_experiments/."
        )

    try:
        importlib.import_module("cv2")
    except ImportError:
        return False, (
            "The `cv2` module is missing. The YCB-Video loader reads "
            "depth/label/colour PNGs through OpenCV.\n"
            "Install it with:  pip install opencv-python"
        )

    try:
        load_sage_modules()
    except Exception as error:
        return False, f"Import failed: {type(error).__name__}: {error}"

    return True, "SAGE pipeline is importable."


def load_sage_modules():
    """
    Import the parent repository's training and evaluation modules.

    Returns
    -------
    (training_module, evaluation_module, Registry)

    Raises
    ------
    SagePipelineUnavailable
        With a message naming the likely fix.
    """

    _register_parent_package()

    try:
        training = importlib.import_module(
            f"{_PACKAGE}.ycbv_training.train_registry_multiview"
        )

        evaluation = importlib.import_module(
            f"{_PACKAGE}.ycbv_training.evaluate_on_ycbv"
        )

        registry = importlib.import_module(
            f"{_PACKAGE}.registry"
        )

    except ImportError as error:

        raise SagePipelineUnavailable(
            f"Could not import the SAGE pipeline from {REPO_ROOT}.\n"
            f"Underlying error: {error}\n\n"
            f"Most likely causes:\n"
            f"  - `pip install opencv-python` (the dataset loader needs cv2)\n"
            f"  - this project is not inside the SAGE repository\n"
        ) from error

    return training, evaluation, registry.Registry


def resolve_dataset_root(dataset_root):
    """
    Validate a YCB-Video dataset root.

    Checks for the `image_sets/` split files the training and evaluation
    scripts read, so a wrong path fails immediately with a clear message
    instead of hours later.
    """

    root = Path(dataset_root).expanduser().resolve()

    if not root.is_dir():
        raise SagePipelineUnavailable(
            f"Dataset root does not exist: {root}"
        )

    image_sets = root / "image_sets"

    if not image_sets.is_dir():
        raise SagePipelineUnavailable(
            f"No image_sets/ directory under {root}.\n"
            f"Expected the standard YCB-Video layout:\n"
            f"    {root}/image_sets/train.txt\n"
            f"    {root}/image_sets/val.txt\n"
            f"    {root}/data/0000/...  (or data2/, data3/)"
        )

    for split_name in ("train", "val"):

        split_file = image_sets / f"{split_name}.txt"

        if not split_file.exists():
            raise SagePipelineUnavailable(
                f"Missing split file: {split_file}"
            )

        # A byte-size check is not enough: the stub files shipped in this
        # repo contain a single blank line, so they are 2 bytes rather
        # than 0. Count actual frame keys instead.
        n_frames = sum(
            1
            for line in split_file.read_text().splitlines()
            if line.strip()
        )

        if n_frames == 0:
            raise SagePipelineUnavailable(
                f"Split file has no frame keys: {split_file}\n"
                f"The SAGE repo ships empty stubs for these; point "
                f"--dataset_root at a real YCB-Video copy."
            )

    return str(root)
