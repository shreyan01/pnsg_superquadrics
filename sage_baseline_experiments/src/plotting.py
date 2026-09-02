import sys
from pathlib import Path

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from evaluation import (
    LABEL_ORDER,
    calculate_confusion_matrix,
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    output_path=None,
):
    """
    Plot a raw-count confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.

    y_pred : array-like
        Predicted labels.

    model_name : str
        Model name displayed in the title.

    output_path : str or Path, optional
        Location where the figure will be saved.
    """

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_xticks(
        np.arange(
            len(LABEL_ORDER)
        )
    )

    ax.set_yticks(
        np.arange(
            len(LABEL_ORDER)
        )
    )

    ax.set_xticklabels(
        LABEL_ORDER,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        LABEL_ORDER
    )

    # Add count values
    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of instances",
    )

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    return fig


# =========================================================
# NORMALIZED CONFUSION MATRIX
# =========================================================

def plot_normalized_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    output_path=None,
):
    """
    Plot a row-normalized confusion matrix.

    Each row represents one true class and sums
    approximately to 100%.
    """

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    matrix = matrix.astype(
        float
    )

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix
        ),
        where=row_sums != 0,
    )

    normalized *= 100.0

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        normalized
    )

    ax.set_title(
        f"{model_name} Normalized Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_xticks(
        np.arange(
            len(LABEL_ORDER)
        )
    )

    ax.set_yticks(
        np.arange(
            len(LABEL_ORDER)
        )
    )

    ax.set_xticklabels(
        LABEL_ORDER,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        LABEL_ORDER
    )

    # Add percentage values
    for i in range(
        normalized.shape[0]
    ):

        for j in range(
            normalized.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{normalized[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Percentage of true class",
    )

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    return fig