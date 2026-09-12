# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Figure style for the demonstration and validation cases.

The solver does not plot. This module exists so the scripts under ``examples/``
and ``validation/`` draw consistently, and it is the only place in the package
that imports matplotlib -- lazily, inside :func:`set_plot_style`, so a headless
install never pays for it.

The palette is Okabe-Ito, which stays distinguishable under the common forms of
colour vision deficiency and survives greyscale printing.

Units follow the Copernicus convention, since the figures are submitted as
drawn: parentheses rather than square brackets, negative exponents rather than
slashes (``m s$^{-1}$``), upright unit symbols against italic quantity symbols,
and no unit marker at all on a dimensionless quantity.
"""

from __future__ import annotations

#: Okabe-Ito qualitative palette.
PALETTE = {
    "Black": "#000000",
    "Orange": "#E69F00",
    "Sky Blue": "#3A9DC5",
    "Bluish Green": "#009E73",
    "Yellow": "#F0E442",
    "Blue": "#2C497F",
    "Vermillion": "#D55E00",
    "Reddish Purple": "#CC79A7",
}

#: The order the palette cycles in, so a figure's nth series is reproducible.
COLOR_CYCLE = [
    PALETTE["Black"],
    PALETTE["Orange"],
    PALETTE["Sky Blue"],
    PALETTE["Bluish Green"],
    PALETTE["Yellow"],
    PALETTE["Blue"],
    PALETTE["Vermillion"],
    PALETTE["Reddish Purple"],
]

#: Colours for the three membrane regimes, in ``SLACK, WRINKLED, TAUT`` order.
REGIME_COLORS = (PALETTE["Vermillion"], PALETTE["Orange"], PALETTE["Sky Blue"])


def set_plot_style(use_latex: bool = False) -> None:
    """Apply the package figure style to the global matplotlib state.

    ``use_latex`` switches on the LaTeX text renderer, which needs a working
    TeX installation; the default keeps mathtext, which renders the same
    symbols without one.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=COLOR_CYCLE),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "text.usetex": bool(use_latex),
            "font.family": "serif" if use_latex else "sans-serif",
        }
    )


__all__ = ["PALETTE", "COLOR_CYCLE", "REGIME_COLORS", "set_plot_style"]
