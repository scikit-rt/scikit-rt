# Keyboard shortcuts for pop-out interactive viewer

A minimal pop-up interactive viewer can be created by running `.view()` on
any `Image`, `Dose`, `ROI` or `StructureSet` from python outside of a
Jupyter notebook. The controls are:

- scroll: move 1 slice
- left/right keys: move 1 slice
- up/down keys: move 5 slices
- v: toggle orientation between axial, sagittal, and coronal
- d: toggle dose field opacity (if viewing dose field overlaid on image)
- c: toggle ROI plotting type between contour, filled, and mask (if viewing ROIs)
- j: jump between ROIs (if viewing a `StructureSet`)
