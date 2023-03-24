# Legacy code

Parts of scikit-rt have been migrated from the package
[quickviewer](https://github.com/hlpullen/quickviewer).  In particular,
the [BetterViewer](https://scikit-rt.github.io/scikit-rt/skrt.better_viewer.html#skrt.better_viewer.BetterViewer) class of scikit-rt
reimplements, and improves on, most of the image-viewing functionality of the
[QuickViewer](https://scikit-rt.github.io/scikit-rt/skrt.viewer.viewer.html#skrt.viewer.viewer.QuickViewer) class of quickviewer, but doesn't include
the following:

- [viewing of images forming a time series](https://github.com/hlpullen/quickviewer#7-time-series);
- [viewing an image alongside an orthogonal view](https://github.com/hlpullen/quickviewer#orthogonal-view).

To make available from scikit-rt the QuickViewer functionality not implemented
in BetterViewer, code for the former is included in the package
`skrt.viewer`.  The QuickViewer class can be imported as:

```
from skrt.viewer.viewer import QuickViewer
```

The class may then be used as described in the quickviewer documentation:
[How to use QuickViewer](https://github.com/hlpullen/quickviewer#ii-how-to-use-quickviewer).
