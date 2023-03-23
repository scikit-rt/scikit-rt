# Legacy code

Parts of scikit-rt, and in particular the image-viewing functionality,
have been migrated from the package [quickviewer](https://github.com/hlpullen/quickviewer).  The [BetterViewer](https://scikit-rt.github.io/scikit-rt/skrt.better_viewer.html#skrt.better_viewer.BetterViewer) class of scikit-rt
reimplements, and improves on, most of the image-viewing functionality of the
[QuickViewer](https://scikit-rt.github.io/scikit-rt/skrt.viewer.viewer.html#skrt.viewer.viewer.QuickViewer) class of quickviewer.  Known exceptions are:

- QuickViewer provides for [viewing of images forming a time series](https://github.com/hlpullen/quickviewer#7-time-series);
- QuickViewer provides for [viewing an image alongside an orthogonal view](https://github.com/hlpullen/quickviewer#orthogonal-view)

To make available from scikit-rt the QuickViewer functionality not implemented
in BetterViewer, code for the former is included in the package
`skrt.viewer`.  The QuickViewer class can be imported as:

```
from skrt.viewer.viewer import QuickViewer
```

then may be used as described in the quickviewer documentation:
[How to use QuickViewer](https://github.com/hlpullen/quickviewer#ii-how-to-use-quickviewer).
