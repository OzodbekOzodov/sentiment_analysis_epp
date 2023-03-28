from __future__ import annotations

import sentiment_analysis_epp


def test_import():
    assert hasattr(sentiment_analysis_epp, "__version__")
