"""
Mask bit definitions for dla dla fit warnings
"""


class DLAFLAG(object):
    ZBOUNDARY_COARSE = 2**0  # inital DLA solve relaxed to spectra edge
    ZBOUNDARY_REFINE = 2**1  # refined DLA solve relaxed to z window boundary
    NHIBOUNDARY_REFINE = 2**2  # refined DLA solve relaxed to nhi window boundary
    POTENTIAL_BAL = (
        2**3
    )  # DLA solution overlaps with Lya or NV BAL, potential false positive
    BAD_ZFIT = 2**4  # bad parabola fit to chi2(refined z) surface
    BAD_NHIFIT = 2**5  # bad parabola fit to chi2(refined nhi) surface
