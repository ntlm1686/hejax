#
# polynomial class
#
from ._polybase import ABCPolyBase

import jax
import jax.numpy as jnp
from jax.numpy import (polydiv, polymul, polyadd, polysub, polyval, polyint, polyder, polyfit)

# Polynomial default domain.
polydomain = jnp.array([-1, 1])

# Polynomial coefficients representing zero.
polyzero = jnp.array([0])

# Polynomial coefficients representing one.
polyone = jnp.array([1])

# Polynomial coefficients representing the identity x.
polyx = jnp.array([0, 1])


class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed in the `ABCPolyBase` documentation.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1, 1].

        .. versionadded:: 1.6.0

    """
    # Virtual Functions
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    # _pow = staticmethod(polypow)
    _pow = staticmethod(lambda : None)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    # _line = staticmethod(polyline)
    # _roots = staticmethod(polyroots)
    # _fromroots = staticmethod(polyfromroots)

    _line = staticmethod(lambda : None)
    _roots = staticmethod(lambda : None)
    _fromroots = staticmethod(lambda : None)

    # Virtual properties
    domain = jnp.array(polydomain)
    window = jnp.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"