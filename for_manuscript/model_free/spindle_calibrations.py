# spindle_calibrations.py
# Field-Dependent Calibration Constants (Derived from Synthetic "Final Exam")
CALIBRATIONS = {
        500.0: {
            'TAUC': {'slope': 1.0008, 'intercept': -0.0369},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        600.0: {
            'TAUC': {'slope': 1.0028, 'intercept': -0.0482},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        700.0: {
            'TAUC': {'slope': 1.0019, 'intercept': -0.0109},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        750.0: {
            'TAUC': {'slope': 1.0027, 'intercept': -0.0221},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        800.0: {
            'TAUC': {'slope': 1.0003, 'intercept': 0.0119},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        850.0: {
            'TAUC': {'slope': 1.0032, 'intercept': -0.0155},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        900.0: {
            'TAUC': {'slope': 1.0024, 'intercept': -0.0098},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            },
        1100.0: {
            'TAUC': {'slope': 1.0034, 'intercept': -0.0066},
            'S2': {'slope': 1, 'intercept': 0},
            'TAUE': {'slope': 1, 'intercept': 0},
            'REX': {'a': 0, 'b': 1, 'c': 0}
            }
        }

MULTIPLIERS = {
        500.0: {
            'TAUC': {'mult': 1.022},
            'S2': {'mult': 3.972},
            'TAUE': {'mult': 6.788},
            'REX': {'mult': 3.360}
            },
        600.0: {
            'TAUC': {'mult': 0.779},
            'S2': {'mult': 3.182},
            'TAUE': {'mult': 5.777},
            'REX': {'mult': 2.726}
            },
        700.0: {
            'TAUC': {'mult': 0.951},
            'S2': {'mult': 3.735},
            'TAUE': {'mult': 7.118},
            'REX': {'mult': 3.423}
            },
        750.0: {
            'TAUC': {'mult': 0.853},
            'S2': {'mult': 3.553},
            'TAUE': {'mult': 6.502},
            'REX': {'mult': 3.151}
            },
        800.0: {
            'TAUC': {'mult': 0.831},
            'S2': {'mult': 3.315},
            'TAUE': {'mult': 6.499},
            'REX': {'mult': 3.145}
            },
        850.0: {
            'TAUC': {'mult': 1.094},
            'S2': {'mult': 3.361},
            'TAUE': {'mult': 5.421},
            'REX': {'mult': 2.933}
            },
        900.0: {
            'TAUC': {'mult': 0.994},
            'S2': {'mult': 3.244},
            'TAUE': {'mult': 6.494},
            'REX': {'mult': 2.886}
            },
        1100.0: {
            'TAUC': {'mult': 0.844},
            'S2': {'mult': 3.087},
            'TAUE': {'mult': 5.779},
            'REX': {'mult': 3.395}
            }
        }

