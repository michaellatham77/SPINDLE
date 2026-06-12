# spindle_calibrations.py
# Field-Dependent Calibration Constants (Derived from Synthetic "Final Exam")
CALIBRATIONS = {
        500.0: {
            'TAUC': {'slope': 1.0068, 'intercept': 0.0234},
            'S2': {'slope': 1.0062, 'intercept': -0.0077},
            'TAUE': {'slope': 1.0115, 'intercept': 1.0214},
            'REX': {'a': -0.002, 'b': 1.0186, 'c': -0.0411}
            },
        600.0: {
            'TAUC': {'slope': 1.0078, 'intercept': 0.0391},
            'S2': {'slope': 1.0066, 'intercept': -0.0087},
            'TAUE': {'slope': 1.0117, 'intercept': 2.0487},
            'REX': {'a': 0.0004, 'b': 1.0079, 'c': -0.0270}
            },
        700.0: {
            'TAUC': {'slope': 1.0068, 'intercept': 0.0234},
            'S2': {'slope': 1.0040, 'intercept': -0.0060},
            'TAUE': {'slope': 1.0118, 'intercept': 0.8772},
            'REX': {'a': 0.0007, 'b': 0.9983, 'c': -0.0130}
            },
        800.0: {
            'TAUC': {'slope': 1.0049, 'intercept': 0.0391},
            'S2': {'slope': 1.0039, 'intercept': -0.0073},
            'TAUE': {'slope': 1.0150, 'intercept': 0.4269},
            'REX': {'a': 0.0017, 'b': 0.9782, 'c': -0.0217}
            },
        850.0: {
            'TAUC': {'slope': 1.0146, 'intercept': -0.2109},
            'S2': {'slope': 1.0023, 'intercept': -0.0054},
            'TAUE': {'slope': 1.0087, 'intercept': -0.1837},
            'REX': {'a': 0.0023, 'b': 0.9713, 'c': -0.0220}
            },
        900.0: {
            'TAUC': {'slope': 1.0010, 'intercept': 0.0938},
            'S2': {'slope': 1.0047, 'intercept': -0.0071},
            'TAUE': {'slope': 1.0134, 'intercept': 0.8964},
            'REX': {'a': 0.0032, 'b': 0.9457, 'c': -0.0321}
            },
        1100.0: {
            'TAUC': {'slope': 1.0049, 'intercept': 0.0234},
            'S2': {'slope': 1.0053, 'intercept': -0.0061},
            'TAUE': {'slope': 1.0190, 'intercept': -1.2488},
            'REX': {'a': 0.0035, 'b': 0.9383, 'c': 0.0126}
            }
        }

MULTIPLIERS = {
        500.0: {
            'TAUC': {'mult': 2.193},
            'S2': {'mult': 4.158},
            'TAUE': {'mult': 2.952},
            'REX': {'mult': 0.708}
            },
        600.0: {
            'TAUC': {'mult': 1.820},
            'S2': {'mult': 3.708},
            'TAUE': {'mult': 2.731},
            'REX': {'mult': 0.727}
            },
        700.0: {
            'TAUC': {'mult': 2.140},
            'S2': {'mult': 3.902},
            'TAUE': {'mult': 2.674},
            'REX': {'mult': 0.799}
            },
        800.0: {
            'TAUC': {'mult': 1.964},
            'S2': {'mult': 3.497},
            'TAUE': {'mult': 2.467},
            'REX': {'mult': 0.606}
            },
        850.0: {
            'TAUC': {'mult': 2.333},
            'S2': {'mult': 4.152},
            'TAUE': {'mult': 3.138},
            'REX': {'mult': 0.632}
            },
        900.0: {
            'TAUC': {'mult': 1.833},
            'S2': {'mult': 3.542},
            'TAUE': {'mult': 2.716},
            'REX': {'mult': 0.594}
            },
        1100.0: {
            'TAUC': {'mult': 2.116},
            'S2': {'mult': 3.489},
            'TAUE': {'mult': 2.947},
            'REX': {'mult': 0.917}
            }
        }

