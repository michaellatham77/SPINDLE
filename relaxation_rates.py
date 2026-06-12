"""
Created on Tue Nov 12 14:21:03 2024

@author: olivi

From Olivia's independent project
Library to calculate relaxation rates

@author: MP Latham
@date: March 13, 2025 (Updated Jan 2026 for dynamic field arguments)
"""

#Libraries
import numpy as np

# Constants 
gyroP = 267.522 * 10**6 # rad s-1 T-1
gyroN = -27.126 * 10**6 # rad s-1 T-1

#Functions
def orig_sp_den(freq, tauC, S2s, S2f, tauS):
    '''
    "Original" Spectral density function
    Slow isotropic tumbling with faster internal motions
    Jarymowycz, VA and Stone, MJ (2006) Chem Rev 106: 1624-1671
    eqn 7
    '''

    tauS_arr = np.broadcast_to(tauS, S2s.shape)
    tauC_arr = np.broadcast_to(tauC, S2s.shape)
    denominator = tauC_arr + tauS_arr
    tauPr = np.divide(tauC_arr * tauS_arr, denominator, out=np.zeros_like(denominator), where=denominator!=0)

    term1_num = S2s * tauC_arr
    term1_den = 1 + (freq * tauC_arr)**2
    term1 = np.divide(term1_num, term1_den, out=np.zeros_like(term1_den), where=term1_den!=0)

    term2_num = (1 - S2s) * tauPr
    term2_den = 1 + (freq * tauPr)**2
    term2 = np.divide(term2_num, term2_den, out=np.zeros_like(term2_den), where=term2_den!=0)

    spden = term1 + term2
    return spden*(2/5)

def ext_sp_den(freq, tauC, S2s, S2f, tauS):
    '''
    Spectral density function from Clore, GM et al. (1990) JACS 112(12):
    4989-4991 eqn. 4
    Note: there is a mistake in eqn. 9 of Jarymowycz and Stone Review
    '''

    S2 = (S2s)*(S2f)
    tauPr = (tauS*tauC)/(tauC+tauS)

    spden = (((S2)*tauC)/(1+(freq*tauC)**2))+((((S2f)-(S2))*tauPr)/(1+(freq*tauPr)**2))

    return spden*(2/5)

def longitudinal_relaxation_rate_total(J_func, tauC, S2s, S2f, tauS, OmeH, OmeN, d_const_sq, c_const_sq):
    '''
    R1 relaxation rate
    Jarymowycz, VA and Stone, MJ (2006) Chem Rev 106: 1624-1671
    eqn. 14a and 14b
    '''

    # DD contribution 
    J_DD_terms = (J_func(OmeH - OmeN, tauC, S2s, S2f, tauS) +
                  3 * J_func(OmeN, tauC, S2s, S2f, tauS) +
                  6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauS))
    R1_DD = (d_const_sq / 4.0) * J_DD_terms

    # CSA contribution
    R1_CSA = (c_const_sq) * J_func(OmeN, tauC, S2s, S2f, tauS)

    return R1_DD + R1_CSA

def transverse_relaxation_rate_total(J_func, tauC, S2s, S2f, tauS, Rex, OmeH, OmeN, d_const_sq, c_const_sq):
    '''
    R2 relaxation rate with Rex
    Jarymowycz, VA and Stone, MJ (2006) Chem Rev 106: 1624-1671
    eqn. 15a and 15b
    '''

    # DD contribution
    J_DD_terms = (4 * J_func(0, tauC, S2s, S2f, tauS) +
                  J_func(OmeH - OmeN, tauC, S2s, S2f, tauS) +
                  3 * J_func(OmeN, tauC, S2s, S2f, tauS) +
                  6 * J_func(OmeH, tauC, S2s, S2f, tauS) +
                  6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauS))
    R2_DD = (d_const_sq / 8.0) * J_DD_terms

    # CSA contribution 
    J_CSA_terms = (4 * J_func(0, tauC, S2s, S2f, tauS) +
                   3 * J_func(OmeN, tauC, S2s, S2f, tauS))
    R2_CSA = (c_const_sq / 6.0) * J_CSA_terms

    return R2_DD + R2_CSA + Rex

def nuclear_overhauser_effect(J_func, tauC, S2s, S2f, tauS, OmeH, OmeN, d_const_sq, c_const_sq):
    '''
    Heteronuclear NOE
    Jarymowycz, VA and Stone, MJ (2006) Chem Rev 106: 1624-1671
    eqn. 16a
    '''

    r1 = longitudinal_relaxation_rate_total(J_func, tauC, S2s, S2f, tauS, OmeH, OmeN, d_const_sq, c_const_sq)

    # Cross-relaxation 
    sigma_NH = ((d_const_sq / 4.0) *
                (6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauS) -
                 J_func(OmeH - OmeN, tauC, S2s, S2f, tauS)))

    noe = 1 + (gyroP / gyroN) * (1/r1) * sigma_NH
    return noe
