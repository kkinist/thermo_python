#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a Gaussian output file, or a custom data file, and create a 
JANAF-style table of ideal-gas thermochemical data.
Basic data are saved to data file 'thermo.dat', which may be
modified for advanced use. 
This program is a Python replacement for the Perl program, 'thermo.pl',
which is described at 
https://www.nist.gov/mml/csd/chemical-informatics-research-group/products-and-services/program-computing-ideal-gas 
Karl Irikura, NIST, karl.irikura@nist.gov
    Python program started 7/9/2020
    (7/15/20) Anything in the input after a '#' is ignored
              'nosave' on command line skips writing 'thermo.dat' (no prompt)
              'dt[Tstep]' on command line sets temperature step (in thermo.pl but undocumented)
    (11/10/20) XVIB results differ from old thermo.pl but appear correct
"""
import sys, re, os
import numpy as np

# CODATA 2018 constants from physics.nist.gov, retrieved 7/13/2020
AVOGAD = 6.02214076e23     # mol^-1 (exact, defined value)
BOLTZ = 1.380649e-23       # J/K (exact, defined value)
RGAS = AVOGAD * BOLTZ      # J/mol/K (exact)
PLANCK = 6.62607015e-34    # J s (exact, defined value)
CLIGHT = 299792458.        # m/s (exact, defined value)
PRESS = 1.0e5              # Pa (standard pressure is 1 bar; change to 101325. for 1 atm)
CM2KJ = PLANCK * AVOGAD * CLIGHT / 10  # convert from cm^-1 to kJ/mol
CM2K = 100 * CLIGHT * PLANCK / BOLTZ   # convert from cm^-1 to Kelvin
AMU = 1.66053906660e-27     # kg/u

# list of irregular temperatures to include in output table
specialt = [298.15]

############## function definitions
def thermo_lnQ(T, dlnQ):
    # given temperature and list [lnQ, dlnQ/dT, d2lnQ/dT2],
    # return S, Cp, and [H(T)-H(0)]
    [lnQ, d, d2] = dlnQ  # lnQ and its derivatives
    deriv = T * d + lnQ  # derivative of TlnQ
    S = RGAS * (deriv - np.log(AVOGAD) + 1)
    deriv2 = 2 * d + T * d2  # 2nd derivative of TlnQ
    Cp = RGAS + RGAS * T * deriv2
    ddH = RGAS * T * (1 + T * d) / 1000
    return (S, Cp, ddH)
##
def lnQvrt(T, freqs, symno, ABC_GHz, mass, pressure, deriv=0):
    # Return the total ln(Q) (vib + rot + transl partition function) 
    #   or a derivative. RRHO approximation
    lnQv = lnQvib(T, freqs, deriv)
    lnQr = lnQrot(T, symno, ABC_GHz, deriv)
    lnQt = lnQtrans(T, mass, pressure, deriv)
    lnQ = lnQv + lnQr + lnQt
    return lnQ
##
def lnQtrans(T, mass, pressure, deriv=0):
    # Given a temperature (in K), a molecular mass (in amu),
    #   and a pressure (in Pa), return ln(Q), where
    #   Q is the ideal-gas translational partition function.
    # If deriv > 0, return a (1st or 2nd) derivative of lnQ
    #   instead of lnQ. 
    if deriv not in [0,1,2]:
        print('*** unimplemented deriv =', deriv, 'in lnQtrans()')
        return None
    if deriv == 1:
        # return (d/dT)lnQ = (3/2T)
        return (1.5 / T)
    if deriv == 2:
        # return (d2/dT2)lnQ = -(3/2T**2)
        return (-1.5 / (T*T))
    kT = BOLTZ * T  # in J
    m = mass * AMU   # in kg
    V = RGAS * T / pressure  # in m**3
    lnQ = 1.5 * np.log(2 * np.pi * m * kT)
    lnQ -= 3 * np.log(PLANCK)
    lnQ += np.log(V)
    return lnQ
##
def lnQrot(T, symno, ABC_GHz, deriv=0, numeric=None):
    # Given a temperature (in K), symmetry number, and list of
    #   rotational constants (in GHz), return ln(Q), where Q is
    #   the rigid-rotor partition function.
    # 'numeric' is for testing
    n = len(ABC_GHz)
    if n == 0:
        # atom; no rotations possible
        return 0.
    if numeric is None:
        # analytical derivatives
        if deriv not in [0,1,2]:
            print('*** Unimplemented deriv =', deriv,'in lnQrot()')
            return None
        if deriv == 1:
            # first derivative of lnQ depends only on temperature
            if n < 3:
                # linear case
                return (1/T)
            else:
                # non-linear
                return (1.5/T)
        if deriv == 2:
            # second derivative of lnQ 
            if n < 3:
                # linear case
                return (-1 / (T*T))
            else:
                # non-linear
                return (-1.5 / (T*T))
    ln_kTh = np.log(T) + np.log(BOLTZ) - np.log(PLANCK)  # ln(kT/h) expressed in ln(Hz)
    if n < 3:
        # linear molecule
        B = ABC_GHz[0] * 1.0e9  # convert to Hz
        lnQ = ln_kTh - np.log(symno * B)
    else:
        # polyatomic molecule with 3 constants
        lnQ = 1.5 * ln_kTh + 0.5 * np.log(np.pi) - np.log(symno)
        for c in ABC_GHz:
            B = c * 1.0e9 # convert to Hz
            lnQ -= 0.5 * np.log(B)
    if deriv == 0:
        return lnQ
    # reach here only if numerical differentiation is requested
    dT = numeric
    qplus = lnQrot(T + dT, symno, ABC_GHz, deriv - 1, numeric)
    qminus = lnQrot(T - dT, symno, ABC_GHz, deriv - 1, numeric)
    d = (qplus - qminus) / (2 * dT)
    return d
##
def lnQvib(T, freqs, deriv=0, numeric=None):
    # Given a temperature (in K) and array of vibrational 
    #   frequencies (in cm^-1), return ln(Q) where Q is
    #   the harmonic-oscillator partition function.
    # 'numeric' is for testing
    # the zero-point energy is the zero of energy
    nu = np.array(freqs) * CM2K  # convert to Kelvin
    fred = nu / T # reduced frequencies
    x = np.exp(-fred)  # exponentiated, reduced frequencies
    xm1 = 1 - x
    if deriv == 0:
        # return lnQ itself
        lnq = np.log(xm1)
        lnQ = -1 * lnq.sum()
        return lnQ
    # first or second derivative of lnQ
    if numeric is not None:
        # central differencing
        dT = numeric
        qplus = lnQvib(T + dT, freqs, deriv - 1, numeric)
        qminus = lnQvib(T - dT, freqs, deriv - 1, numeric)
        d = (qplus - qminus) / (2 * dT)
        return d
    if deriv not in [1,2]:
        print('Unimplemented deriv =', deriv,'in lnQvib()')
        return None
    y = x / xm1
    nuy = nu * y
    sum1 = nuy.sum()
    d = sum1 / (T*T)
    if deriv == 1:
        # 1st derivative of lnQ
        return d
    sum2 = np.dot(nu, nuy)
    sum3 = np.dot(nuy, nuy)
    if deriv == 2:
        # 2nd derivative of lnQ
        d2 = (sum2 + sum3) / (T**4)
        d2 -= 2 * d / T
        return d2
    # unknown derivative
    return None
##
def lnQelec(T, degen, energy, deriv=0, numeric=None):
    # Given (electronic) energy levels and degeneracies, 
    # return ln(Q) or its first or second derivative
    # Energies expected in cm^-1
    # 'numeric' argument is only for testing: requests numeric derivatives
    #    in that case, it is the temperature increment for differencing
    if len(degen) != len(energy):
        print('*** Error in lnQelec(): len(degen) = {:d} but len(energy) = {:d}'.format(len(degen), len(energy)))
        return None
    g = np.array(degen)
    E = np.array(energy) * CM2K  # convert to Kelvin
    E /= T  # reduced energies, unitless
    x = np.exp(-E)  # exponentiated, reduced energies
    y0 = g * x
    Q = y0.sum()
    if deriv == 0:
        return np.log(Q)
    # compute derivative
    if numeric is not None:
        # numerical differencing instead of analytic derivative
        dT = numeric
        qplus = lnQelec(T + dT, degen, energy, deriv - 1, dT)
        qminus = lnQelec(T - dT, degen, energy, deriv - 1, dT)
        d = (qplus - qminus) / (2 * dT)
        return d
    # analytical derivative
    if deriv not in [1,2]:
        print('*** Unknown option deriv =', deriv, 'in lnQelec()')
        return None
    y1 = E * y0
    Qp = y1.sum() / T
    d = Qp / Q
    if deriv == 1:
        # d/dT of lnQ
        return d
    y2 = E * y1
    Qpp = y2.sum() / (T*T)
    Qpp -= 2 * Qp / T
    d2 = Qpp / Q
    d2 -= d*d
    if deriv == 2:
        return d2
##
def lnQavib1(T, avib, deriv=0, numeric=None):
    # one simple, 1D anharmonic oscillator
    # convert to an energy ladder (list of levels starting at 0)
    # then apply lnQladder()
    # the zero-point energy is the zero of energy
    [w, x, ulim] = avib
    # get maximum v 
    # ulim is not enough for the while loop because it may not be unique
    vmax = int(avib_vlim(avib))
    elist = [w*v - x*v*(v+1) for v in range(vmax+1)]
    retval = lnQladder(T, elist, deriv, numeric)
    return retval
##
def lnQavib(T, avibs, deriv=0, numeric=None):
    # a list of 1D anharmonic oscillators
    if len(avibs) == 0:
        # nothing here
        return 0
    retval = 0
    for avib in avibs:
        retval += lnQavib1(T, avib, deriv, numeric)
    return retval
##
def lnQladder(T, energies, deriv=0, numeric=None):
    # Given a list of non-degenerate energy levels,
    # return ln(Q) or its first or second derivative
    # Energies expected in cm^-1
    # 'numeric' argument is only for testing: requests numeric derivatives
    #    in that case, 'numeric' is the temperature increment for differencing
    # Input energies should be relative to the ground state
    if len(energies) == 0:
        # nothing to do
        return 0
    E = np.array(energies) * CM2K  # convert to Kelvin
    E = E/T  # reduced energies, unitless
    x = np.exp(-E)  # exponentiated, reduced energies
    Q = x.sum()
    if deriv == 0:
        return np.log(Q)
    # compute derivative
    if numeric is not None:
        # numerical differencing instead of analytic derivative
        dT = numeric
        qplus = lnQladder(T + dT, energies, deriv - 1, dT)
        qminus = lnQladder(T - dT, energies, deriv - 1, dT)
        d = (qplus - qminus) / (2 * dT)
        return d
    # analytical derivative
    if deriv not in [1,2]:
        print('*** Unknown option deriv =', deriv, 'in lnQladder()')
        return None
    y1 = E * x
    Qp = y1.sum() / T
    d = Qp / Q
    if deriv == 1:
        # d/dT of lnQ
        return d
    y2 = E * y1
    Qpp = y2.sum() / (T*T)
    Qpp -= 2 * Qp / T
    d2 = Qpp / Q
    d2 -= d*d
    if deriv == 2:
        return d2
##
def test_derivs(T, freq, rot, symno, el_degen, el_energy, avibs, ladder, dT=0.1):
    # compare analytical and numerical derivatives, for debugging
    # Don't do this for translational partition function because of
    # problematic V = RT/p (it will fail)
    print('Comparing numerical and analytical derivatives of lnQ')

    if len(el_degen):
        print('\nelec')
        print('\tlnQ\t\td\t\td2')
        d = [lnQelec(T, el_degen, el_energy, der, dT) for der in [0,1,2]]
        print('numeric\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        d = [lnQelec(T, el_degen, el_energy, der) for der in [0,1,2]]
        print('analyt\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        # also compare with lnQladder()
        energies = []
        for g, e in zip(el_degen, el_energy):
            energies.extend([e] * g)
        d = [lnQladder(T, energies, der) for der in [0,1,2]]
        print('Qlad()\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))

    if len(rot):
        print('\nrot')
        print('\tlnQ\t\td\t\td2')
        d = [lnQrot(T, symno, rot, der, dT) for der in [0,1,2]]
        print('numeric\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        d = [lnQrot(T, symno, rot, der) for der in [0,1,2]]
        print('analyt\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))

    if len(freq):        
        print('\nvib')
        print('\tlnQ\t\td\t\td2')
        d = [lnQvib(T, freq, der, dT) for der in [0,1,2]]
        print('numeric\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        d = [lnQvib(T, freq, der) for der in [0,1,2]]
        print('analyt\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
    
    if len(avibs):
        print('\navib')
        print('\tlnQ\t\td\t\td2')
        d = [lnQavib(T, avibs, der, dT) for der in [0,1,2]]
        print('numeric\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        d = [lnQavib(T, avibs, der) for der in [0,1,2]]
        print('analyt\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
    
    if len(ladder):
        print('\nladder')
        print('\tlnQ\t\td\t\td2')
        d = [lnQladder(T, ladder, der, dT) for der in [0,1,2]]
        print('numeric\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        d = [lnQladder(T, ladder, der) for der in [0,1,2]]
        print('analyt\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))
        # also compare with lnQelec()
        g = [1] * len(ladder)
        d = [lnQelec(T, g, ladder, der) for der in [0,1,2]]
        print('Qel()\t{:.5e}\t{:.5e}\t{:.5e}'.format(*d))   
##
def avib_limit(avib):
    # set upper energy limit to anharmonic oscillator, if missing
    # also check for reasonable values
    # no return value, modify in-place
    # if you know the dissociation energy, be sure the limit does not exceed it
    [w, x] = avib[0:2]
    if w <= 0:
        print('** Error: we =', w, 'must be positive')
        sys.exit(1)
    if x <= 0:
        print('** Warning: wexe =', w, 'should probably be positive')
    vmax = w // (2*x) - 1
    emax = w*vmax - x*vmax*(vmax + 1)
    if emax < 0:
        # anharmonicity is negative
        emax = 20 * 1000   # arbitrary upper limit (cm-1)
    if len(avib) > 2:
        ulim = avib[2]  # user's upper limit
        # replace user's limit if it's too low or too high
        if (ulim < (w - 2*x)) or (ulim > emax):
            print('** Warning: replacing invalid anharmonic upper limit of', ulim)
            ulim = emax
        avib[2] = ulim
    else:
        # user did not specify an upper limit
        ulim = emax
        avib.append(ulim)
    return 
##
def avib_vlim(avib):
    # return the upper-limiting value of the vibrational quantum number v
    # there are two roots to the quadratic equation; return the lower one
    [w, x, u] = avib
    d = (w-x)**2 - 4*x*u
    vlim = (w - x - np.sqrt(d)) // (2 * x)
    return vlim
##

############## read the command line
if len(sys.argv) < 2:
    sys.exit('\tUsage: python thermo.py [gaufreq_output_file|data_file] <freq scale factor> <temperature> <NOSAVE>')
    # any file suffix is OK; it does not have to be .out or .dat
finp = sys.argv[1]  # name of input file

# additional command-line arguments?
scale = 1.0   # for scaling vibrational frequencies
symno = 1     # external symmetry number
tstep = 50.   # temperature increment
nowrite = False  # default is to write the data file
for i in range(2, len(sys.argv)):
    a = sys.argv[i]
    try:
        a = float(a)
        if abs(a - 1) <= 0.5:
            # number is close to 1; assume it's a vibrational scaling factor
            scale = a
        elif a >= 10:
            # assume it's a temperature
            specialt.append(a)
    except ValueError:
        # could not cast as float; look for special codes
        if a[0].upper() == 'S':
            # allow a wild scaling factor when prefixed with 's'
            scale = float(a[1:])
        elif a[0:2].upper() == 'DT':
            # user-specified temperature increment, e.g. dt25
            tstep = float(a[2:])
        elif a.upper() == 'NOSAVE':
            # never write the data file
            nowrite = True

# some initialization
atom = linear = False  # flags
zpe = nneg = natom = 0

# regular expressions 
# keyword parsing in custom data file should be case-insensitive
re_gau = re.compile(r'Gaussian \d+.*Revision [A-Z]\.\d+') # indicates Gaussian
re_dat = re.compile(r'^MASS\s+\d+\.\d+')  # indicates my custom data file
re_mass = re.compile(r'^ Molecular mass:|^[mM][aA][sS][sS]')
re_symno = re.compile(r'^ ROTATIONAL SYMMETRY NUMBER|^SYMNO', re.IGNORECASE)
re_rot = re.compile(r'^ ROTATIONAL CONSTANTS \(GHZ\)|^GHZ', re.IGNORECASE)
re_freq = re.compile(r' Frequencies|^[vV][iI][bB]')
re_natom = re.compile(r'^ Atom\s+(\d+) has atomic number')
re_mult = re.compile(r'^ Charge =\s+[-]?\d+\s+Multiplicity =\s+\d+|^[eE][lL][eE][cC]')
re_avib = re.compile(r'^AVIB', re.IGNORECASE)
re_xvib = re.compile(r'^XVIB', re.IGNORECASE)

################ read the input file
print('Reading data from file {:s}\n'.format(finp))
if scale == 1.:
    print('Vibrational frequencies will not be changed')
else:
    print('Vibrational frequencies will be scaled by {:.4f}'.format(scale))
print('Standard pressure = {:.3f} kPa\n'.format(PRESS / 1000))

with open(finp, 'r') as FINP:
    intype = None   # type of input file: 'gau' or 'dat'
    rot = []  # rotational constants, if any
    freq = [] # vibrational frequencies, if any
    el_degen = []  # degeneracy of electronic states
    el_energy = [] # energies of electronic states
    avibs = []  # list of lists for anharmonic oscillators
    xvib = []   # list of all explicit energy levels
    for fline in FINP:
        # discard any comments (following '#')
        words = fline.split('#')
        line = words[0]
        if intype is None:
            # INPUT TYPE
            if re_gau.search(line):
                intype = 'gau'
            if re_dat.match(line):
                intype = 'dat'
        if re_mass.match(line):
            # MASS
            words = line.split()
            if intype == 'gau':
                mass = float(words[2])
            else:
                mass = float(words[1])
        if re_symno.match(line):
            # SYMNO
            words = line.split()
            if intype == 'gau':
                symno = int(float(words[3]))
            else:
                symno = int(words[1])
        if re_rot.match(line):
            # GHZ
            # rotational constants are in GHz
            words = line.split()
            if intype == 'gau':
                i = 3
            else:
                i = 1
            rot = [float(a) for a in words[i:]]
        if re_freq.match(line):
            # VIB
            words = line.split()
            if intype == 'gau':
                freq.extend([float(a) for a in words[2:]])
            else:
                freq.extend([float(a) for a in words[1:]])
        if re_mult.match(line):
            # ELEC
            words = line.split()
            if intype == 'gau':
                # ground-state degeneracy as spin multiplicity
                g = int(words[-1])
                el_energy = [0.]
                el_degen = [g]
                if g > 1:
                    print('** WARNING: degenerate electronic state **')
                    print('This program does not detect electronic spatial degeneracies.')
                    print('If they apply to this molecule, save and edit "thermo.dat".')
                    print('Then run:\n\tpython thermo.py thermo.dat')
            else:
                # read level degeneracy and energy
                el_degen.append(int(words[1]))
                el_energy.append(float(words[2]))
        if re_avib.match(line):
            # AVIB
            # simple anharmonic oscillator
            words = line.split()
            if len(words) >= 3:
                avib = [float(w) for w in words[1:]]
                avibs.append(avib)
            else:
                print('** Warning:  AVIB needs at least two arguments--line ignored')
                print('\t"{:s}"'.format(line.rstrip()))
        if re_xvib.match(line):
            # XVIB
            # list of energy levels (could be OPLA vibrational)
            words = line.split()
            xvib.extend([float(w) for w in words[1:]])
        m = re_natom.match(line)
        if m:
            # only Gaussian files include atom count
            n = int(m.group(1))
            natom = max(n, natom)
            
# atom, linear, or nonlinear?
# for linear, rot = [ghz] (one element)
if len(rot) == 3:
    if 0 in rot:
        linear = True
        for a in rot:
            if a != 0:
                rot = [a]
                break
elif len(rot) == 1:
    if rot[0] == 0:
        atom = True
    else:
        linear = True
else:
    atom = True

if mass == 0:
    print('*** The mass must be greater than zero ***')
    print('Be sure not to request terse output ("#T") on the Gaussian command line')
print('Molecular mass = {:.3f} u'.format(mass))
print('External symmetry number = {:d}'.format(symno))
if atom:
    print('Molecule is an atom')
elif linear:
    print('Molecule is linear with rotational constant = {:.5f} GHz'.format(rot[0]))
else:
    print('Rotational constants (GHz) =' + (' {:.5f}'*3).format(*rot))

# if number of atoms is known, check number of vibrations
nvib = 0
if natom:
    if not atom:
        nvib = 3 * natom - 6
        if linear:
            nvib += 1
vibcount = len(freq) + len(avibs)
if len(xvib):
    # assume that each zero respresents a different oscillator
    vibcount += len(xvib) - np.count_nonzero(xvib)
if natom and (nvib != vibcount):
    if (intype != 'dat'):
        print('*** The number of frequencies ({:d}) is not that expected ({:d}) ***'.format(vibcount, nvib))
        print('This may occur in a Gaussian job combining "freq=calcall" with "freq".')
        if len(freq) == (2 * nvib):
            # it's probably the case described; offer to truncate the list
            q = input('Do you want to delete the second half of the frequencies? ')
            if 'y' == q[0].lower():
                freq = freq[:nvib]
            else:
                print('Please check your input file and try again.')
                sys.exit(1)
    else:
        # custom data file
        print('** Warning: Found {:d} oscillators, expected {:d}'.format(vibcount, nvib))
if natom == 0:
    # custom data input; infer the number of atoms 
    if linear or atom:
        natom = (vibcount + 5) // 3
    else:
        natom = (vibcount + 6) // 3
    if vibcount == 1:
        print('There is 1 oscillator so there should be 2 atoms')
    else:
        print('There are {:d} oscillators so there should be {:d} atoms'.format(vibcount, natom))

# print frequencies
if len(freq):
    print('Harmonic vibrational frequencies (cm^-1):')
    if scale != 1:
        print('\tunscaled\tscaled')
    else:
        print('\tunscaled')
rawfreq = np.array(freq)
freq = scale * rawfreq
for i, fr in enumerate(freq):
    if scale == 1:
        print('{:3d}\t{:6.1f}'.format(i+1, fr))
    else:
        print('{:3d}\t{:6.1f}\t\t{:6.1f}'.format(i+1, rawfreq[i], fr))
    if (fr == 0):
        print('\t*** Vibrational frequencies cannot be zero!')
        sys.exit(1)
    if (fr < 0):
        print('\t*** Vibrational frequencies cannot be negative!')
        sys.exit(1)

# if no electronic states, add singlet ground state
if len(el_energy) == 0:
    el_degen = [1]
    el_energy = [0.]
else:
    print('Electronic states:')
    print('\tdegen\tcm^-1\tkJ/mol')
    for i, (d, e) in enumerate(zip(el_degen, el_energy)):
        print('{:3d}\t{:2d}\t{:.1f}\t{:.2f}'.format(i+1, d, e, e * CM2KJ))

# any anharmonic oscillators
if len(avibs):
    print('Anharmonic oscillators (cm^-1):')
    print('\twe\twexe\ttruncation')
    for i, avib in enumerate(avibs):
        # install upper limit if missing
        avib_limit(avib)  # modifies avib in-place
        print('{:3d}\t{:.1f}\t{:.1f}\t{:.1f}'.format(i+1, *avib))

# any explicit energy ladder
if len(xvib):
    print('Explicit vibrational energy ladder:')
    # sort
    xvib.sort()
    # print unique values only (with degeneracies)
    d = {e: xvib.count(e) for e in xvib}
    s = '\t'
    for e in sorted(d.keys())[:13]:
        if d[e] == 1:
            s += '{} '.format(e)
        else:
            s += '{}({:d}) '.format(e, d[e])
    print(s)
    if len(d) > 12:
        print('\t({:d} levels up to {:.1f} cm^-1)'.format(len(d), xvib[-1]))

# save data to custom data file, unless that was the input
if (intype != 'dat') and (not nowrite):
    fdat = 'thermo.dat'
    writedat = True
    if os.path.isfile(fdat):
        q = input('\n** Overwrite file "{:s}"? '.format(fdat))
        writedat = False
        if len(q) and (q[0].lower() == 'y'):
            writedat = True
    if writedat:
        with open(fdat, 'w') as FDAT:
            FDAT.write('MASS\t{:.6f}\n'.format(mass))
            FDAT.write('GHZ' + ('\t{:.6f}'*len(rot)).format(*rot) + '\n')
            if len(freq) > 0:
                FDAT.write('VIB\t' + ('{:.1f} '*len(freq)).format(*freq) + '\n')
            FDAT.write('SYMNO\t{:d}\n'.format(symno))
            for d, e in zip(el_degen, el_energy):
                FDAT.write('ELEC\t{:d}\t{:.1f}'.format(d, e) + '\n')
        print('Input data (including scaled vibrational frequencies) written to file "{:s}"'.format(fdat))


########## begin calculations ###########
z = freq.sum() / 2
for avib in avibs:
    z += avib[0] / 2 - avib[1] / 4
if len(xvib):
    xmin = min(xvib)
    if xmin != 0:
        print('** Warning: smallest XVIB =', xmin, ' is non-zero; add to ZPE')
        # add xmin to the VZPE
        z += xmin
zpe = z * CM2KJ  # convert to kJ/mol
print('\nVibrational zero-point energy = {:.1f} cm^-1 = {:.2f} kJ/mol'.format(z, zpe))

# list of temperatures
t = tmin = 100
tmax = 1000
tlist = specialt.copy()
while t <= tmax:
    tlist.append(t)
    t += tstep
tlist.sort()

print('\nT (K)\tS (J/K/mol)\tCp (J/K/mol)\tddH (kJ/mol)')
for T in tlist:
    # loop over temperatures
    # get lnQ and its derivatives for the total partition function
    dlnQ_rrho = [lnQvrt(T, freq, symno, rot, mass, PRESS, deriv=der) for der in [0,1,2]]
    dlnQ_elec = [lnQelec(T, el_degen, el_energy, deriv=der) for der in [0,1,2]]
    dlnQ_avib = [lnQavib(T, avibs, deriv=der) for der in [0,1,2]]
    dlnQ_xvib = [lnQladder(T, xvib, deriv=der) for der in [0,1,2]]
    # add them
    dlnQ = [dlnQ_rrho[d] + dlnQ_elec[d] + dlnQ_avib[d] + dlnQ_xvib[d] for d in [0,1,2]]
    x = thermo_lnQ(T, dlnQ)
    print('{:.2f}\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(T, *x))

#test_derivs(1000, freq, rot, symno, el_degen, el_energy, avibs, xvib, dT=0.1)
