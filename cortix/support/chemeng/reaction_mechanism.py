#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment
# https://cortix.org
"""Support class for working with chemical reactions.
"""

import copy
import math
import numpy       # used in asserts
import numpy as np
import scipy as sp
from itertools import combinations
import random

from cortix.support.species import Species
from cortix.support.units import Units as unit

class ReactionMechanism:
    """Chemical reaction mechanism.

    Quantities and services: stoichiometric matrix, mass conservation, reaction rate density vector,
    species production rate density vector.

    Attributes
    ----------

    header: str
        Concatenated string of comments preceded by # in the reaction header.

    reactions: list(str)
        List of strings for each reaction.

    species_names: list(str)
        List of species names.

    species: list(Species)
        List of Species corresponding to the `species_names` list.

    data: list(dict)
        List (one element per reaction) of dicitionaries (key, value) of data given in the reaction
        mechanism.  Special keys: `alpha`, `beta` contain a dictionary of the empirical power-law
        exponents of reaction rates for reactants and products respectively; `k_f`, `k_b` contain the
        reaction rate constants  per reaction; `info` contains any `string` type for information about
        the reaction.
        E.g.: self.data[idx]['alpha'] = {'e^-': 1e7, 'NO3': 15}
              self.data[idx]['k_f'] = 1e3
              self.data[idx]['info'] = 'condensation reaction'

    stoic_mtrx: numpy.ndarray
        Stoichiometric matrix; 2D `numpy` array.

    latex_species: str
        String containing the LaTeX typesetting of the species in the order they appear in the stoichiometric
        matrix. Use the Python print() function to print this attribute and copy/paste into a LaTex
        environment. For example do print(rxn_mech.latex_species) in a python script or Jupyter notebook.

    latex_rxn: str
        String containing the LaTeX typsetting of all reactions into the LaTeX `align` environment.
        Use the Python print() function to print this attribute and copy/paste into a LaTex environment.
        This is also used with md_print() for printing in a Jupyter notebook cell. For example do
        print(rxn_mech.latex_rxn) in a python script or Jupyter notebook.
    """

    def __init__(self, header='no-header', file_name=None, mechanism=None, order_species=True,
                 reparam=False, bounds=None, log=None, args_dict=None):
        """Module class constructor.

        Build data structures for a reaction mechanism. Namely, species list,
        reactions list, data (equilibrium constant list, etc.), and stoichiometric matrix.

        Reaction mechanism format is as follows:

        # comment
        # comment
        4 NH3 + 5 O2 <=> 4 NO + 6 H2O : quantity1 = 2.5e+02 : quantity2 = '2+2'

        Data follows reaction after ":" as `quantity` = `number`. A `data` dictionary will be
        created as: data['`quantity`'] = `number`. Any number of pairs of keys,value can be
        given.

        Special control word for `alpha` and `beta` data given as power-law exponent of the
        reaction rate empirical formula in the order of reactants and products. E.g.:

        4 NH3 + 5 O2 <=> 4 NO + 6 H2O  :  alpha = (4.3, 5.8) : beta = (4.1, 6.7)

        therefore, alpha[0] is the empirical power-law exponent of the NH3 reactant.
        The data will be stored as a dictionary of a dictionary.
        E.g.: self.data['alpha'] = {'NH3': 4.3, 'O2': 5.8}


        Any amount of spacing in the reaction mechanism description is allowed except:

        a) there must be only one blank space between the stoichiometric coefficient and its
           species name.

        b) there must be at least one blank space before and after each + sign in the reaction.

        c) species formula name are stated as described in Cortix Species() e.g.:
            charge preceded with ^: O2^2+, X^1-, X^+, X^-
            phase in parenthesis (): O2^2+(a), O2^2+(aq), O2^2+(aqu)
            etc.

        There must be no blank lines. A stoichiometric coefficient equal to 1 can be ommited.

        Additional information can be passed into the object by using the keyword *dict*
        starting on the first column as follows, e.g.:

        dict: H2O*[C4H9O]3PO(o) : TBP-water complexation in (o) phase

        Parameters
        ----------
        file_name: str
            Full path file name of the reaction mechanism file. If the reaction has data
            sucha as equilibrium constant or reaction rate constant, etc., it will follow the
            reaction separated by a colon.

        mechanism: list(str)
            List of reaction strings per above convention. Alternative to information contained
            in a file.

        order_species: bool
            Alphabetically order the species names in the mechanism.

        reparam: bool
           TODO: Danger; this needs a whole re-design because, reparameterization depends on bounds!!!
           That is, if reparam is True and bounds is None, then the reparameterization is a plain
           exponential.

           This following comment only applies for the case when bounds is None.
           Reparameterize the reaction rate parameters: all reaction rate constants and power-law
           exponents using an exponential transformation: e.g. k_f = exp(k_f'). Now k_f' is the
           actual parameter used in the reaction rate law. This way k_f is guaranteed to be positive
           for any value of k_f'. The entire mechanism is affected.

        bounds: dict(tuple)

           Dictionary of bounds for k_f, k_b, alpha, and beta. E.g. bounds['k_f'] is a tuple (pair) of
           vectors, first vector lower bounds (m reactions long), second vector upper bounds (all reactions).

           E.g. bounds['alpha'] is a tuple (pair) of lists (m reactions long) of vectors of lower bounds and
           upper bounds for reactant species.
           If bounds is not None, then the exponential reparameterization is used with bounds.
           If bounds is None, then a different exponential reparameterization without bounds is used.
           Hence the data type of bounds decides the functional form of the reparameterization.

        Examples
        --------
       """

        self.file_name = file_name

        assert file_name is not None or mechanism is not None
        assert isinstance(header, str)
        self.header = header

        if mechanism is not None:
            assert isinstance(mechanism, list)
            assert file_name is None

        if file_name is not None:
            assert isinstance(file_name, str)
            assert mechanism is None

            finput = open(file_name, 'rt')

            mechanism = list()

            for line in finput:
                stripped_line = line.strip()
                mechanism.append(stripped_line)

            finput.close()

        assert isinstance(reparam, bool)

        self.reparam = reparam

        try:
            self.kf_bnds = bounds['kf']
        except:
            self.kf_bnds = None

        try:
            self.kb_bnds = bounds['kb']
        except:
            self.kb_bnds = None

        try:
            self.alpha_bnds = bounds['alpha']
        except:
            self.alpha_bnds = None

        try:
            self.beta_bnds = bounds['beta']
        except:
            self.beta_bnds = None

        self.log = log

        if args_dict is not None:
            assert isinstance(args_dict, dict)
        else:
            args_dict = dict()

        self.args_dict = args_dict

        self.reactions = list()
        self.data = list()

        # Keep the original data internally for future use in sub-mechanisms
        # Keep header and comment lines
        # Also, use any zeros in the alpha/beta coefficients to indicate that the corresponding species
        # is chemically inactive for kinetic purposes.
        self.__original_mechanism = list()

        # First: read the data in the mechanism

        info_dict = dict() # keyword dict in the input to create an info dict

        for m_i in mechanism:

            if m_i[0].strip() == '#':
                if self.header == 'no-header':
                    self.header = m_i.strip() + '\n'
                else:
                    self.header += m_i.strip() + '\n'
                continue

            if m_i[:5] == 'dict:':
                data = m_i.split(':')
                assert len(data) == 3, 'dict entry must be: dict: key:value'
                info_dict[data[1].strip()] = data[2].strip()
                continue

            self.__original_mechanism.append(m_i)

            if m_i[:5] != 'dict:':
                assert m_i.find(':') == -1, 'Change all ":" to ";".'

            data = m_i.split(';')

            self.reactions.append(data[0].strip())  # do not save comments

            tmp_dict = dict()

            if len(data) > 1: # if semi-colon separated data exists

                for d in data[1:]: # other information beyond the reaction
                    datum = d.strip()

                    name = datum.split('=')[0].strip()
                    val_str = datum.split('=')[1].strip()

                    # alpha and beta names are reserved for the exponents of the reaction rate law
                    # alpha and beta cases; convert tuple of integers into dictionary
                    if name in ('alpha', 'beta'):
                        assert '(' in val_str and ')' in val_str
                        alpha_or_beta = val_str[1:-1] # ignore "(" and ")"
                        alpha_or_beta_dict = {} # dict()
                        i = 0
                        for val_s in alpha_or_beta.split(','):
                            val = float(val_s.strip())
                            if self.reparam and val != 0.0:
                                #WARNING alpha_or_beta_dict[i] = math.log(val)
                                alpha_or_beta_dict[i] = val # user must pass the reparm theta values
                            elif val == 0:
                                alpha_or_beta_dict[i] = -9999  # internal flag for inactive species
                            else:
                                alpha_or_beta_dict[i] = val
                            i += 1
                        tmp_dict[name] = alpha_or_beta_dict
                    # any other colon separated data
                    elif name == 'info':
                        tmp_dict[name] = val_str
                    elif name == 'k_f' or name == 'k_b':
                        if self.reparam:
                            #WARNING tmp_dict[name] = math.log(float(val_str))
                            tmp_dict[name] = float(val_str) # user must pass the reparam theta values
                        else:
                            tmp_dict[name] = float(val_str)
                    else:
                        tmp_dict[name] = float(val_str)

            # mass transfer relaxation reaction sanity check
            if 'k_eq' in tmp_dict:
                assert 'tau' in tmp_dict
            if 'tau' in tmp_dict and 'k_eq' not in tmp_dict:
                s1 = 'ReactionMechanism: user must implement a k_eq_func(rxn_mech, temperature, '
                s2 = 'spc_molar_cc, arg_dict=None) for '
                if self.log:
                    self.log.warn(s1+s2+'%s'%(data[0]))
                else:
                    print('WARNING: '+s1+s2+'%s'%(data[0]))

            self.data.append(tmp_dict)

        # Second: find species

        species_tmp = list()  # temporary list for species

        for idx, rxn in enumerate(self.reactions):

            # the order of the following test matters; test reversible reaction first
            tmp = rxn.split('<=>')
            n_terms = len(tmp)
            assert n_terms == 1 or n_terms == 2

            if n_terms == 1: # if no previous split
                tmp = rxn.split('<->')
                n_terms = len(tmp)
                assert n_terms == 1 or n_terms == 2
                if n_terms == 1: # if no previous split
                    tmp = rxn.split('->')
                    n_terms = len(tmp)
                    assert n_terms == 1 or n_terms == 2
                    if n_terms == 1: # if no previous split
                        tmp = rxn.split('<-')
                        n_terms = len(tmp)
                        assert n_terms == 1 or n_terms == 2

            assert n_terms == 2 , 'tmp = %r '%tmp # must have two terms

            left = tmp[0].strip()
            right = tmp[1].strip()

            left_terms = left.split(' + ')
            right_terms = right.split(' + ')

            terms = [t.strip() for t in left_terms] + [t.strip() for t in right_terms]

            for i in terms:
                tmp = i.split(' ')
                assert len(tmp)==1 or len(tmp)==2, ' tmp = %r '%tmp
                if len(tmp) == 2:
                    species_tmp.append(tmp[1].strip())
                else:
                    species_tmp.append(i.strip())

        species_filter = set(species_tmp) # filter species as a set

        self.species_names = list(species_filter) # convert species set to list

        if order_species:
            self.species_names = sorted(self.species_names)

        # Create the species list
        self.species = list()

        for id, name in enumerate(self.species_names):
            if name in info_dict.keys():
                info = info_dict[name]
                spc = Species(name=name, formula_name=name, flag=id, info=info)
            else:
                spc = Species(name=name, formula_name=name, flag=id)
            self.species.append(spc)

        # Third: build the stoichiometric matrix

        s_mtrx = np.zeros((len(self.reactions), len(self.species)), dtype=np.float64)

        self.reaction_direction_symbol = list()

        for r in self.reactions:

            i_row = self.reactions.index(r)  # leave this here to catch repeated reactions!

            tmp = r.split(' -> ')
            n_terms = len(tmp)
            assert n_terms == 1 or n_terms == 2  # n_terms = 1 means no split
            if n_terms == 2:
                self.reaction_direction_symbol.append('->')
            if n_terms == 1:
                tmp = r.split(' <-> ')
                n_terms = len(tmp)
                assert n_terms == 1 or n_terms == 2  # n_terms = 1 means no split
                if n_terms == 2:
                    self.reaction_direction_symbol.append('<->')
                if n_terms == 1:
                    tmp = r.split(' <=> ')
                    n_terms = len(tmp)
                    assert n_terms == 1 or n_terms == 2  # n_terms = 1 means no split
                    if n_terms == 2:
                        self.reaction_direction_symbol.append('<=>')
                    if n_terms == 1:
                        tmp = r.split(' <- ')
                        n_terms = len(tmp)
                        assert n_terms == 1 or n_terms == 2  # n_terms = 1 means no split
                        if n_terms == 2:
                            self.reaction_direction_symbol.append('<-')

            assert n_terms == 2

            left = tmp[0]
            left_terms = left.split(' + ')
            left_terms = [t.strip() for t in left_terms]

            right = tmp[1]
            right_terms = right.split(' + ')
            right_terms = [t.strip() for t in right_terms]

            for t in left_terms:
                tmp = t.split(' ')
                if len(tmp) == 2:
                    coeff = float(tmp[0].strip())
                    species_member = tmp[1].strip()
                    j_col = self.species_names.index(species_member)
                    assert s_mtrx[i_row, j_col] == 0.0, \
                           'duplicates not allowed r%r: %r %r %r'%\
                           (i_row, r, species_member, s_mtrx[i_row, j_col])
                    s_mtrx[i_row, j_col] = -1.0 * coeff
                else:
                    species_member = tmp[0].strip()
                    j_col = self.species_names.index(species_member)
                    assert s_mtrx[i_row, j_col] == 0.0, \
                           'duplicates not allowed r%r: %r %r %r\n%r\n%r'%\
                           (i_row, r, species_member, s_mtrx[i_row, j_col], s_mtrx[i_row,:],
                            self.species_names)
                    s_mtrx[i_row, j_col] = -1.0

                if 'alpha' in self.data[i_row].keys():
                    species_name = self.species_names[j_col]
                    assert len(self.data[i_row]['alpha']) == len(left_terms), 'Incorrect length of alpha.'
                    # replace species index with name
                    self.data[i_row]['alpha'][species_name] = self.data[i_row]['alpha'].pop(left_terms.index(t))

            for t in right_terms:
                tmp = t.split(' ')
                if len(tmp) == 2:
                    coeff = float(tmp[0].strip())
                    species_member = tmp[1].strip()
                    j_col = self.species_names.index(species_member)
                    assert s_mtrx[i_row, j_col] == 0.0, \
                           'duplicates not allowed r%r: %r %r %r'%\
                           (i_row, r, species_member, s_mtrx[i_row, j_col])
                    s_mtrx[i_row, j_col] = 1.0 * coeff
                else:
                    species_member = tmp[0].strip()
                    j_col = self.species_names.index(species_member)
                    assert s_mtrx[i_row, j_col] == 0.0, \
                           'duplicates not allowed r%r: %r %r %r'%\
                           (i_row, r, species_member, s_mtrx[i_row, j_col])
                    s_mtrx[i_row, j_col] = 1.0

                if 'beta' in self.data[i_row].keys():
                    species_name = self.species_names[j_col]
                    assert len(self.data[i_row]['beta']) == len(right_terms), 'Incorrect length of beta.'
                    # replace species index with name
                    self.data[i_row]['beta'][species_name] = self.data[i_row]['beta'].pop(right_terms.index(t))

        self.stoic_mtrx = s_mtrx

        # Create the latex typesetting of reactions and species
        (self.latex_species, self.latex_rxn) = self.__latex()

        # Fill-in missing k_f, k_b, alpha, and beta
        for idx, dat in enumerate(self.data):
            if 'k_f' not in dat.keys():
                if self.reparam:
                    dat['k_f'] = -1e+10 # math.log(0.0)
                else:
                    dat['k_f'] = 0.0
            if 'k_b' not in dat.keys():
                if self.reparam:
                    dat['k_b'] = -1e+10 # math.log(0.0)
                else:
                    dat['k_b'] = 0.0
            if 'alpha' not in dat.keys():
                (reactants_ids, ) = np.where(self.stoic_mtrx[idx, :] < 0)
                tmp = dict()
                for j in reactants_ids:
                    spc_name = self.species_names[j]
                    if self.reparam:
                        tmp[spc_name] = math.log(abs(self.stoic_mtrx[idx, j]))
                    else:
                        tmp[spc_name] = abs(self.stoic_mtrx[idx, j])
                dat['alpha'] = tmp
            if 'beta' not in dat.keys():
                (products_ids, ) = np.where(self.stoic_mtrx[idx, :] > 0)
                tmp = dict()
                for j in products_ids:
                    spc_name = self.species_names[j]
                    if self.reparam:
                        tmp[spc_name] = math.log(abs(self.stoic_mtrx[idx, j]))
                    else:
                        tmp[spc_name] = abs(self.stoic_mtrx[idx, j])
                dat['beta'] = tmp

        # Fill-in missing info
        for idx, dat in enumerate(self.data):
            if 'info' not in dat.keys():
                dat['info'] = 'no-info'

    def mass_balance_residuals(self):
        """Reaction mass balance residual vector.

        Returns
        -------

        mb_residual_vec: numpy.ndarray
            1D vector of mass balance residuals for each reaction.
        """

        m_vec = np.zeros(len(self.species), dtype=np.float64)

        for idx, spc in enumerate(self.species):
            m_vec[idx] = spc.molar_mass

        assert np.prod(m_vec) > 0.0

        mb_residual_vec = self.stoic_mtrx @ m_vec

        return mb_residual_vec

    def max_mass_balance_residual(self):
        """Compute the maximum magnitude reaction mass balance residual.

        Returns
        -------

        max(abs(mb): float
        """

        mb_residual_vec = self.mass_balance_residuals()

        return np.max(np.abs(mb_residual_vec))

    def is_mass_conserved(self, tol=1e-10):
        """Check mass conservation if species have a molar mass value.

        Returns
        -------

        bool
        """

        residual = self.max_mass_balance_residual()

        return True if residual < tol else False

    def rank_analysis(self, verbose=False, tol=1e-8):
        '''Compute the rank of the stoichiometric matrix.

        This will establish rank deficiency.

        Parameters
        ----------

        verbose: bool
        tol: float

        Returns
        -------

        rank: int
        '''

        #assert self.is_mass_conserved(tol), 'fatal: mass conservation failed'

        s_rank = np.linalg.matrix_rank(self.stoic_mtrx, tol=tol)

        assert s_rank <= min(self.stoic_mtrx.shape)

        if verbose:
            print('# reactions = ', len(self.reactions))
            print('# species   = ', len(self.species))
            print('rank of S = ', s_rank)

            if s_rank == min(self.stoic_mtrx.shape):
                print('S is full rank.')
            else:
                print('S is rank deficient.')

        if s_rank == self.stoic_mtrx.shape[1]:
            print('***********************')
            print('Warning: rank = n')
            print('Reaction mechanism fails mass conservation')
            print('***********************')

        return s_rank

    def r_vec(self, spc_molar_cc_vec, temperature=273.15*unit.K, theta_kf_vec=None, theta_kb_vec=None,
              theta_alpha_lst=None, theta_beta_lst=None):
        '''Compute a reaction rate density vector.

        This is the most common reaction rate expression for homogeneous reactions.

                   __  alpha_i,j        __  beta_i,j
        r_i = kf_i || c          - kb_i || c
                   j   j                j   j

        Provided as a convenience to the user of a given reaction mechanism. If the reaction mechanism
        is meant to be reparameterized, the presence of theta named arguments decide whether that parameter
        is actually reparameterized or not.

        If k_eq (partition or distribution coefficient) and tau (relaxation time) are given, these will
        override the previous reaction rate expression with a mass transfer relaxation reaction
        expression where k_eq is used directly.

        r_i = - 1/tau (c  - k_eq c )
                        j         j

        If tau is given but not k_eq, then the user is responsible for providing the reaction rate
        expression. This typically happens when k_eq is a function of temperature, concentration and
        other quantities.

        Parameters
        ----------
        spc_molar_cc_vec: numpy.ndarray
        Ordered (same as the reaction mechanism) molar concentration vector.

        theta_kf_vec: numpy.ndarray
        Vector of forward reaction rate constants per reaction. If not provided, it will be assembled
        internally from `self.data`.
        Must be reparameterized (theta) if reparam is True.

        theta_kb_vec: numpy.ndarray
        Vector of backward reaction rate constants per reaction. If not provided, it will be assembled
        internally from `self.data`.
        Must be repameterized (theta) if reparam is True.

        alpha_lst: list(numpy.ndarray)
        List of alpha power-law exponents as a matrix containing the reactant species ids.
        If not provided, it will be assembled internally from `self.data`.
        Must be repameterized  (theta) if reparam is True.

        beta_lst: list(numpy.ndarray)
        List of beta power-law exponents as a matrix containing the product species ids.
        If not provided, it will be assembled internally from `self.data`.
        Must be repameterized  (theta) if reparam is True.
        '''
        assert isinstance(spc_molar_cc_vec, numpy.ndarray), 'type(spc_molar_cc_vec) = %r'%(type(spc_molar_cc_vec))
        assert spc_molar_cc_vec.size == len(self.species)

        # Initialize kf_vec
        if theta_kf_vec is not None:
            assert isinstance(theta_kf_vec, np.ndarray)
            assert theta_kf_vec.size == len(self.reactions)
            kf_vec = self.perform_reparam(theta_kf_vec, self.kf_bnds)
        else:
            kf_vec = self.__get_kf()

        # Initialize kb_vec
        if theta_kb_vec is not None:
            assert isinstance(theta_kb_vec, np.ndarray), 'type(theta_kb_vec)=%r'%(type(theta_kb_vec))
            assert theta_kb_vec.size == len(self.reactions)
            kb_vec = self.perform_reparam(theta_kb_vec, self.kb_bnds)
        else:
            kb_vec = self.__get_kb()

        # Initialize alpha_lst
        if theta_alpha_lst is not None:
            assert isinstance(theta_alpha_lst, list)
            assert len(theta_alpha_lst) == len(self.reactions), '# reactions=%r theta_alpha_lst=\n%r'%(len(self.reactions),theta_alpha_lst)
            assert isinstance(theta_alpha_lst[0], np.ndarray)
            assert theta_alpha_lst[0].shape[0] == 2

            theta_alpha_lst = copy.deepcopy(theta_alpha_lst)
            alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)
        else:
            (alpha_lst, _) = self.__get_power_law_exponents()

        # Initialize beta_lst
        if theta_beta_lst is not None:
            assert isinstance(theta_beta_lst, list)
            assert len(theta_beta_lst) == len(self.reactions)
            assert isinstance(theta_beta_lst[0], np.ndarray)
            assert theta_beta_lst[0].shape[0] == 2 #WARNING some issue with backward reaction with one species

            theta_beta_lst = copy.deepcopy(theta_beta_lst)
            beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)
        else:
            (_, beta_lst) = self.__get_power_law_exponents()

        # Compute the reaction rate density vector
        r_vec = np.zeros(len(self.reactions), dtype=np.float64)

        for (idx, rxn_data) in enumerate(self.data):

            alpha_mtrx = alpha_lst[idx]
            reactants_ids = alpha_mtrx[0, :].astype(int)

            reactants_molar_cc = spc_molar_cc_vec[reactants_ids] # must be ordered as in rxn_mech

            #reactants_molar_cc[reactants_molar_cc < 0] = 0.0

            r_vec[idx] = kf_vec[idx] * np.prod(reactants_molar_cc**alpha_mtrx[1, :])

        for (idx, rxn_data) in enumerate(self.data):

            beta_mtrx = beta_lst[idx]
            products_ids = beta_mtrx[0, :].astype(int)

            products_molar_cc = spc_molar_cc_vec[products_ids] # must be oredered as in rxn_mech

            #products_molar_cc[products_molar_cc < 0] = 0.0

            r_vec[idx] -= kb_vec[idx] * np.prod(products_molar_cc**beta_mtrx[1, :])

        # If k_eq and tau are present, override reaction rate with mass transfer relaxation
        # Very careful here...experimental coding
        # Implemented for A(a) <-> A(b)
        # The convention here is that k_eq multiplies the reactant to obtain the product.
        # A(b)_eq = k_eq * A(a)
        # Implemented for x A(a) + y B(a) <-> Ax*By(b)
        # The convention here is that k_eq multiplies the reactant to obtain the product.
        # AxBy(b)_eq = k_eq * A(a)
        for (idx, rxn_data) in enumerate(self.data):
            alpha_mtrx = alpha_lst[idx]
            beta_mtrx = beta_lst[idx]
            if 'tau' in rxn_data:
                tau = rxn_data['tau']

                if 'k_eq' in rxn_data:
                    k_eq = rxn_data['k_eq']
                elif 'k_eq_func' in rxn_data:
                    k_eq_func = rxn_data['k_eq_func']
                    k_eq      = k_eq_func(self, temperature, spc_molar_cc_vec, self.args_dict)
                else:
                    assert False, 'rxn_data = %r'%rxn_data

                reactants_ids = alpha_mtrx[0, :].astype(int)

                reactants_molar_cc = spc_molar_cc_vec[reactants_ids] # must be ordered as in rxn_mech

                products_ids = beta_mtrx[0, :].astype(int)

                products_molar_cc = spc_molar_cc_vec[products_ids] # must be oredered as in rxn_mech

                # Note: based on first reactanct and first product
                reactant_molar_cc  = reactants_molar_cc[0] # first reactant
                product_molar_cc = products_molar_cc[0]  # first product

                product_eq_molar_cc = k_eq * reactant_molar_cc

                #print('')
                #print('k_eq=', k_eq)
                #print('reactant_molar_cc=', reactant_molar_cc)
                #print('product_molar_cc=', product_molar_cc)
                #print('product_eq_molar_cc=', product_eq_molar_cc)

                r_vec[idx] = - 1/tau * (product_molar_cc - product_eq_molar_cc)

        return r_vec

    def rxn_rate_law(self, spc_molar_cc_vec, kf_vec=None, kb_vec=None, alpha_lst=None, beta_lst=None):
        '''See r_vec.
        '''

        return self.r_vec(spc_molar_cc_vec, kf_vec, kb_vec, alpha_lst, beta_lst)

    def g_vec(self, spc_molar_cc_vec, temperature=273.15*unit.K,
              theta_kf_vec=None, theta_kb_vec=None, 
              theta_alpha_lst=None, theta_beta_lst=None):
        '''Compute the species production rate density vector.

        Compute g(c_vec, theta) with theta being a reparameterized value

        Parameters:
        -----------
        '''

        g_vec = self.stoic_mtrx.transpose() @ self.r_vec(spc_molar_cc_vec, temperature, 
                                                         theta_kf_vec, theta_kb_vec,
                                                         theta_alpha_lst, theta_beta_lst)

        return g_vec

    def __unbounded_reparam(self, lst_or_vec):

        assert False

        lst_or_vec = copy.deepcopy(lst_or_vec)

        if isinstance(lst_or_vec, list):

            beta_or_alpha_lst = lst_or_vec

            for idx, mtrx in enumerate(beta_or_alpha_lst):
                beta_or_alpha_lst[idx] = np.array((mtrx[0, :], np.exp(mtrx[1, :])))

            reparamed = beta_or_alpha_lst
        else:
            k_vec = lst_or_vec
            reparamed = np.exp(k_vec)

        return reparamed

    def __bounded_reparam(self, lst_or_vec, bnds):
        '''Phi(theta) reparam with bounds on phi.

        That is, pass theta values (new parameters) through argument, return phi values
        (original parameters).
        '''

        lst_or_vec = copy.deepcopy(lst_or_vec)  # theta values

        if isinstance(lst_or_vec, list):

            min_beta_or_alpha = bnds[0]
            max_beta_or_alpha = bnds[1]

            theta_beta_or_alpha_lst = lst_or_vec
            beta_or_alpha_lst = lst_or_vec  # reusing space

            for irxn, mtrx in enumerate(theta_beta_or_alpha_lst):

                a_vec = min_beta_or_alpha[irxn]
                b_vec = max_beta_or_alpha[irxn]

                theta_beta_or_alpha_vec = mtrx[1, :]

                theta_beta_or_alpha_vec[theta_beta_or_alpha_vec>700] = 700

                beta_or_alpha_vec = b_vec - (b_vec - a_vec) / (1 + np.exp(theta_beta_or_alpha_vec))

                beta_or_alpha_lst[irxn] = np.vstack([mtrx[0, :], beta_or_alpha_vec])

            phi = beta_or_alpha_lst

        elif isinstance(lst_or_vec, np.ndarray):

            theta_k_vec = lst_or_vec
            assert len(theta_k_vec.shape) == 1

            min_k = bnds[0]
            max_k = bnds[1]

            theta_k_vec[theta_k_vec>700] = 700

            phi = max_k - (max_k - min_k) / (1 + np.exp(theta_k_vec))

        else:
            assert False

        return phi

    def perform_reparam(self, theta_lst_or_vec, bnds=None):
        '''Phi(theta) function (reparameterization function).

        Phi is the original parameters (k_f, k_b, alpha, beta), theta is the nonlinear reparameterized
        values.

        Reparam function
        phi(theta) = min_theta + (max_theta - min_theta) / (1 + np.exp(theta))

        '''
        #print(self.reparam)

        if self.reparam == False:
            return theta_lst_or_vec
        elif bnds is not None:
            reparamed = self.__bounded_reparam(theta_lst_or_vec, bnds)
        else:
            reparamed = self.__unbounded_reparam(theta_lst_or_vec)

        return reparamed # return phi(theta)

    def __inv_unbounded_reparam(self, lst_or_vec):

        lst_or_vec = copy.deepcopy(lst_or_vec)

        if isinstance(lst_or_vec, list):

            beta_or_alpha_lst = lst_or_vec

            for idx, mtrx in enumerate(beta_or_alpha_lst):
                beta_or_alpha_lst[idx] = np.array((mtrx[0, :], np.log(mtrx[1, :])))

            reparamed = beta_or_alpha_lst
        else:
            k_vec = lst_or_vec
            reparamed = np.log(k_vec)

        return reparamed

    def __inv_bounded_reparam(self, lst_or_vec, bnds):
        '''Theta(phi) reparam with bounds on phi.

        That is, pass phi values (orig parameters) through argument, return theta values
        (new parameters).
        '''

        lst_or_vec = copy.deepcopy(lst_or_vec) # phi values

        if isinstance(lst_or_vec, list):
            min_beta_or_alpha = bnds[0]
            max_beta_or_alpha = bnds[1]

            beta_or_alpha_lst = lst_or_vec
            theta_beta_or_alpha_lst = lst_or_vec # reusing space

            for irxn, mtrx in enumerate(beta_or_alpha_lst):

                a_vec = min_beta_or_alpha[irxn]
                b_vec = max_beta_or_alpha[irxn]

                beta_or_alpha_vec = mtrx[1, :]

                theta_beta_or_alpha_vec = np.log((a_vec - beta_or_alpha_vec) / \
                                          (beta_or_alpha_vec - b_vec))

                theta_beta_or_alpha_lst[irxn] = np.vstack([mtrx[0, :], theta_beta_or_alpha_vec])

            theta = theta_beta_or_alpha_lst

        else:
            k_vec = lst_or_vec
            min_k = bnds[0]
            max_k = bnds[1]

            theta = np.log((min_k - k_vec)/(k_vec - max_k))

        return theta

    def inv_reparam(self, phi_lst_or_vec, bnds=None):
        '''Theta(phi) function (inverse parameterization function).

        Phi is the original parameters (k_f,k_b,alpha,beta), theta is the nonlinear reparameterized
        values.

        Inverse of reparam function
        theta(phi) = ln( min_phi - phi / phi - max_phi )
        for min_phi < phi < max_phi
        '''
        if bnds is not None:
            params = self.__inv_bounded_reparam(phi_lst_or_vec, bnds)
        else:
            params = self.__inv_unbounded_reparam(phi_lst_or_vec)

        return params # return theta(phi)

    def __dphi_dtheta(self, theta_lst_or_vec, bnds=None):
        '''Derivative of the original parameter (phi) wrt the working parameter (theta).

        That is, derivative of k_f, k_b, alpha, and beta wrt the nonlinear parameter theta.
        '''
        theta_lst_or_vec = copy.deepcopy(theta_lst_or_vec)

        # TODO: move this to constructor for a single test
        if bnds is not None:
            assert isinstance(bnds, tuple)
            assert len(bnds) == 2

        if bnds is not None and isinstance(theta_lst_or_vec, list):
            assert isinstance(bnds[0], list)
            assert isinstance(bnds[1], list)

        if bnds is not None and isinstance(theta_lst_or_vec, np.ndarray):
            assert isinstance(bnds[0], np.ndarray)
            assert isinstance(bnds[1], np.ndarray)

        lst_or_vec = theta_lst_or_vec

        if self.reparam is False:

            if isinstance(lst_or_vec, list):

                dphi_dtheta_lst = lst_or_vec  # reuse space

                for irxn, theta_mtrx in enumerate(dphi_dtheta_lst):

                    dphi_dtheta_vec = np.ones(theta_mtrx[1, :].size)
                    dphi_dtheta_lst[irxn] = np.vstack([theta_mtrx[0, :], dphi_dtheta_vec])

                dphi_dtheta = dphi_dtheta_lst
            else:

                dphi_dtheta_vec = np.ones(lst_or_vec.size)
                dphi_dtheta = dphi_dtheta_vec

        elif bnds is not None:

            if isinstance(lst_or_vec, list):

                min_beta_or_alpha = bnds[0]
                max_beta_or_alpha = bnds[1]

                dphi_dtheta_lst = lst_or_vec  # reuse space

                for irxn, theta_mtrx in enumerate(lst_or_vec):

                    a_vec = min_beta_or_alpha[irxn]
                    b_vec = max_beta_or_alpha[irxn]

                    theta_vec = theta_mtrx[1, :]

                    dphi_dtheta_vec = ((b_vec - a_vec)*np.exp(theta_vec))/(np.exp(theta_vec) + 1)**2

                    dphi_dtheta_lst[irxn] = np.vstack([theta_mtrx[0, :], dphi_dtheta_vec])

                dphi_dtheta = dphi_dtheta_lst

            else:

                theta_vec = lst_or_vec
                a_vec = bnds[0]
                b_vec = bnds[1]

                #theta_vec[np.abs(theta_vec)<=1e-50] = 1e+50

                dphi_dtheta_vec = ((b_vec - a_vec)*np.exp(theta_vec))/(np.exp(theta_vec) + 1)**2
                dphi_dtheta = dphi_dtheta_vec

        else:

            assert False # fix this
            dphi_dtheta = self.perform_reparam(lst_or_vec)

        return dphi_dtheta

    def __d2phi_dtheta2(self, theta_lst_or_vec, bnds=None):
        '''2nd derivative of the original parameter (phi) wrt the working parameter (theta).
        '''

        theta_lst_or_vec = copy.deepcopy(theta_lst_or_vec)

        if self.reparam is False:

            if isinstance(theta_lst_or_vec, list):

                d2phi_dtheta2_lst = theta_lst_or_vec # reuse data type

                for idx, theta_mtrx in enumerate(d2phi_dtheta2_lst):

                    d2phi_dtheta2_vec = np.zeros(theta_mtrx[1, :].size)
                    d2phi_dtheta2_lst[idx] = np.vstack([theta_mtrx[0, :], d2phi_dtheta2_vec])

                d2phi_dtheta2 = d2phi_dtheta2_lst
            else:

                d2phi_dtheta2_vec = np.zeros(theta_lst_or_vec.size)
                d2phi_dtheta2 = d2phi_dtheta2_vec

        elif bnds is not None:
            if isinstance(theta_lst_or_vec, list):

               min_beta_or_alpha = bnds[0]
               max_beta_or_alpha = bnds[1]

               d2phi_dtheta2_lst = theta_lst_or_vec

               for idx, theta_mtrx in enumerate(d2phi_dtheta2_lst):

                   a_vec = min_beta_or_alpha[idx]
                   b_vec = max_beta_or_alpha[idx]

                   theta_vec = theta_mtrx[1, :]

                   d2phi_dtheta2_vec = (b_vec - a_vec) * (1 - np.exp(theta_vec)) * np.exp(theta_vec) / \
                                       (np.exp(theta_vec) + 1)**3

                   #print('a_vec = ', a_vec)
                   #print('b_vec = ', b_vec)

                   #print('d2phi_dtheta2_vec = ', d2phi_dtheta2_vec)

                   d2phi_dtheta2_lst[idx] = np.vstack([theta_mtrx[0, :], d2phi_dtheta2_vec])

               d2phi_dtheta2 = d2phi_dtheta2_lst

            else:

                theta_vec = theta_lst_or_vec
                a_vec = bnds[0]
                b_vec = bnds[1]

                #print('a_vec = ', a_vec)
                #print('b_vec = ', b_vec)

                d2phi_dtheta2_vec = (b_vec - a_vec) * (1 - np.exp(theta_vec)) * np.exp(theta_vec) / \
                                    (np.exp(theta_vec) + 1)**3

                #print('d2phi_dtheta2_vec =  ', d2phi_dtheta2_vec[0])

                d2phi_dtheta2 = d2phi_dtheta2_vec

        else:

            d2phi_dtheta2 = self.perform_reparam(theta_lst_or_vec)

        return d2phi_dtheta2

    def species_prod_rate_dens(self, spc_molar_cc_vec, theta_kf_vec=None, theta_kb_vec=None, theta_alpha_lst=None, theta_beta_lst=None):
        '''Compute the species production rate density vector.

        Parameters:
        -----------
        '''

        return self.g_vec(spc_molar_cc_vec, theta_kf_vec, theta_kb_vec, theta_alpha_lst, theta_beta_lst)

    def dr_dtheta_mtrx(self, spc_molar_cc_vec,
                       theta_kf_vec=None, theta_kb_vec=None, theta_alpha_lst=None, theta_beta_lst=None):
        '''Partial derivative of the reaction rate law vector wrt working parameters.

        Theta is the working parameter (theta(phi)).

        Note: dr_dtheta = dr_dphi dphi_dtheta

        The parameters in the derivative are ordered as: k_fs, k_bs, alphas, betas.
        If a parameter is `None`, it indicates that it is not to be considered a varying parameter.
        As of now parameter sensitivity is either on or off for all kf's , or k'bs, or alphas, or betas.
        Maybe this can be relaxed for individual reaction parameters later.

        The matrix is m x p. Where m is the number of reactions, p is the total number of parameters.
        That is, p = 2 * m + n_alpha + n_beta, where n_alpha is the number of total active species in the
        forward reactions, and n_beta is the number of total active species in the reverse reactions.
        If say, theta_alpha_lst is not a varying parameter, then n_alpha = 0.

          m x p
        dr_dtheta = ( P dkf_dtheta_kf  -Q dkb_dtheta_kb  X dalpha_dtheta_alpha  -Y dbeta_dtheta_beta )

        This partial derivative matrix is instrumental to compute other quantities in particular the
        partial derivative of g wrt parameters, dg_dtheta. This is equal to the Jacobian matrix of the
        least squares residual.
        '''

        assert isinstance(spc_molar_cc_vec, np.ndarray)
        assert spc_molar_cc_vec.size == len(self.species)
        assert np.all(spc_molar_cc_vec >= 0), 'spc_molar_cc_vec =\n%r' % spc_molar_cc_vec

        # --------------------------------------------
        # partial_theta_kf(r) = P partial_theta_kf(kf)
        # --------------------------------------------
        if theta_kf_vec is not None:

            # Compute P

            if theta_alpha_lst is None:
                (alpha_lst, _) = self.__get_power_law_exponents()
            else:
                theta_alpha_lst = copy.deepcopy(theta_alpha_lst)
                alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)

            p_vec = np.zeros(len(self.reactions), dtype = np.float64)

            for (irxn, alpha_data_mtrx) in enumerate(alpha_lst):

                active_spc_ids = alpha_data_mtrx[0, :].astype(int)

                alpha_i_vec = alpha_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

                p_vec[irxn] = spc_cc_power_prod

            p_mtrx = np.diag(p_vec)

            # Compute partial_theta_kf(kf)

            theta_kf_vec = copy.deepcopy(theta_kf_vec)

            dkf_dtheta_vec = self.__dphi_dtheta(theta_kf_vec, self.kf_bnds)

            dkf_dtheta_mtrx = np.diag(dkf_dtheta_vec)

            # Store product

            try:
                dr_dtheta_mtrx = np.hstack([dr_dtheta_mtrx, p_mtrx @ dkf_dtheta_mtrx])
            except NameError:
                dr_dtheta_mtrx = np.hstack([p_mtrx @ dkf_dtheta_mtrx])

        # --------------------------------------------
        # partial_theta_kb(r) = - Q partial_theta_kb(kb)
        # --------------------------------------------
        if theta_kb_vec is not None:

            # Compute Q

            if theta_beta_lst is None:
                (_, beta_lst) = self.__get_power_law_exponents()
            else:
                theta_beta_lst = copy.deepcopy(theta_beta_lst)
                beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)

            q_vec = np.zeros(len(self.reactions), dtype = np.float64)

            for (irxn, beta_data_mtrx) in enumerate(beta_lst):

                active_spc_ids = beta_data_mtrx[0, :].astype(int)

                beta_i_vec = beta_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

                q_vec[irxn] = spc_cc_power_prod

            q_mtrx = np.diag(q_vec)

            # Compute partial_theta_kb(kb)

            theta_kb_vec = copy.deepcopy(theta_kb_vec)

            dkb_dtheta_vec = self.__dphi_dtheta(theta_kb_vec, self.kb_bnds)

            dkb_dtheta_mtrx = np.diag(dkb_dtheta_vec)

            # Store product

            try:
                dr_dtheta_mtrx = np.hstack([dr_dtheta_mtrx, -q_mtrx @ dkb_dtheta_mtrx])
            except NameError:
                dr_dtheta_mtrx = np.hstack([-q_mtrx @ dkb_dtheta_mtrx])

        # -----------------------------------------------------
        # partial_theta_alpha(r) = X partial_theta_alpha(alpha)
        # -----------------------------------------------------
        if theta_alpha_lst is not None:

            # Compute X = partial_alpha(r)

            # get kf
            if theta_kf_vec is None:
                kf_vec = self.__get_kf()
            else:
                theta_kf_vec = copy.deepcopy(theta_kf_vec)
                kf_vec = self.perform_reparam(theta_kf_vec, self.kf_bnds)

            # get alphas
            theta_alpha_lst = copy.deepcopy(theta_alpha_lst)

            alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)

            for (irxn, alpha_data_mtrx) in enumerate(alpha_lst):

                active_spc_ids = alpha_data_mtrx[0, :].astype(int)

                alpha_i_vec = alpha_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

                rf_i = kf_vec[irxn] * spc_cc_power_prod

                min_c_j = active_spc_molar_cc.min()
                if min_c_j <= 1e-25:
                    (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                    active_spc_molar_cc[jdx] = 1.0 # any non-zero value will do since rf_i will be zero

                w_alpha_i_vec = np.log(active_spc_molar_cc)
                x_vec_i = rf_i * w_alpha_i_vec

                n_alpha_i = alpha_data_mtrx.shape[1]
                x_mtrx_j_block = np.zeros((len(self.reactions), n_alpha_i), dtype = np.float64)

                x_mtrx_j_block[irxn, :] = x_vec_i

                try:
                    x_mtrx = np.hstack([x_mtrx, x_mtrx_j_block])
                except NameError:
                    x_mtrx = np.hstack([x_mtrx_j_block])

            n_alphas = 0
            for alpha_data_mtrx in alpha_lst:
                n_alphas += alpha_data_mtrx.shape[1]

            assert x_mtrx.shape == (len(alpha_lst), n_alphas), 'n_alphas = %r; U shape = %r' % (n_alphas, x_mtrx.shape)

            # Compute partial_theta_alpha(alpha)

            dalpha_dtheta_lst = self.__dphi_dtheta(theta_alpha_lst, self.alpha_bnds)

            for dalpha_dtheta_data_mtrx in dalpha_dtheta_lst:

                #assert(dalpha_dtheta_data_mtrx[0,:] == alpha_data_mtrx[0,:]) # reactant IDs must match
                dalpha_dtheta_mtrx_i = np.diag(dalpha_dtheta_data_mtrx[1,:])

                try:
                    dalpha_dtheta_mtrx = sp.linalg.block_diag(dalpha_dtheta_mtrx, dalpha_dtheta_mtrx_i)
                except NameError:
                    dalpha_dtheta_mtrx = sp.linalg.block_diag(dalpha_dtheta_mtrx_i)

            # Store product

            try:
                dr_dtheta_mtrx = np.hstack([dr_dtheta_mtrx, x_mtrx @ dalpha_dtheta_mtrx])
            except NameError:
                dr_dtheta_mtrx = np.hstack([x_mtrx @ dalpha_dtheta_mtrx])

        # --------------------------------------------------
        # partial_theta_beta(r) = Y partial_theta_beta(beta)
        # --------------------------------------------------
        if theta_beta_lst is not None:

            # Compute Y = partial_beta(r)

            # get kb
            if theta_kb_vec is None:
                kb_vec = self.__get_kb()
            else:
                theta_kb_vec = copy.deepcopy(theta_kb_vec)
                kb_vec = self.perform_reparam(theta_kb_vec, self.kb_bnds)

            # get betas
            theta_beta_lst = copy.deepcopy(theta_beta_lst)

            beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)

            for (irxn, beta_data_mtrx) in enumerate(beta_lst):

                active_spc_ids = beta_data_mtrx[0, :].astype(int)

                beta_i_vec = beta_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

                rb_i = kb_vec[irxn] * spc_cc_power_prod

                min_c_j = active_spc_molar_cc.min()
                if min_c_j <= 1e-25:
                    (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                    active_spc_molar_cc[jdx] = 1.0  # any non-zero value will do it since rb_i will be zero

                w_beta_i_vec = np.log(active_spc_molar_cc)
                y_vec_i = rb_i * w_beta_i_vec

                n_beta_i = beta_data_mtrx.shape[1]
                y_mtrx_j_block = np.zeros((len(self.reactions), n_beta_i), dtype = np.float64)

                y_mtrx_j_block[irxn, :] = y_vec_i

                try:
                    y_mtrx = np.hstack([y_mtrx, y_mtrx_j_block])
                except NameError:
                    y_mtrx = np.hstack([y_mtrx_j_block])

            n_betas=0
            for beta_data_mtrx in beta_lst:
                n_betas += beta_data_mtrx.shape[1]

            assert y_mtrx.shape == (len(beta_lst), n_betas), 'n_betas = %r; V shape = %r' % (n_betas, y_mtrx.shape)

            # Compute partial_theta_beta(beta)

            dbeta_dtheta_lst = self.__dphi_dtheta(theta_beta_lst, self.beta_bnds)

            for dbeta_dtheta_data_mtrx in dbeta_dtheta_lst:

                #assert(dbeta_dtheta_data_mtrx[0,:] == beta_data_mtrx[0,:]) # reactant IDs must match
                dbeta_dtheta_mtrx_i = np.diag(dbeta_dtheta_data_mtrx[1,:])

                try:
                    dbeta_dtheta_mtrx = sp.linalg.block_diag(dbeta_dtheta_mtrx, dbeta_dtheta_mtrx_i)
                except NameError:
                    dbeta_dtheta_mtrx = sp.linalg.block_diag(dbeta_dtheta_mtrx_i)

            # Store product

            try:
                dr_dtheta_mtrx = np.hstack([dr_dtheta_mtrx, (-y_mtrx @ dbeta_dtheta_mtrx)])
            except NameError:
                dr_dtheta_mtrx = np.hstack([-y_mtrx @ dbeta_dtheta_mtrx])

        return dr_dtheta_mtrx

    def dr_dtheta_mtrx_numerical(self, spc_molar_cc_vec,
                      kf_vec = None, kb_vec =None, alpha_lst=None, beta_lst=None,
                      h_small = 1e-6):
        '''Numerical partial derivative of the reaction rate law vector wrt parameters.

        The parameters in the derivative are ordered as: k_fs, k_bs, alphas, betas.
        If a parameter is `None`, it is not considered a varying parameter.
        As of now parameter sensitivity is either on or off for all kf's , or k'bs, or alphas, or betas.
        Maybe this can be extended for individual reaction parameters later.

        The matrix is m x p. Where m is the number of reactions, p is the total number of parameters.
        That is, p = 2 * m + n_Ri + n_Pi, where n_Ri is the number of active reactant species, and
        n_Pi is the number of active product species. If say, alpha_lst is not a varying parameter,
        then n_Ri = 0.
        '''

        assert isinstance(spc_molar_cc_vec, np.ndarray)
        assert spc_molar_cc_vec.size == len(self.species)
        assert np.all(spc_molar_cc_vec >= 0), 'spc_molar_cc_vec =\n%r' % spc_molar_cc_vec

        # --------------------------------
        # Partial r_vec partial kf matrix
        # --------------------------------
        if kf_vec is not None:

            if alpha_lst is None:
                (alpha_lst_local, _)=self.__get_power_law_exponents()
            else:
                alpha_lst_local=alpha_lst

            dr_dk_f = np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)
            i_mtrx= np.eye(len(self.reactions), dtype = np.float64)

            for jdx in range(len(self.data)):

                r_vec_h = self.r_vec(spc_molar_cc_vec, kf_vec = kf_vec + h_small*i_mtrx[:,jdx], alpha_lst =alpha_lst_local)
                r_vec = self.r_vec(spc_molar_cc_vec, kf_vec = kf_vec, alpha_lst =alpha_lst_local)

                dr_dk_f[:, jdx]=(r_vec_h - r_vec) / h_small

            try:
                dr_dtheta_mtrx=np.hstack([dr_dtheta_mtrx, dr_dk_f])
            except NameError:
                dr_dtheta_mtrx=np.hstack([dr_dk_f])

        # --------------------------------
        # Partial r_vec partial kb matrix
        # --------------------------------
        if kb_vec is not None:

            if beta_lst is None:
                (_, beta_lst_local)=self.__get_power_law_exponents()
            else:
                beta_lst_local=beta_lst

            dr_dk_b = np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)
            i_mtrx= np.eye(len(self.reactions), dtype = np.float64)

            for jdx in range(len(self.data)):

                r_vec_h = self.r_vec(spc_molar_cc_vec, kb_vec = kb_vec + h_small*i_mtrx[:,jdx], beta_lst =beta_lst_local)
                r_vec = self.r_vec(spc_molar_cc_vec, kb_vec = kb_vec, beta_lst =beta_lst_local)

                dr_dk_b[:, jdx]=(r_vec_h - r_vec) / h_small

            try:
                dr_dtheta_mtrx=np.hstack([dr_dtheta_mtrx, dr_dk_b])
            except NameError:
                dr_dtheta_mtrx=np.hstack([dr_dk_b])

        # -----------------------------------
        # Partial r_vec partial alpha matrix
        # -----------------------------------
        if alpha_lst is not None:

            n_alphas=0
            for alpha_mtrx in alpha_lst:
                n_alphas += alpha_mtrx.shape[1]

            dr_dalpha= np.zeros((len(self.reactions), n_alphas), dtype = np.float64)
            i_mtrx= np.eye(n_alphas, dtype = np.float64)

            if kf_vec is None:
                (kf_vec_local, _)=self.__get_ks()
            else:
                kf_vec_local=kf_vec

            for jdx in range(n_alphas):

                assert False, 'FIXME'

                r_vec_h = self.r_vec(spc_molar_cc_vec, alpha_lst = alpha_lst + h_small*i_mtrx[:,jdx], kf_vec =kf_vec_local)
                r_vec = self.r_vec(spc_molar_cc_vec, alpha_lst = alpha_lst, kf_vec =kf_vec_local)

                dr_dalpha[:, jdx]=(r_vec_h - r_vec) / h_small

            try:
                dr_dtheta_mtrx=np.hstack([dr_dtheta_mtrx, dr_dalpha])
            except NameError:
                dr_dtheta_mtrx=np.hstack([dr_dalpha])

        # ----------------------------------
        # Partial r_vec partial beta matrix
        # ----------------------------------
        if beta_lst is not None:

            n_betas=0
            for beta_vec in beta_lst:
                # assert np.all(beta_vec>=0)
                n_betas += beta_vec.size

            dr_dbeta = np.zeros((len(self.reactions), n_betas), dtype = np.float64)
            i_mtrx= np.eye(n_betas, dtype = np.float64)

            if kb_vec is None:
                (_, kb_vec_local)=self.__get_ks()
            else:
                kb_vec_local=kb_vec

            for jdx in range(n_betas):

                assert False, 'FIXME'

                r_vec_h = self.r_vec(spc_molar_cc_vec, beta_lst = beta_lst + h_small*i_mtrx[:,jdx], kb_vec =kb_vec_local)
                r_vec = self.r_vec(spc_molar_cc_vec, beta_lst = beta_lst, kb_vec =kb_vec_local)

                dr_dbeta[:, jdx]=(r_vec_h - r_vec) / h_small

            try:
                dr_dtheta_mtrx=np.hstack([dr_dtheta_mtrx, dr_dbeta])
            except NameError:
                dr_dtheta_mtrx=np.hstack([dr_dbeta])

        return dr_dtheta_mtrx

    def dg_dtheta_mtrx(self, spc_molar_cc_vec,
                       theta_kf_vec=None, theta_kb_vec=None, theta_alpha_lst=None, theta_beta_lst=None):
        '''Compute the partial derivative of the reaction rate density vector wrt to operating parameters.

        Compute dg_dtheta with theta being the operating parameters.
        This quantity is typically the negative of the Jacobian matrix in the leasts-squares optimization
        of parameters.

        Parameters:
        -----------
        '''

        dg_dtheta_mtrx = self.stoic_mtrx.transpose() @ self.dr_dtheta_mtrx(spc_molar_cc_vec,
                                                                           theta_kf_vec, theta_kb_vec,
                                                                           theta_alpha_lst, theta_beta_lst)

        return dg_dtheta_mtrx

    def d2ri_theta2_mtrx(self, rxn_idx, spc_molar_cc_vec,
                               kf_vec = None, kb_vec =None, alpha_lst=None, beta_lst=None):
        '''Second partial derivatives of the ith reaction rate law wrt parameters.

        Only the forward case reaction case is implemented.

        The parameters in the derivative are ordered as: k_fs, k_bs, alphas, betas.
        The matrix is p x p. Where p is the total number of parameters.
        That, is p = 2 * m + n_Ri + n_Pi, where n_Ri is the number of active reactant species, and
        n_Pi is the number of active product species.
        '''

        assert isinstance(rxn_idx, int)
        assert rxn_idx <= len(self.reactions)

        assert isinstance(spc_molar_cc_vec, np.ndarray)
        assert spc_molar_cc_vec.size == len(self.species)
        assert np.all(spc_molar_cc_vec >= 0), 'spc_molar_cc_vec =\n%r' % spc_molar_cc_vec

        # *******************************************************************************************
        # 1st row block

        # ---------------------------
        # partial_kf(partial_kf r_i)
        # ---------------------------
        if kf_vec is not None:

            d_kf_d_kf_ri_mtrx= np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)

            theta_vec = copy.deepcopy(kf_vec)

            d2kf_dtheta2_vec = self.__d2phi_dtheta2(theta_vec, self.kf_bnds)

            if alpha_lst is None:
                (alpha_lst_local, _)=self.__get_power_law_exponents()
            else:
                alpha_lst_local= copy.deepcopy(alpha_lst)

            alpha_lst_local = self.perform_reparam(alpha_lst_local, self.alpha_bnds)


            alpha_mtrx=alpha_lst_local[rxn_idx]

            reactants_ids=alpha_mtrx[0, :].astype(int)

            reactants_molar_cc=spc_molar_cc_vec[reactants_ids]

            spc_cc_power_prod=np.prod(reactants_molar_cc**alpha_mtrx[1, :])

            d_kf_d_kf_ri_mtrx[rxn_idx,rxn_idx]= d2kf_dtheta2_vec[rxn_idx] * spc_cc_power_prod

        # ---------------------------
        # partial_kb(partial_kf r_i)
        # ---------------------------
        if kf_vec is not None and kb_vec is not None:

            d_kb_d_kf_ri_mtrx= np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)

        # ------------------------------
        # partial_alpha(partial_kf r_i)
        # ------------------------------
        if kf_vec is not None and alpha_lst is not None:


            n_alphas=0
            for alpha_mtrx in alpha_lst:
                n_alphas += alpha_mtrx.shape[1]

            d_alpha_d_kf_ri_mtrx= np.zeros((len(self.reactions), n_alphas), dtype = np.float64)

            theta_lst = copy.deepcopy(alpha_lst) #alpha's

            dalpha_dtheta_lst = self.__dphi_dtheta(theta_lst, self.alpha_bnds)

            theta_vec = copy.deepcopy(kf_vec) #kf's

            dkf_dtheta_vec = self.__dphi_dtheta(theta_vec, self.kf_bnds)


            alpha_lst_local = copy.deepcopy(alpha_lst)
            alpha_lst_local = self.perform_reparam(alpha_lst_local, self.alpha_bnds)

            #kf_vec_local = copy.deepcopy(kf_vec)
            #kf_vec = self.reparam(kf_vec, self.kf_bnds)

            jdx_start=0

            for idx in range(rxn_idx):

                alpha_mtrx=alpha_lst_local[idx]
                jdx_start += alpha_mtrx.shape[1]


            rxn_idx_alpha_mtrx = alpha_lst_local[rxn_idx]
            rxn_idx_dalpha_dtheta_mtrx = dalpha_dtheta_lst[rxn_idx]
            #assert(rxn_idx_dalpha_dtheta_mtrx[0,:] == rxn_idx_alpha_mtrx[0,:]) # reactant IDs must match

            reactants_ids=rxn_idx_alpha_mtrx[0, :].astype(int)

            reactants_molar_cc=spc_molar_cc_vec[reactants_ids]

            spc_cc_power_prod=np.prod(reactants_molar_cc**rxn_idx_alpha_mtrx[1, :])

            min_c_j=reactants_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, )=np.where(reactants_molar_cc == min_c_j)
                reactants_molar_cc[jdx]=1.0  # any non-zero value will do since rb_i will be zero



            for jdx in range(rxn_idx_alpha_mtrx[1,:].size):

                c_j = reactants_molar_cc[jdx]

                d_alpha_d_kf_ri_mtrx[rxn_idx, jdx_start + jdx] = spc_cc_power_prod * math.log(c_j) * rxn_idx_dalpha_dtheta_mtrx[1, jdx] * dkf_dtheta_vec [rxn_idx]


            '''
            for (jdx, c_j) in enumerate(reactants_molar_cc):
                d_alpha_d_kf_ri_mtrx[idx, jdx_start + jdx] = spc_cc_power_prod * math.log(c_j)

                for (jdx, c_j) in enumerate(reactants_molar_cc):

                    d_alpha_d_kf_ri_mtrx[idx, jdx_start + jdx] = dkf_dtheta_vec[idx] * math.log(c_j) * dalpha_dtheta_mtrx[1, jdx] * spc_cc_power_prod

                jdx_start += alpha_mtrx.shape[1]
            '''
            #assert jdx_start == n_alphas, 'n_alphas = %r; sum = %r' % (n_alphas, jdx_start)

        # ------------------------------
        # partial_beta(partial_kf r_i)
        # ------------------------------
        if kf_vec is not None and beta_lst is not None:

            n_betas=0
            for beta_mtrx in beta_lst:
                n_betas += beta_mtrx.shape[1]

            d_beta_d_kf_ri_mtrx= np.zeros((len(self.reactions), n_betas), dtype = np.float64)

        # *******************************************************************************************
        # 2nd row block

        # ---------------------------
        # partial_kb(partial_kb r_i)
        # ---------------------------
        if kb_vec is not None:

            d_kb_d_kb_ri_mtrx= np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)

            theta_vec = copy.deepcopy(kb_vec)

            d2kb_dtheta2_vec = self.__d2phi_dtheta2(theta_vec, self.kb_bnds)


            if beta_lst is None:
                    (_, beta_lst_local)=self.__get_power_law_exponents()
            else:
                    beta_lst_local= copy.deepcopy(beta_lst)

            beta_lst_local = self.perform_reparam(beta_lst_local, self.beta_bnds)



            beta_mtrx=beta_lst_local[rxn_idx]
            products_ids=beta_mtrx[0, :].astype(int)

            products_molar_cc=spc_molar_cc_vec[products_ids]

            spc_cc_power_prod=np.prod(products_molar_cc**beta_mtrx[1, :])

            d_kb_d_kb_ri_mtrx[rxn_idx, rxn_idx]=- \
                        d2kb_dtheta2_vec[rxn_idx] * spc_cc_power_prod

        # ------------------------------
        # partial_alpha(partial_kb r_i)
        # ------------------------------
        if kb_vec is not None and alpha_lst is not None:

            n_alphas=0
            for alpha_mtrx in alpha_lst:
                n_alphas += alpha_mtrx.shape[1]

            d_alpha_d_kb_ri_mtrx= np.zeros((len(self.reactions), n_alphas), dtype = np.float64)

        # ------------------------------
        # partial_beta(partial_kb r_i)
        # ------------------------------
        if kb_vec is not None and beta_lst is not None:

            n_betas=0
            for beta_mtrx in beta_lst:
                n_betas += beta_mtrx.shape[1]

            d_beta_d_kb_ri_mtrx= np.zeros((len(self.reactions), n_betas), dtype = np.float64)

            theta_lst = copy.deepcopy(beta_lst)
            theta_vec = copy.deepcopy(kb_vec)

            dbeta_dtheta_lst = self.__dphi_dtheta(theta_lst, self.beta_bnds)
            dkb_dtheta_vec = self.__dphi_dtheta(kb_vec, self.kb_bnds)


            beta_lst_local=copy.deepcopy(beta_lst)
            beta_lst_local = self.perform_reparam(beta_lst_local, self.beta_bnds)

            #kb_vec= self.perform_reparam(kb_vec, self.kb_bnds)

            jdx_start=0

            for idx in range(rxn_idx):

                beta_mtrx = beta_lst_local[idx]
                jdx_start += beta_mtrx.shape[1]

            rxn_idx_beta_mtrx = beta_lst_local[rxn_idx]
            rxn_idx_dbeta_dtheta_mtrx = dbeta_dtheta_lst[rxn_idx]

            products_ids=rxn_idx_beta_mtrx[0, :].astype(int)

            products_molar_cc=spc_molar_cc_vec[products_ids]


            spc_cc_power_prod= - np.prod(products_molar_cc**rxn_idx_beta_mtrx[1, :])

            min_c_j=products_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, )=np.where(products_molar_cc == min_c_j)
                products_molar_cc[jdx]=1.0  # any non-zero value will do since rb_i will be zero

            for jdx in range(rxn_idx_beta_mtrx[1,:].size):


                c_j = products_molar_cc[jdx]
                d_beta_d_kb_ri_mtrx[rxn_idx, jdx_start + jdx] = spc_cc_power_prod * math.log(c_j) * rxn_idx_dbeta_dtheta_mtrx[1, jdx] * dkb_dtheta_vec[rxn_idx]

            #assert jdx_start == n_betas, 'n_betas = %r; sum = %r' % (
            #    n_betas, jdx_start)

        # *******************************************************************************************
        # 3rd row block

        # ---------------------------------
        # partial_alpha(partial_alpha r_i)
        # ---------------------------------
        if alpha_lst is not None:

            n_alphas=0
            for alpha_mtrx in alpha_lst:
                n_alphas += alpha_mtrx.shape[1]

            d_alpha_d_alpha_ri_mtrx= np.zeros((n_alphas, n_alphas), dtype = np.float64)

            theta_lst = copy.deepcopy(alpha_lst)

            d2alpha_dtheta2_lst = self.__d2phi_dtheta2(theta_lst, self.alpha_bnds)

            theta_lst = copy.deepcopy(alpha_lst)

            dalpha_dtheta_lst = self.__dphi_dtheta(theta_lst, self.alpha_bnds)

            alpha_lst_local=copy.deepcopy(alpha_lst)

            if kf_vec is None:
                kf_vec_local = self.__get_kf()
            else:
                kf_vec_local= copy.deepcopy(kf_vec)

            kf_vec_local = self.perform_reparam(kf_vec_local, self.kf_bnds)

            jdx_start=0
            idx_start=0

            for idx in range(rxn_idx):

                alpha_mtrx=alpha_lst_local[idx]
                jdx_start += alpha_mtrx.shape[1]

            alpha_mtrx = alpha_lst_local[rxn_idx]

            reactants_ids=alpha_mtrx[0, :].astype(int)

            reactants_molar_cc=spc_molar_cc_vec[reactants_ids]

            spc_cc_power_prod=np.prod(reactants_molar_cc**alpha_mtrx[1, :])

            rf_i=kf_vec_local[rxn_idx] * spc_cc_power_prod

            min_c_j=reactants_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, )=np.where(reactants_molar_cc == min_c_j)
                reactants_molar_cc[jdx]=1.0  # any non-zero value will do since rb_i will be zero

            dalpha_dtheta_mtrx = dalpha_dtheta_lst[rxn_idx]
            d2alpha_dtheta2_mtrx = d2alpha_dtheta2_lst[rxn_idx]

            for jdx in range(alpha_mtrx[1,:].size):

                c_j = reactants_molar_cc[jdx]

                dalpha_j_dtheta_j = dalpha_dtheta_mtrx[1 , jdx]
                d2alpha_dtheta2 = d2alpha_dtheta2_mtrx[1 , jdx]

                for Jdx in range(alpha_mtrx[1,:].size):


                    dalpha_J_dtheta_J = dalpha_dtheta_mtrx[1 , Jdx]
                    c_J = reactants_molar_cc[Jdx]

                    d2r_dalpha2 = rf_i * math.log(c_j) * math.log(c_J)
                    dr_dtheta_prod = dalpha_j_dtheta_j * dalpha_J_dtheta_J

                    if jdx == Jdx:

                        d_alpha_d_alpha_ri_mtrx[jdx_start + jdx, jdx_start + Jdx] = d2r_dalpha2 * dr_dtheta_prod + rf_i * math.log(c_j) * d2alpha_dtheta2

                    else:

                        d_alpha_d_alpha_ri_mtrx[jdx_start + jdx, jdx_start + Jdx] = d2r_dalpha2 * dr_dtheta_prod

        # --------------------------------
        # partial_beta(partial_alpha r_i)
        # --------------------------------
        if alpha_lst is not None and beta_lst is not None:

            n_alphas=0
            for alpha_mtrx in alpha_lst:
                n_alphas += alpha_mtrx.shape[1]

            n_betas=0
            for beta_mtrx in beta_lst:
                n_betas += beta_mtrx.shape[1]

            d_beta_d_alpha_ri_mtrx= np.zeros((n_alphas, n_betas), dtype = np.float64)

        # *******************************************************************************************
        # 4th row block

        # ---------------------------------
        # partial_beta(partial_beta r_i)
        # ---------------------------------
        if beta_lst is not None:

            n_betas=0
            for beta_mtrx in beta_lst:
                n_betas += beta_mtrx.shape[1]

            d_beta_d_beta_ri_mtrx = np.zeros((n_betas, n_betas), dtype = np.float64)

            theta_lst = copy.deepcopy(beta_lst)

            d2beta_dtheta2_lst = self.__d2phi_dtheta2(theta_lst, self.beta_bnds)

            theta_lst = copy.deepcopy(beta_lst)

            dbeta_dtheta_lst = self.__dphi_dtheta(theta_lst, self.beta_bnds)

            beta_lst_local = copy.deepcopy(beta_lst)
            beta_lst_local = self.perform_reparam(beta_lst_local, self.beta_bnds)

            if kb_vec is None:
                kb_vec_local = self.__get_kb()
            else:
                kb_vec_local = copy.deepcopy(kb_vec)

            kb_vec_local = self.perform_reparam(kb_vec_local, self.kb_bnds)

            jdx_start=0
            idx_start=0

            for idx in range(rxn_idx):
                beta_mtrx = beta_lst_local[idx]
                jdx_start += beta_mtrx.shape[1]

            beta_mtrx=beta_lst_local[rxn_idx]

            products_ids=beta_mtrx[0, :].astype(int)

            products_molar_cc=spc_molar_cc_vec[products_ids]

            spc_cc_power_prod= np.prod(products_molar_cc**beta_mtrx[1, :])

            rb_i=- kb_vec_local[rxn_idx] * spc_cc_power_prod

            min_c_j=products_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, )=np.where(products_molar_cc == min_c_j)
                products_molar_cc[jdx]=1.0  # any non-zero value will do since rb_i will be zero

            dbeta_dtheta_mtrx = dbeta_dtheta_lst[rxn_idx]
            d2beta_dtheta2_mtrx = d2beta_dtheta2_lst[rxn_idx]

            for jdx in range(beta_mtrx[1,:].size):

                c_j = reactants_molar_cc[jdx]

                dbeta_j_dtheta_j = dbeta_dtheta_mtrx[1 , jdx]
                d2beta_dtheta2 = d2beta_dtheta2_mtrx[1 , jdx]

                for Jdx in range(beta_mtrx[1,:].size):

                    dbeta_J_dtheta_J = dbeta_dtheta_mtrx[1 , Jdx]
                    c_J = reactants_molar_cc[Jdx]

                    d2r_dbeta2 = rb_i * math.log(c_j) * math.log(c_J)
                    dr_dtheta_prod = dbeta_j_dtheta_j * dbeta_J_dtheta_J

                    if jdx == Jdx:

                        d_beta_d_beta_ri_mtrx[jdx_start + jdx, jdx_start + Jdx] = d2r_dbeta2 * dr_dtheta_prod + rb_i * math.log(c_j) * d2beta_dtheta2

                    else:

                        d_beta_d_beta_ri_mtrx[jdx_start + jdx, jdx_start + Jdx] = d2r_dbeta2 * dr_dtheta_prod

        # *******************************************************************************************
        # Assembly

        # General case
        if kf_vec is not None and kb_vec is not None and alpha_lst is not None and beta_lst is not None:

            hessian_ri_1st_row=np.hstack(
                [d_kf_d_kf_ri_mtrx, d_kb_d_kf_ri_mtrx, d_alpha_d_kf_ri_mtrx, d_beta_d_kf_ri_mtrx])

            hessian_ri_2nd_row=np.hstack([d_kb_d_kf_ri_mtrx.transpose(
            ), d_kb_d_kb_ri_mtrx, d_alpha_d_kb_ri_mtrx, d_beta_d_kb_ri_mtrx])

            hessian_ri_3rd_row=np.hstack([d_alpha_d_kf_ri_mtrx.transpose(
            ), d_alpha_d_kb_ri_mtrx.transpose(), d_alpha_d_alpha_ri_mtrx, d_beta_d_alpha_ri_mtrx])

            hessian_ri_4th_row=np.hstack([d_beta_d_kf_ri_mtrx.transpose(), d_beta_d_kb_ri_mtrx.transpose(
            ), d_beta_d_alpha_ri_mtrx.transpose(), d_beta_d_beta_ri_mtrx])

            hessian_ri=np.vstack(
                [hessian_ri_1st_row, hessian_ri_2nd_row, hessian_ri_3rd_row, hessian_ri_4th_row])

        # Forward case
        elif kf_vec is not None and alpha_lst is not None and kb_vec is None and beta_lst is None:

            hessian_ri_1st_row=np.hstack(
                [d_kf_d_kf_ri_mtrx, d_alpha_d_kf_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_alpha_d_kf_ri_mtrx.transpose(), d_alpha_d_alpha_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row])

        # k's only case
        elif kf_vec is not None and kb_vec is not None and alpha_lst is None and beta_lst is None:

            hessian_ri_1st_row=np.hstack(
                [d_kf_d_kf_ri_mtrx, d_kb_d_kf_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_kb_d_kf_ri_mtrx.transpose(), d_kb_d_kb_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row])

        # kfs only case
        elif kf_vec is not None and kb_vec is None and alpha_lst is None and beta_lst is None:

            hessian_ri=d_kf_d_kf_ri_mtrx

        # k's and alphas only case
        elif kf_vec is not None and kb_vec is not None and alpha_lst is not None and beta_lst is None:

            hessian_ri_1st_row=np.hstack(
                [d_kf_d_kf_ri_mtrx, d_kb_d_kf_ri_mtrx, d_alpha_d_kf_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_kb_d_kf_ri_mtrx.transpose(), d_kb_d_kb_ri_mtrx, d_alpha_d_kb_ri_mtrx])
            hessian_ri_3rd_row=np.hstack([d_alpha_d_kf_ri_mtrx.transpose(
            ), d_alpha_d_kb_ri_mtrx.transpose(), d_alpha_d_alpha_ri_mtrx])

            hessian_ri=np.vstack(
                [hessian_ri_1st_row, hessian_ri_2nd_row, hessian_ri_3rd_row])

        # alphas only case
        elif kf_vec is None and kb_vec is None and alpha_lst is not None and beta_lst is None:

            hessian_ri=d_alpha_d_alpha_ri_mtrx

        # betas only case
        elif kf_vec is None and kb_vec is None and alpha_lst is None and beta_lst is not None:

            hessian_ri=d_beta_d_beta_ri_mtrx

        # alphas and betas only case
        elif kf_vec is None and kb_vec is None and alpha_lst is not None and beta_lst is not None:

            hessian_ri_1st_row=np.hstack(
                [d_alpha_d_alpha_ri_mtrx, d_beta_d_alpha_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_beta_d_alpha_ri_mtrx.transpose(), d_beta_d_beta_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row])

        # kfs and betas only case
        elif kf_vec is not None and kb_vec is None and alpha_lst is None and beta_lst is not None:

            hessian_ri_1st_row=np.hstack(
                [d_kf_d_kf_ri_mtrx, d_beta_d_kf_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_beta_d_kf_ri_mtrx.transpose(), d_beta_d_beta_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row])

        # kbs and alphas only case
        elif kf_vec is None and kb_vec is not None and alpha_lst is not None and beta_lst is None:

            hessian_ri_1st_row=np.hstack(
                [d_kb_d_kb_ri_mtrx, d_alpha_d_kb_ri_mtrx])
            hessian_ri_2nd_row=np.hstack(
                [d_alpha_d_kb_ri_mtrx.transpose(), d_alpha_d_alpha_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row])

        else:
            assert False, 'Hessian ri case not implemented.'

        return hessian_ri

    def d2ri_theta2_mtrx_new(self, rxn_idx, spc_molar_cc_vec,
                             theta_kf_vec = None, theta_kb_vec =None,
                             theta_alpha_lst=None, theta_beta_lst=None):
        '''Second partial derivatives of the ith reaction rate law wrt parameters.

        The parameters in the derivative are ordered as: k_fs, k_bs, alphas, betas.
        The matrix is p x p. Where p is the total number of parameters.
        That, is p = 2 * m + n_alpha + n_beta, where n_alpha is the number of active forward reaction
        species, and n_beta is the number of active reverse reaction species.
        '''

        assert isinstance(rxn_idx, int)
        assert 0 <= rxn_idx <= len(self.reactions), 'rxn_idx = %r'%(rxn_idx)

        assert isinstance(spc_molar_cc_vec, np.ndarray)
        assert spc_molar_cc_vec.size == len(self.species)
        assert np.all(spc_molar_cc_vec >= 0), 'spc_molar_cc_vec =\n%r' % spc_molar_cc_vec

        # *******************************************************************************************
        # 1st row block  partial_kf(r_i)

        # ----------------------------------------
        # partial_theta_kf(partial_theta_kf r_i) =
        # D2_theta_kf2(kf) diag(pi)
        # ----------------------------------------
        if theta_kf_vec is not None:

            # Compute D2_theta_kf2(kf)

            theta_kf_vec = copy.deepcopy(theta_kf_vec)

            #print('theta_kf_vec=',theta_kf_vec)
            #print('kf_bnds=',self.kf_bnds)

            d2kf_dtheta_kf2_vec = self.__d2phi_dtheta2(theta_kf_vec, self.kf_bnds)

            d2kf_dtheta_kf2_mtrx = np.diag(d2kf_dtheta_kf2_vec)

            # Compute diag(pi)

            if theta_alpha_lst is None:
                alpha_lst = self.__get_alpha()
            else:
                theta_alpha_lst = copy.deepcopy(theta_alpha_lst)
                alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)

            alpha_data_mtrx = alpha_lst[rxn_idx]
            active_spc_ids = alpha_data_mtrx[0, :].astype(int)
            alpha_i_vec = alpha_data_mtrx[1, :]
            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]
            spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)
            p_i = spc_cc_power_prod

            diag_pi = np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)
            diag_pi[rxn_idx, rxn_idx] = p_i

            # Compute product

            #print('\t d2kkf_dtheta_kf2_mtrx=\n','\t',d2kf_dtheta_kf2_mtrx)
            #print('\t diag_pi=\n','\t',diag_pi)

            d_theta_kf_d_theta_kf_ri_mtrx = d2kf_dtheta_kf2_mtrx @ diag_pi

        # --------------------------------------
        # partial_theta_kb(partial_theta_kf r_i)
        # --------------------------------------
        if theta_kf_vec is not None and theta_kb_vec is not None:

            d_theta_kb_d_theta_kf_ri_mtrx = np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)

        # -----------------------------------------
        # partial_theta_alpha(partial_theta_kf r_i) =
        # partial_theta_kf(kf) P W_alpha_i partial_theta_alpha(alpha)
        # -----------------------------------------
        if theta_kf_vec is not None and theta_alpha_lst is not None:

            # Compute partial_theta_kf(kf)

            theta_kf_vec = copy.deepcopy(theta_kf_vec)

            dkf_dtheta_vec = self.__dphi_dtheta(theta_kf_vec, self.kf_bnds)

            dkf_dtheta_mtrx = np.diag(dkf_dtheta_vec)

            # Compute P

            theta_alpha_lst = copy.deepcopy(theta_alpha_lst)

            alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)

            p_vec = np.zeros(len(self.reactions), dtype = np.float64)

            for (irxn, alpha_data_mtrx) in enumerate(alpha_lst):

                active_spc_ids = alpha_data_mtrx[0, :].astype(int)

                alpha_i_vec = alpha_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

                p_vec[irxn] = spc_cc_power_prod

            p_mtrx = np.diag(p_vec)

            # Compute W_alpha_i where i is rxn_idx

            n_alphas=0
            for alpha_data_mtrx in alpha_lst:
                n_alphas += alpha_data_mtrx.shape[1]

            alpha_data_i_mtrx = alpha_lst[rxn_idx]

            active_spc_ids = alpha_data_i_mtrx[0, :].astype(int)

            alpha_i_vec = alpha_data_i_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

            min_c_j = active_spc_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                active_spc_molar_cc[jdx] = 1.0 # any non-zero value will do since rf_i will be zero

            w_alpha_i_vec = np.log(active_spc_molar_cc)

            w_alpha_i_mtrx = np.zeros((len(self.reactions), n_alphas), dtype = np.float64)

            w_alpha_i_column_id = \
                    sum([alpha_data_mtrx.shape[1] for alpha_data_mtrx in alpha_lst[:rxn_idx]])

            end_plus_one = w_alpha_i_column_id + w_alpha_i_vec.size # end + 1

            assert end_plus_one <= n_alphas

            # insertion
            #print('n_alphas ', n_alphas)
            #print('alpha_data_mtrx\n ', alpha_data_mtrx)
            #print('w_alpha_i_mtrx.shape ', w_alpha_i_mtrx.shape)
            #print('w_alpha_i_mtrx ', w_alpha_i_mtrx)
            #print('w_alpha_i_column_id ', w_alpha_i_column_id)
            #print('end ', end)

            w_alpha_i_mtrx[rxn_idx, w_alpha_i_column_id:end_plus_one] = w_alpha_i_vec[:]

            # Compute partial_theta_alpha(alpha)

            dalpha_dtheta_alpha_lst = self.__dphi_dtheta(theta_alpha_lst, self.alpha_bnds)

            for dalpha_dtheta_alpha_data_mtrx in dalpha_dtheta_alpha_lst:

                #assert(dalpha_dtheta_alpha_data_mtrx[0,:] == alpha_mtrx[0,:]) # reactant IDs must match
                dalpha_dtheta_alpha_mtrx_i = np.diag(dalpha_dtheta_alpha_data_mtrx[1,:])

                try:
                    dalpha_dtheta_alpha_mtrx = sp.linalg.block_diag(dalpha_dtheta_alpha_mtrx, dalpha_dtheta_alpha_mtrx_i)
                except NameError:
                    dalpha_dtheta_alpha_mtrx = sp.linalg.block_diag(dalpha_dtheta_alpha_mtrx_i)

            # Compute product

            d_theta_alpha_d_theta_kf_ri_mtrx = dkf_dtheta_mtrx @ p_mtrx @ w_alpha_i_mtrx @ dalpha_dtheta_alpha_mtrx

            del dalpha_dtheta_alpha_mtrx

            assert d_theta_alpha_d_theta_kf_ri_mtrx.shape == (len(self.reactions), n_alphas)

        # ----------------------------------------
        # partial_theta_beta(partial_theta_kf r_i)
        # ----------------------------------------
        if theta_kf_vec is not None and theta_beta_lst is not None:

            n_betas=0
            for theta_beta_data_mtrx in theta_beta_lst:
                n_betas += theta_beta_data_mtrx.shape[1]

            d_theta_beta_d_theta_kf_ri_mtrx = np.zeros((len(self.reactions), n_betas), dtype = np.float64)

        # *******************************************************************************************
        # 2nd row block  partial_kb(r_i)

        # --------------------------------------
        # partial_theta_kf(partial_theta_kb r_i)
        # --------------------------------------
        if theta_kb_vec is not None and theta_kf_vec is not None:

            d_theta_kf_d_theta_kb_ri_mtrx = d_theta_kb_d_theta_kf_ri_mtrx.transpose()

        # ----------------------------------------
        # partial_theta_kb(partial_theta_kb r_i) =
        # - D2_theta_kb2(kb) diag(qi)
        # ----------------------------------------
        if theta_kb_vec is not None:

            # Compute D2_theta_kb2(kb)

            theta_kb_vec = copy.deepcopy(theta_kb_vec)

            d2kb_dtheta_kb2_vec = self.__d2phi_dtheta2(theta_kb_vec, self.kb_bnds)

            d2kb_dtheta_kb2_mtrx = np.diag(d2kb_dtheta_kb2_vec)

            # Compute diag(qi)

            if theta_beta_lst is None:
                beta_lst = self.__get_beta()
            else:
                theta_beta_lst = copy.deepcopy(theta_beta_lst)
                beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)

            beta_data_mtrx = beta_lst[rxn_idx]
            active_spc_ids = beta_data_mtrx[0, :].astype(int)
            beta_i_vec = beta_data_mtrx[1, :]
            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]
            spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)
            q_i = spc_cc_power_prod

            diag_qi = np.zeros((len(self.reactions), len(self.reactions)), dtype = np.float64)
            diag_qi[rxn_idx, rxn_idx] = q_i

            # Compute product

            d_theta_kb_d_theta_kb_ri_mtrx = - d2kb_dtheta_kb2_mtrx @ diag_qi

        # -----------------------------------------
        # partial_theta_alpha(partial_theat_kb r_i)
        # -----------------------------------------
        if theta_kb_vec is not None and theta_alpha_lst is not None:

            n_alphas=0
            for theta_alpha_data_mtrx in theta_alpha_lst:
                n_alphas += theta_alpha_data_mtrx.shape[1]

            d_theta_alpha_d_theta_kb_ri_mtrx= np.zeros((len(self.reactions), n_alphas), dtype = np.float64)

        # ----------------------------------------
        # partial_theta_beta(partial_theta_kb r_i) =
        # - partial_theta_kb(kb) Q W_beta_i partial_theta_beta(beta)
        # ----------------------------------------
        if theta_kb_vec is not None and theta_beta_lst is not None:

            # Compute partial_theta_kb(kb)

            theta_kb_vec = copy.deepcopy(theta_kb_vec)

            dkb_dtheta_vec = self.__dphi_dtheta(theta_kb_vec, self.kb_bnds)

            dkb_dtheta_mtrx = np.diag(dkb_dtheta_vec)

            # Compute Q

            theta_beta_lst = copy.deepcopy(theta_beta_lst)

            beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)

            q_vec = np.zeros(len(self.reactions), dtype = np.float64)

            for (irxn, beta_data_mtrx) in enumerate(beta_lst):

                active_spc_ids = beta_data_mtrx[0, :].astype(int)

                beta_i_vec = beta_data_mtrx[1, :]

                active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

                spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

                q_vec[irxn] = spc_cc_power_prod

            q_mtrx = np.diag(q_vec)

            # Compute W_beta_i where i is rxn_idx

            n_betas=0
            for beta_data_mtrx in beta_lst:
                n_betas += beta_data_mtrx.shape[1]

            beta_data_i_mtrx = beta_lst[rxn_idx]

            active_spc_ids = beta_data_i_mtrx[0, :].astype(int)

            beta_i_vec = beta_data_i_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

            min_c_j = active_spc_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                active_spc_molar_cc[jdx] = 1.0 # any non-zero value will do since rf_i will be zero

            w_beta_i_vec = np.log(active_spc_molar_cc)

            w_beta_i_mtrx = np.zeros((len(self.reactions), n_betas), dtype = np.float64)

            w_beta_i_column_id = \
                    sum([beta_data_mtrx.shape[1] for beta_data_mtrx in beta_lst[:rxn_idx]])

            end_plus_one = w_beta_i_column_id + w_beta_i_vec.size  # end + 1

            assert end_plus_one <= n_betas

            # insertion
            w_beta_i_mtrx[rxn_idx, w_beta_i_column_id:end_plus_one] = w_beta_i_vec[:]

            # Compute partial_theta_beta(beta)

            dbeta_dtheta_beta_lst = self.__dphi_dtheta(theta_beta_lst, self.beta_bnds)

            for dbeta_dtheta_beta_data_mtrx in dbeta_dtheta_beta_lst:

                #assert(dbeta_dtheta_beta_data_mtrx[0,:] == beta_mtrx[0,:]) # reactant IDs must match
                dbeta_dtheta_beta_mtrx_i = np.diag(dbeta_dtheta_beta_data_mtrx[1,:])

                try:
                    dbeta_dtheta_beta_mtrx = sp.linalg.block_diag(dbeta_dtheta_beta_mtrx, dbeta_dtheta_beta_mtrx_i)
                except NameError:
                    dbeta_dtheta_beta_mtrx = sp.linalg.block_diag(dbeta_dtheta_beta_mtrx_i)

            # Compute product

            d_theta_beta_d_theta_kb_ri_mtrx = - dkb_dtheta_mtrx @ q_mtrx @ w_beta_i_mtrx @ dbeta_dtheta_beta_mtrx

            del dbeta_dtheta_beta_mtrx

            assert d_theta_beta_d_theta_kf_ri_mtrx.shape == (len(self.reactions), n_betas)

        # *******************************************************************************************
        # 3rd row block

        # -----------------------------------------
        # partial_theta_kf(partial_theta_alpha r_i)
        # -----------------------------------------
        if theta_alpha_lst is not None and theta_kf_vec is not None:

            d_theta_kf_d_theta_alpha_ri_mtrx = d_theta_alpha_d_theta_kf_ri_mtrx.transpose()

        # -----------------------------------------
        # partial_theta_kb(partial_theta_alpha r_i)
        # -----------------------------------------
        if theta_alpha_lst is not None and theta_kb_vec is not None:

            d_theta_kb_d_theta_alpha_ri_mtrx = d_theta_alpha_d_theta_kb_ri_mtrx.transpose()

        # --------------------------------------------
        # partial_theta_alpha(partial_theta_alpha r_i) =
        # rfi ((partial_theta_alpha(alpha))^2  W_alpha_iT W_alpha_i + D2_theta_alpha2(alpha) Diag(w_alpha_i)
        # --------------------------------------------
        if theta_alpha_lst is not None:

            # Compute rf_i where i is rxn_idx

            # get kf
            if theta_kf_vec is None:
                kf_vec = self.__get_kf()
            else:
                theta_kf_vec = copy.deepcopy(theta_kf_vec)
                kf_vec = self.perform_reparam(theta_kf_vec, self.kf_bnds)

            theta_alpha_lst = copy.deepcopy(theta_alpha_lst)

            # get alphas
            alpha_lst = self.perform_reparam(theta_alpha_lst, self.alpha_bnds)

            alpha_data_mtrx = alpha_lst[rxn_idx]

            active_spc_ids = alpha_data_mtrx[0, :].astype(int)

            alpha_i_vec = alpha_data_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

            rf_i = kf_vec[rxn_idx] * spc_cc_power_prod

            # Compute (partial_theta_alpha(alpha))^2

            dalpha_theta_alpha_lst = self.__dphi_dtheta(theta_alpha_lst, self.alpha_bnds)

            for dalpha_dtheta_alpha_data_mtrx in dalpha_theta_alpha_lst:

                #assert(dalpha_dtheta_alpha_data_mtrx[0,:] == alpha_data_mtrx[0,:]) # reactant IDs must match
                dalpha_dtheta_alpha_mtrx_i = np.diag(dalpha_dtheta_alpha_data_mtrx[1,:])

                try:
                    dalpha_dtheta_alpha_mtrx = sp.linalg.block_diag(dalpha_dtheta_alpha_mtrx, dalpha_dtheta_alpha_mtrx_i)
                except NameError:
                    dalpha_dtheta_alpha_mtrx = sp.linalg.block_diag(dalpha_dtheta_alpha_mtrx_i)

            dalpha_dtheta_alpha_mtrx_pwr2 = dalpha_dtheta_alpha_mtrx @ dalpha_dtheta_alpha_mtrx

            del dalpha_dtheta_alpha_mtrx

            # Compute W_alpha_iT W_alpha_i where i is rxn_idx

            # get W_alpha_i

            n_alphas=0
            for alpha_data_mtrx in alpha_lst:
                n_alphas += alpha_data_mtrx.shape[1]

            alpha_data_i_mtrx = alpha_lst[rxn_idx]

            active_spc_ids = alpha_data_i_mtrx[0, :].astype(int)

            alpha_i_vec = alpha_data_i_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**alpha_i_vec)

            min_c_j = active_spc_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                active_spc_molar_cc[jdx] = 1.0 # any non-zero value will do since rf_i will be zero

            w_alpha_i_vec = np.log(active_spc_molar_cc)

            w_alpha_i_mtrx = np.zeros((len(self.reactions), n_alphas), dtype = np.float64)

            w_alpha_i_column_id = \
                    sum([alpha_data_mtrx.shape[1] for alpha_data_mtrx in alpha_lst[:rxn_idx]])

            end_plus_one = w_alpha_i_column_id + w_alpha_i_vec.size # end + 1

            assert end_plus_one <= n_alphas

            # insertion
            w_alpha_i_mtrx[rxn_idx, w_alpha_i_column_id:end_plus_one] = w_alpha_i_vec[:]

            w_alpha_i_T_w_alpha_i_mtrx = w_alpha_i_mtrx.transpose() @ w_alpha_i_mtrx

            # Compute D2_theta_alpha2(alpha)

            d2alpha_dtheta2_lst = self.__d2phi_dtheta2(theta_alpha_lst, self.alpha_bnds)

            for d2alpha_dtheta2_data_mtrx in d2alpha_dtheta2_lst:

                d2alpha_dtheta2_mtrx_i = np.diag(d2alpha_dtheta2_data_mtrx[1,:])

                try:
                    d2alpha_dtheta2_mtrx = sp.linalg.block_diag(d2alpha_dtheta2_mtrx, d2alpha_dtheta2_mtrx_i)
                except NameError:
                    d2alpha_dtheta2_mtrx = sp.linalg.block_diag(d2alpha_dtheta2_mtrx_i)

            # Compute Diag(W_alpha_i_irow)

            w_alpha_i_irow = w_alpha_i_mtrx[rxn_idx]

            diag_w_alpha_i_irow = np.diag(w_alpha_i_irow)

            # Compute product

            #print('dalpha_dtheta_alpha_mtrx_pwr2 \n', dalpha_dtheta_alpha_mtrx_pwr2)
            #print('w_alpha_i_T_w_alpha_i_mtrx \n', w_alpha_i_T_w_alpha_i_mtrx)

            d_theta_alpha_d_theta_alpha_ri_mtrx = \
                    rf_i * (dalpha_dtheta_alpha_mtrx_pwr2 @ w_alpha_i_T_w_alpha_i_mtrx \
                            + \
                            d2alpha_dtheta2_mtrx @ diag_w_alpha_i_irow)

        # -------------------------------------------
        # partial_theta_beta(partial_theta_alpha r_i)
        # -------------------------------------------
        if theta_alpha_lst is not None and theta_beta_lst is not None:

            n_alphas = 0
            for theta_alpha_data_mtrx in theta_alpha_lst:
                n_alphas += theta_alpha_data_mtrx.shape[1]

            n_betas = 0
            for theta_beta_data_mtrx in theta_beta_lst:
                n_betas += theta_beta_data_mtrx.shape[1]

            d_theta_beta_d_theta_alpha_ri_mtrx = np.zeros((n_alphas, n_betas), dtype = np.float64)

        # *******************************************************************************************
        # 4th row block

        # ----------------------------------------
        # partial_theta_kf(partial_theta_beta r_i)
        # ----------------------------------------
        if theta_beta_lst is not None and theta_kf_vec is not None:

            d_theta_kf_d_theta_beta_ri_mtrx = d_theta_beta_d_theta_kf_ri_mtrx.transpose()

        # ----------------------------------------
        # partial_theta_kb(partial_theta_beta r_i)
        # ----------------------------------------
        if theta_beta_lst is not None and theta_kf_vec is not None:

            d_theta_kb_d_theta_beta_ri_mtrx = d_theta_beta_d_theta_kb_ri_mtrx.transpose()

        # -------------------------------------------
        # partial_theta_alpha(partial_theta_beta r_i)
        # -------------------------------------------
        if theta_beta_lst is not None and theta_alpha_lst is not None:

            d_theta_alpha_d_theta_beta_ri_mtrx = d_theta_beta_d_theta_alpha_ri_mtrx.transpose()

        # --------------------------------------------
        # partial_theta_beta(partial_theta_beta r_i) =
        # - rbi ((partial_theta_beta(beta))^2  W_beta_iT W_beta_i + D2_theta_beta2(beta) Diag(w_beta_i)
        # --------------------------------------------
        if theta_beta_lst is not None:

            # Compute rb_i where i is rxn_idx

            # get kb
            if theta_kb_vec is None:
                kb_vec = self.__get_kb()
            else:
                theta_kb_vec = copy.deepcopy(theta_kb_vec)
                kb_vec = self.perform_reparam(theta_kb_vec, self.kb_bnds)

            theta_beta_lst = copy.deepcopy(theta_beta_lst)

            # get betas
            beta_lst = self.perform_reparam(theta_beta_lst, self.beta_bnds)

            beta_data_mtrx = beta_lst[rxn_idx]

            active_spc_ids = beta_data_mtrx[0, :].astype(int)

            beta_i_vec = beta_data_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

            rb_i = kb_vec[rxn_idx] * spc_cc_power_prod

            # Compute (partial_theta_beta(beta))^2

            dbeta_dtheta_beta_lst = self.__dphi_dtheta(theta_beta_lst, self.beta_bnds)

            for dbeta_dtheta_beta_data_mtrx in dbeta_dtheta_beta_lst:

                #assert(dbeta_dtheta_beta_data_mtrx[0,:] == beta_data_mtrx[0,:]) # reactant IDs must match
                dbeta_dtheta_beta_mtrx_i = np.diag(dbeta_dtheta_beta_data_mtrx[1,:])

                try:
                    dbeta_dtheta_beta_mtrx = sp.linalg.block_diag(dbeta_dtheta_beta_mtrx, dbeta_dtheta_beta_mtrx_i)
                except NameError:
                    dbeta_dtheta_beta_mtrx = sp.linalg.block_diag(dbeta_dtheta_beta_mtrx_i)

            dbeta_dtheta_beta_mtrx_pwr2 = dbeta_dtheta_beta_mtrx @ dbeta_dtheta_beta_mtrx

            del dbeta_dtheta_beta_mtrx

            # Compute W_beta_iT W_beta_i where i is rxn_idx

            # get W_beta_i

            n_betas=0
            for beta_data_mtrx in beta_lst:
                n_betas += beta_data_mtrx.shape[1]

            beta_data_i_mtrx = beta_lst[rxn_idx]

            active_spc_ids = beta_data_i_mtrx[0, :].astype(int)

            beta_i_vec = beta_data_i_mtrx[1, :]

            active_spc_molar_cc = spc_molar_cc_vec[active_spc_ids]

            spc_cc_power_prod = np.prod(active_spc_molar_cc**beta_i_vec)

            min_c_j = active_spc_molar_cc.min()
            if min_c_j <= 1e-25:
                (jdx, ) = np.where(active_spc_molar_cc == min_c_j)
                active_spc_molar_cc[jdx] = 1.0 # any non-zero value will do since rb_i will be zero

            w_beta_i_vec = np.log(active_spc_molar_cc)

            w_beta_i_mtrx = np.zeros((len(self.reactions), n_betas), dtype = np.float64)

            w_beta_i_column_id = \
                    sum([beta_data_mtrx.shape[1] for beta_data_mtrx in beta_lst[:rxn_idx]])

            end_plus_one = w_beta_i_column_id + w_beta_i_vec.size # end + 1

            assert end_plus_one <= n_betas

            # insertion
            w_beta_i_mtrx[rxn_idx, w_beta_i_column_id:end_plus_one] = w_beta_i_vec[:]

            w_beta_i_T_w_beta_i_mtrx = w_beta_i_mtrx.transpose() @ w_beta_i_mtrx

            # Compute D2_theta_beta2(beta)

            d2beta_dtheta2_lst = self.__d2phi_dtheta2(theta_beta_lst, self.beta_bnds)

            for d2beta_dtheta2_data_mtrx in d2beta_dtheta2_lst:

                d2beta_dtheta2_mtrx_i = np.diag(d2beta_dtheta2_data_mtrx[1,:])

                try:
                    d2beta_dtheta2_mtrx = sp.linalg.block_diag(d2beta_dtheta2_mtrx, d2beta_dtheta2_mtrx_i)
                except NameError:
                    d2beta_dtheta2_mtrx = sp.linalg.block_diag(d2beta_dtheta2_mtrx_i)

            # Compute Diag(W_beta_i_irow)

            w_beta_i_irow = w_beta_i_mtrx[rxn_idx]

            diag_w_beta_i_irow = np.diag(w_beta_i_irow)

            # Compute product

            d_theta_beta_d_theta_beta_ri_mtrx = \
                   - rb_i * (dbeta_dtheta_beta_mtrx_pwr2 @ w_beta_i_T_w_beta_i_mtrx \
                            + \
                            d2beta_dtheta2_mtrx @ diag_w_beta_i_irow)

        # *******************************************************************************************
        # Assembly

        # General case
        if theta_kf_vec is not None and theta_kb_vec is not None and \
           theta_alpha_lst is not None and theta_beta_lst is not None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_kb_d_theta_kf_ri_mtrx,
                                            d_theta_alpha_d_theta_kf_ri_mtrx,
                                            d_theta_beta_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_kb_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_kb_d_theta_kb_ri_mtrx,
                                            d_theta_alpha_d_theta_kb_ri_mtrx,
                                            d_theta_beta_d_theta_kb_ri_mtrx])

            hessian_ri_3rd_row = np.hstack([d_theta_alpha_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_kb_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_alpha_ri_mtrx,
                                            d_theta_beta_d_theta_alpha_ri_mtrx])

            hessian_ri_4th_row = np.hstack([d_theta_beta_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_kb_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_alpha_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_beta_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row,
                                    hessian_ri_3rd_row,
                                    hessian_ri_4th_row])

        # Forward case
        elif theta_kf_vec is not None and theta_alpha_lst is not None and \
             theta_kb_vec is None and theta_beta_lst is None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_alpha_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_alpha_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_alpha_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row])

        # k's only case
        elif theta_kf_vec is not None and theta_kb_vec is not None and \
             theta_alpha_lst is None and theta_beta_lst is None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_kb_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_kb_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_kb_d_theta_kb_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row])

        # kfs only case
        elif theta_kf_vec is not None and theta_kb_vec is None and \
             theta_alpha_lst is None and theta_beta_lst is None:

            hessian_ri = d_theta_kf_d_theta_kf_ri_mtrx

        # kbs only case
        elif theta_kf_vec is None and theta_kb_vec is not None and \
             theta_alpha_lst is None and theta_beta_lst is None:

            hessian_ri = d_theta_kb_d_theta_kb_ri_mtrx

        # k's and alphas only case
        elif theta_kf_vec is not None and theta_kb_vec is not None and \
             theta_alpha_lst is not None and theta_beta_lst is None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_kb_d_theta_kf_ri_mtrx,
                                            d_theta_alpha_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_kb_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_kb_d_theta_kb_ri_mtrx,
                                            d_theta_alpha_d_theta_kb_ri_mtrx])

            hessian_ri_3rd_row = np.hstack([d_theta_alpha_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_kb_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_alpha_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row, hessian_ri_2nd_row,
                                    hessian_ri_3rd_row])

        # alphas only case
        elif theta_kf_vec is None and theta_kb_vec is None and \
             theta_alpha_lst is not None and theta_beta_lst is None:

            hessian_ri = d_theta_alpha_d_theta_alpha_ri_mtrx

        # betas only case
        elif theta_kf_vec is None and theta_kb_vec is None and \
             theta_alpha_lst is None and theta_beta_lst is not None:

            hessian_ri = d_theta_beta_d_theta_beta_ri_mtrx

        # alphas and betas only case
        elif theta_kf_vec is None and theta_kb_vec is None and \
             theta_alpha_lst is not None and theta_beta_lst is not None:

            hessian_ri_1st_row = np.hstack([d_theta_alpha_d_theta_alpha_ri_mtrx,
                                            d_theta_beta_d_theta_alpha_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_beta_d_theta_alpha_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_beta_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row])

        # kfs and betas only case
        elif theta_kf_vec is not None and theta_kb_vec is None and \
             theta_alpha_lst is None and theta_beta_lst is not None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_beta_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_beta_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_beta_ri_mtrx])

            hessian_ri=np.vstack([hessian_ri_1st_row,
                                  hessian_ri_2nd_row])

        # kbs and alphas only case
        elif theta_kf_vec is None and theta_kb_vec is not None and \
             theta_alpha_lst is not None and theta_beta_lst is None:

            hessian_ri_1st_row = np.hstack([d_theta_kb_d_theta_kb_ri_mtrx,
                                            d_theta_alpha_d_theta_kb_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_alpha_d_theta_kb_ri_mtrx.transpose(),
                                            d_theta_alpha_d_theta_alpha_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row])

        # kfs and kbs and betas only case
        elif theta_kf_vec is not None and theta_kb_vec is not None and \
             theta_alpha_lst is None and theta_beta_lst is not None:

            hessian_ri_1st_row = np.hstack([d_theta_kf_d_theta_kf_ri_mtrx,
                                            d_theta_kb_d_theta_kf_ri_mtrx,
                                            d_theta_beta_d_theta_kf_ri_mtrx])

            hessian_ri_2nd_row = np.hstack([d_theta_kb_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_kb_d_theta_kb_ri_mtrx,
                                            d_theta_beta_d_theta_kb_ri_mtrx])

            hessian_ri_3rd_row = np.hstack([d_theta_beta_d_theta_kf_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_kb_ri_mtrx.transpose(),
                                            d_theta_beta_d_theta_beta_ri_mtrx])

            hessian_ri = np.vstack([hessian_ri_1st_row,
                                    hessian_ri_2nd_row,
                                    hessian_ri_3rd_row])

        else:
            assert False, 'Hessian ri case not implemented.'

        return hessian_ri

    def __get_kf(self):
        '''Utility for returning a packed kf vector.

        Should this return the stoichiometric coefficients in case there is no data in `self.data`?

        Returns
        -------
        kf_vec: numpy.ndarray
        '''

        kf_vec = np.zeros(len(self.reactions), dtype = np.float64)

        for idx, rxn_data in enumerate(self.data):
            kf_vec[idx] = rxn_data['k_f']

        return kf_vec
    def __set_kf(self, kf_vec):
        '''Utility for setting kf from packed vectors.

        Parameters
        ----------
        kf_vec: numpy.ndarray

        Returns
        -------
        '''

        assert isinstance(kf_vec, numpy.ndarray)
        assert kf_vec.size == len(self.reactions)

        for idx, rxn_data in enumerate(self.data):
            rxn_data['k_f'] = kf_vec[idx]
    kf = property(__get_kf, __set_kf, None, None)

    def __get_kb(self):
        '''Utility for returning a packed kb vector.

        Should this return the stoichiometric coefficients in case there is no data in `self.data`?

        Returns
        -------
        kb_vec: numpy.ndarray
        '''

        kb_vec= np.zeros(len(self.reactions), dtype = np.float64)

        for idx, rxn_data in enumerate(self.data):
            kb_vec[idx]=rxn_data['k_b']

        return kb_vec
    def __set_kb(self, kb_vec):
        '''Utility for setting kb from packed vectors.

        Parameters
        ----------
        kb_vec: numpy.ndarray

        Returns
        -------
        '''

        assert isinstance(kb_vec, numpy.ndarray)
        assert kb_vec.size == len(self.reactions)

        for idx, rxn_data in enumerate(self.data):
            rxn_data['k_b']=kb_vec[idx]
    kb = property(__get_kb, __set_kb, None, None)

    def __get_ks(self):
        '''Utility for returning packed kf and kb into vectors.

        Should this return the stoichiometric coefficients in case there is no data in `self.data`?

        Returns
        -------
        (kf_vec, kb_vec): tuple(numpy.ndarray, numpy.ndarray)
        '''

        return (self.__get_kf(), self.__get_kb())
    def __set_ks(self, kf_kb_pair):
        '''Utility for setting kf and kb from packed vectors.

        Parameters
        ----------
        kf_kb_pair: tuple(numpy.ndarray, numpy.ndarray)
            If any element of the tuple is None, the corresponding data is not updated.

        Returns
        -------
        '''

        assert isinstance(kf_kb_pair, tuple)
        assert len(kf_kb_pair) == 2

        if kf_kb_pair[0] is not None:
            self.__set_kf(kf_kb_pair[0])
        if kf_kb_pair[1] is not None:
            self.__set_kb(kf_kb_pair[1])
    ks = property(__get_ks, __set_ks, None, None)

    def __get_alpha(self):
        '''Utility for packing alpha into a list of matrices.

        The return from this method is a list of compressed unstructured data since each reaction
        typically has a different number of active species, hence different number of associated
        power-law exponents.
        The ids of the active species are passed in the first row of the matrices. Each of the alpha
        matrices have 2 rows. First row with ids, second row with exponents.

        Returns
        -------
        alpha_lst: list(numpy.ndarray)
        '''

        alpha_lst = list()  # list of matrices

        for (idx, rxn_data) in enumerate(self.data):

            (reactants_ids, ) = np.where(self.stoic_mtrx[idx, :] < 0)
            # reactants_ids = reactants_ids_lst[idx]

            if 'alpha' in rxn_data.keys():
                alpha_dict = rxn_data['alpha']
                exponents = list()
                active_reactants_ids = list()
                for j in reactants_ids:
                    spc_name = self.species_names[j]
                    alpha = alpha_dict[spc_name]
                    if alpha != -9999:  # exclude inactive species
                        active_reactants_ids.append(j)
                        exponents.append(alpha_dict[spc_name])

                reactants_ids_alphas = np.array((active_reactants_ids, exponents))  # 2-row matrix

            else:
                assert False
                exponents = -self.stoic_mtrx[idx, reactants_ids]

            alpha_lst.append(reactants_ids_alphas)

        return alpha_lst
    def __set_alpha(self, alpha_lst):
        '''Utility for setting alpha from packed matrices.

        The alpha list of matrices with values for the exponents and corresponding active
        species ids. Note that this will change the internal data of the object including inactive
        species if the user intends to.

        Parameters
        ----------
        alpha: list(numpy.ndarray)
            If any element of the list is None, the corresponding data is not updated.

        '''

        if alpha_lst is not None:
            assert isinstance(alpha_lst, list)

            for idx, rxn_data in enumerate(self.data):

                if 'alpha' in rxn_data.keys():
                    alpha_dict = rxn_data['alpha']
                    alpha_mtrx = alpha_lst[idx]
                    reactants_ids = alpha_mtrx[0, :].astype(int)
                    exponents = alpha_mtrx[1, :]
                    for jdx, j in enumerate(reactants_ids):
                        spc_name = self.species_names[j]
                        alpha_dict[spc_name] = exponents[jdx]
                else:
                    assert False
                    exponents = -self.stoic_mtrx[idx, reactants_ids]
    alpha = property(__get_alpha, __set_alpha, None, None)

    def __get_beta(self):
        """Utility for packing beta exponents into a list of matrices.

        The return from this method is a list of compressed unstructured data since each reaction
        typically has a different number of active species, hence different number of associated
        power-law exponents.
        The ids of the active species are passed in the first row of the matrices. Each of the alpha
        and beta matrices have 2 rows. First row with ids, second row with exponents.

        Returns
        -------
        beta_lst: list(numpy.ndarray)
        """

        beta_lst = list()   # list of matrices

        for (idx, rxn_data) in enumerate(self.data):

            (products_ids, ) = np.where(self.stoic_mtrx[idx, :] > 0)

            if 'beta' in rxn_data.keys():
                beta_dict = rxn_data['beta']
                exponents = list()
                active_products_ids = list()
                for j in products_ids:
                    spc_name = self.species_names[j]
                    beta = beta_dict[spc_name]
                    if beta != -9999:  # exclude inactive species
                        active_products_ids.append(j)
                        exponents.append(beta_dict[spc_name])

                products_ids_betas = np.array((active_products_ids, exponents))  # 2-row matrix

            else:
                assert False
                exponents = self.stoic_mtrx[idx, products_ids]

            beta_lst.append(products_ids_betas)

        return beta_lst
    def __set_beta(self, beta_lst):
        '''Utility for setting beta from packed matrices.

        The alpha list of matrices with values for the exponents and corresponding active
        species ids. Note that this will change the internal data of the object including inactive
        species if the user intends to.

        Parameters
        ----------
        beta: list(numpy.ndarray)
            If any element of the list is None, the corresponding data is not updated.

        '''

        if beta_lst is not None:
            assert isinstance(beta_lst, list)

            for idx, rxn_data in enumerate(self.data):

                if 'beta' in rxn_data.keys():
                    beta_dict = rxn_data['beta']
                    beta_mtrx = beta_lst[idx]
                    products_ids = beta_mtrx[0, :].astype(int)
                    exponents = beta_mtrx[1, :]
                    for jdx, j in enumerate(products_ids):
                        spc_name = self.species_names[j]
                        beta_dict[spc_name] = exponents[jdx]
                else:
                    assert False
                    exponents = self.stoic_mtrx[idx, products_ids]
    beta = property(__get_beta, __set_beta, None, None)

    def __get_power_law_exponents(self):
        '''Utility for packing alpha and beta exponents into a list of vectors.

        The return from this method is a pair of unstructured data since each reaction typically has
        a different number of active species, hence different number of associated power-law exponents.
        The ids of the active species are passed in the first row of the matrices. Each of the alpha
        and beta matrices have 2 rows. First row with ids, second row with exponents.

        Returns
        -------
        (alpha_lst, beta_lst): tuple(list(numpy.ndarray), list(numpy.ndarray))
        '''

        alpha_lst = list()  # list of matrices
        beta_lst = list()   # list of matrices

        for (idx, rxn_data) in enumerate(self.data):

            (reactants_ids, ) = np.where(self.stoic_mtrx[idx, :] < 0)
            # reactants_ids = reactants_ids_lst[idx]

            if 'alpha' in rxn_data.keys():
                alpha_dict = rxn_data['alpha']
                exponents = list()
                active_reactants_ids = list()
                for j in reactants_ids:
                    spc_name = self.species_names[j]
                    alpha = alpha_dict[spc_name]
                    if alpha != -9999:  # exclude inactive species
                        active_reactants_ids.append(j)
                        exponents.append(alpha_dict[spc_name])

                reactants_ids_alphas = np.array((active_reactants_ids, exponents))  # 2-row matrix

            else:
                assert False
                exponents = -self.stoic_mtrx[idx, reactants_ids]

            alpha_lst.append(reactants_ids_alphas)

            (products_ids, ) = np.where(self.stoic_mtrx[idx, :] > 0)

            if 'beta' in rxn_data.keys():
                beta_dict = rxn_data['beta']
                exponents = list()
                active_products_ids = list()
                for j in products_ids:
                    spc_name = self.species_names[j]
                    beta = beta_dict[spc_name]
                    if beta != -9999:  # exclude inactive species
                        active_products_ids.append(j)
                        exponents.append(beta_dict[spc_name])

                products_ids_betas = np.array((active_products_ids, exponents))  # 2-row matrix

            else:
                assert False
                exponents = self.stoic_mtrx[idx, products_ids]

            beta_lst.append(products_ids_betas)

        return (alpha_lst, beta_lst)
    def __set_power_law_exponents(self, alpha_beta_pair):
        '''Utility for setting alpha and beta from packed vectors.

        The alpha and vector lists of matrices with values for the exponents and corresponding active
        species ids. Note that this will change the internal data of the object including inactive
        species if the user intends to.

        Parameters
        ----------
        alpha_beta_pair: tuple(list(numpy.ndarray), list(numpy.ndarray))
            If any element of the tuple is None, the corresponding data is not updated.

        '''

        assert isinstance(alpha_beta_pair, tuple)
        #assert len(alpha_beta_pair) == 2

        if alpha_beta_pair[0] is not None:
            assert isinstance(alpha_beta_pair[0], list)
            alpha_lst=alpha_beta_pair[0]

            for idx, rxn_data in enumerate(self.data):

                if 'alpha' in rxn_data.keys():
                    alpha_dict = rxn_data['alpha']
                    alpha_mtrx = alpha_lst[idx]
                    reactants_ids = alpha_mtrx[0, :].astype(int)
                    exponents = alpha_mtrx[1, :]
                    for jdx, j in enumerate(reactants_ids):
                        spc_name = self.species_names[j]
                        alpha_dict[spc_name] = exponents[jdx]
                else:
                    assert False
                    exponents = -self.stoic_mtrx[idx, reactants_ids]

        if alpha_beta_pair[1] is not None:
            assert isinstance(alpha_beta_pair[1], list)
            beta_lst = alpha_beta_pair[1]

            for idx, rxn_data in enumerate(self.data):

                if 'beta' in rxn_data.keys():
                    beta_dict = rxn_data['beta']
                    beta_mtrx = beta_lst[idx]
                    products_ids = beta_mtrx[0, :].astype(int)
                    exponents = beta_mtrx[1, :]
                    for jdx, j in enumerate(products_ids):
                        spc_name = self.species_names[j]
                        beta_dict[spc_name] = exponents[jdx]
                else:
                    assert False
                    exponents = self.stoic_mtrx[idx, products_ids]
    power_law_exponents = property(__get_power_law_exponents, __set_power_law_exponents, None, None)

    def full_rank_sub_mechanisms(self, n_sub_mec = 1000):
        """Construct sub-mechanisms with full-rank stoichiometric matrix.

        Returns
        -------
        sub_mechanisms: list([ReactionMechanism, gidxs, score])

        """

        s_rank = np.linalg.matrix_rank(self.stoic_mtrx)

        if s_rank == min(self.stoic_mtrx.shape):
            return self

        m_reactions=self.stoic_mtrx.shape[0]

        n_mechanisms = math.factorial(m_reactions) /\
                       math.factorial(m_reactions - s_rank) /\
                       math.factorial(s_rank)
        # print('# of all possible sub_mechanisms = %i'%n_mechanisms)

        tmp = combinations(range(m_reactions), s_rank)
        reaction_sets = [i for i in tmp]
        random.shuffle(reaction_sets)  # this may be time consuming

        sub_mechanisms = list()  # list of list

        for idxs in reaction_sets:

            stoic_mtrx=self.stoic_mtrx[idxs, :]

            rank=np.linalg.matrix_rank(stoic_mtrx)

            if rank == s_rank:

                mechanism = list()
                for idx in idxs:
                    mechanism.append(self.__original_mechanism[idx])

                rxn_mech = ReactionMechanism(mechanism = mechanism, order_species =True, reparam=self.reparam)

                assert np.linalg.matrix_rank(rxn_mech.stoic_mtrx) == s_rank

                sub_mechanisms.append([rxn_mech, idxs])

            if len(sub_mechanisms) >= n_sub_mec:
                break

        # Count number of times a global reaction appear in a sub-mechanism
        reactions_hits=np.zeros(m_reactions)
        for sm in sub_mechanisms:
            gidxs = list(sm[1])
            reactions_hits[gidxs] += 1

        # Score the global reactions
        sub_mech_reactions_score=list()
        for subm in sub_mechanisms:
            score=0
            for gid in subm[1]:
                score += reactions_hits[gid]
            sub_mech_reactions_score.append(score)

        sub_mech_reactions_score = np.array(sub_mech_reactions_score)
        sub_mech_reactions_score /= sub_mech_reactions_score.max()
        sub_mech_reactions_score *= 10.0

        results=sorted(zip(sub_mechanisms, sub_mech_reactions_score),
                          key=lambda entry: entry[1], reverse=True )

        sub_mechanisms           = [a for (a,b) in results]
        sub_mech_reactions_score = [b for (a,b) in results]

        # Encode score in to sub_mech_reactions mech. list of list
        for (sr, score) in zip(sub_mechanisms, sub_mech_reactions_score):
            sr += [score]

        # max_score = max( [sm[3] for sm in sub_mechanisms] )

        return sub_mechanisms

    def print_data(self):
        """Helper to print the reaction data line by line.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        rxn_mech.print_data()
        """

        for idx, data in enumerate(self.data):
            print(self.reactions[idx])
            print(data,'\n')

    def print_reactions(self):
        """Helper to print reactions line by line.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        rxn_mech.print_reactions()
        """

        for idx, rxn in enumerate(self.reactions):
            print('R%i '%idx, ': ', self.reactions[idx])

    def print_species(self):
        """Helper to print the whole species data line by line.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        rxn_mech.print_species()
        """

        for spc in self.species:
            print(spc)
            #print('')

    def md_print(self, group='all'):
        """Markdown cell printout of LaTex reactions and species.

        Use with Jupyter Notebooks in a "code" cell.

        Parameters
        ----------
        group: str
            Flag to print groups of quantities as follows:
            'all', 'species', or 'reactions'.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        rxn_mech.md_print()
        """

        assert group in ['all', 'species', 'reactions']

        from IPython.display import Markdown, display

        if group in ['all', 'species']:
            tmp = self.latex_species.replace(',', ' \\quad ')
            tmp = tmp.replace(r'\\', r'\\ ')
            string = '%i **Species:** \n %s'%(len(self.species_names), tmp)
            display(Markdown(string))

        if group in ['all', 'reactions']:
            string = '%i **Reactions:** \n %s'%(len(self.reactions), self.latex_rxn)
            if len(self.reactions) >= 25:
                print('WARNING: display may not be able to handle more than 25 reactions')
            display(Markdown(string))

    def cat_input(self):
        """A helper function to print on std output the original rxn mechanism.

        Note
        ----
        This is typically used with Jupyter notebooks to avoid the use of `!cat filename` which does
        not work on Windows.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        rxn_mech.cat_input()
        """

        if self.file_name is not None:
            assert isinstance(self.file_name, str)

            finput = open(self.file_name, 'rt')

            for line in finput:
                stripped_line = line.strip()
                print(stripped_line)

            finput.close()

    def __latex(self, n_species_line=8):
        """Internal helper for LaTeX typesetting.

        See attributes description and usage with the Python print() function.

        Returns
        -------
        (species_str, rxn_str): (str, str)
            Tuple with strings containing the species LaTex names one after the other separated by commas,
            and the reactions string typeset into a LaTex align environment.

        Examples
        --------
        rxn_mech = ReactionMechanism(file_name='some_mech.txt')
        print(rxn_mech.species_str)
        print(rxn_mech.rxn_str)

        then copy the output and paste into a markdown cell in a Jupyter notebook or
        LaTeX document.
        """

        # Latex species
        species_str = '\\begin{align*} \n &'
        for idx, spc in enumerate(self.species[:-1]):
            if idx > 0 and idx%(n_species_line-1) == 0: # add a new line after every 8 species
                species_str += spc.latex_name + ', \\\\ \n '
                species_str += '& '
            else:
                species_str += spc.latex_name + ', '

        species_str += self.species[-1].latex_name
        species_str += '\\end{align*}'

        # Latex reactions into align environment
        # No header
        #rxn_str = self.header + '\n'
        #rxn_str += '\\begin{align*} \n'
        rxn_str = '\\begin{align*} \n'
        for idx,row in enumerate(self.stoic_mtrx):

            (reactants_ids, ) = np.where(row < 0)

            for j in reactants_ids[:-1]:
                coeff = abs(self.stoic_mtrx[idx,j])
                if coeff != 1:
                    rxn_str += str(coeff) + '\\,' + self.species[j].latex_name + r'\ + \ '
                else:
                    rxn_str += self.species[j].latex_name + r'\ + \ '

            j = reactants_ids[-1]
            coeff = abs(self.stoic_mtrx[idx,j])
            if coeff != 1:
                rxn_str += str(coeff) + '\\,' + self.species[j].latex_name
            else:
                rxn_str += self.species[j].latex_name

            if self.reaction_direction_symbol[idx] == '->':
                rxn_str += r'\ &\longrightarrow \ '
            elif self.reaction_direction_symbol[idx] == '<->':
                rxn_str += r'\ &\longleftrightarrow \ '
            elif self.reaction_direction_symbol[idx] == '<=>':
                rxn_str += r'\ &\longleftrightarrow \ '
            elif self.reaction_direction_symbol[idx] == '<-':
                rxn_str += r'\ &\longleftarrow \ '
            else:
                assert False, 'Unknown reaction direction.'

            (products_ids, ) = np.where(row > 0)
            for j in products_ids[:-1]:
                coeff = abs(self.stoic_mtrx[idx,j])
                if coeff != 1:
                    rxn_str += str(coeff) + '\\,' + self.species[j].latex_name + r'\ + \ '
                else:
                    rxn_str += self.species[j].latex_name + r'\ + \ '

            j = products_ids[-1]
            coeff = abs(self.stoic_mtrx[idx,j])
            if coeff != 1:
                rxn_str += str(coeff) + '\\,' + self.species[j].latex_name + '\\\\ \n'
            else:
                rxn_str += self.species[j].latex_name + '\\\\ \n'

        rxn_str += '\\end{align*} \n'

        return (species_str, rxn_str)

    def __str__(self):
        s = '\n\t **ReactionMechanism()**:' + \
            '\n\t header: %s;' + \
            '\n\t reactions: %s;' + \
            '\n\t data: %s;' + \
            '\n\t species_names: %s;' + \
            '\n\t species: %s' + \
            '\n\t max mass balance residual: %s'
        return s % (self.header,
                    self.reactions,
                    self.data,
                    self.species_names,
                    self.species,
                    str(self.max_mass_balance_residual()))

    def __repr__(self):
        s = '\n\t **ReactionMechanism()**:' + \
            '\n\t header: %s;' + \
            '\n\t reactions: %s;' + \
            '\n\t data: %s;' + \
            '\n\t species_names: %s;' + \
            '\n\t species: %s;' + \
            '\n\t max mass balance residual: %s'
        return s % (self.header,
                    self.reactions,
                    self.data,
                    self.species_names,
                    self.species,
                    str(self.max_mass_balance_residual()))

# Non-member methods

def print_reaction_sub_mechanisms(sub_mechanisms, mode=None, n_sub_mech=None):
    """Nice printout of a scored reaction sub-mechanism list

    Once the sub-mechanisms have been computed this function makes a printout.
    Note: This is not a class method; just a helper function to the ReactionMechanism class.

    Parameters
    ----------
    sub_mechanisms: list([ReactionMechanism, gidxs, score])
        Sorted reaction mechanims in the form of a list.

    mode: str
        Printing mode: all, top, None.

    Returns
    -------
    None:

    Examples
    --------

    """
    assert mode is None or n_sub_mech is None
    assert mode =='top' or mode =='all' or mode==None
    assert isinstance(n_sub_mech, int) or n_sub_mech is None

    if mode is None and n_sub_mech is None:
        mode = 'all'

    if n_sub_mech is None:
        if mode == 'all':
            n_sub_mech = len(sub_mechanisms)
        elif mode == 'top':
            scores = [sm[2] for sm in sub_mechanisms]
            max_score = max(scores)
            tmp = list()
            for s in scores:
                if s == max_score:
                    tmp.append(s)
            n_sub_mech = len(tmp)
        else:
            assert False, 'illegal mode %r'%mode

    for idx, rsm in enumerate(sub_mechanisms[:n_sub_mech]):
        print('Full-Rank Reaction Sub Mechanism: %s (score %4.2f)'%(idx, rsm[2]))
        print('Species = ',rsm[0].species_names)
        for (r,data,gidx) in zip(rsm[0].reactions, rsm[0].data, rsm[1]):
            print('r%s'%gidx, r,' ', data['info'])
