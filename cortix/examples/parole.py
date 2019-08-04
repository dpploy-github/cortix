#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment.
# https://cortix.org

import pickle

import numpy as np
import scipy.constants as const
from scipy.integrate import odeint
from cortix.src.module import Module
from cortix.support.phase import Phase
from cortix.support.quantity import Quantity

class Parole(Module):
    '''
    Parole Cortix module used to model criminal group population in a parole system.

    Note
    ----
    `prison`: this is a `port` for the rate of population groups to/from the
        Prison domain.

    `freedom`: this is a `port` for the rate of population groups to/from the Freedom
        domain module.

    `visualization`: this is a `port` that sends data to a visualization module.
    '''

    def __init__(self, n_groups=1):

        super().__init__()

        quantities      = list()
        self.ode_params = dict()

        self.initial_time = 0.0 * const.day
        self.end_time     = 100 * const.day
        self.time_step    = 0.5 * const.day

        # Population groups
        self.n_groups = n_groups
        factor = 0.0

        # Parole population groups
        feg_0 = np.random.random(self.n_groups) * factor
        feg = Quantity(name='feg', formalName='parole-pop-grps',
                unit='individual', value=feg_0)
        quantities.append(feg)

        # Model parameters: commitment coefficients and their modifiers

        # Parole to freedom
        ce0g_0 = np.random.random(self.n_groups) / const.day
        ce0g = Quantity(name='ce0g', formalName='commit-freedom-coeff-grps',
               unit='individual', value=ce0g_0)
        self.ode_params['commit-to-freedom-coeff-grps'] = ce0g_0
        quantities.append(ce0g)

        me0g_0 = np.random.random(self.n_groups)
        me0g = Quantity(name='me0g', formalName='commit-freedom-coeff-mod-grps',
               unit='individual', value=me0g_0)
        self.ode_params['commit-to-freedom-coeff-mod-grps'] = me0g_0
        quantities.append(me0g)

        # Parole to prison  
        cepg_0 = np.random.random(self.n_groups) / const.day
        cepg = Quantity(name='cepg', formalName='commit-prison-coeff-grps',
               unit='individual', value=cepg_0)
        self.ode_params['commit-to-prison-coeff-grps'] = cepg_0
        quantities.append(cepg)

        mepg_0 = np.random.random(self.n_groups)
        mepg = Quantity(name='mepg', formalName='commit-prison-coeff-mod-grps',
               unit='individual', value=mepg_0)
        self.ode_params['commit-to-prison-coeff-mod-grps'] = mepg_0
        quantities.append(mepg)

        # Death term
        self.ode_params['parole-death-rates'] = np.zeros(self.n_groups)

        # Phase state
        self.population_phase = Phase(self.initial_time, time_unit='s',
                quantities=quantities)

        self.population_phase.SetValue('feg', feg_0, self.initial_time)

        # Set the state to the phase state
        self.state = self.population_phase

        return

    def run(self, state_comm=None, idx_comm=None):

        time = self.initial_time

        while time < self.end_time:

            # Interactions in the prison port
            #--------------------------------
            # two way "to" and "from" prison

            # to
            message_time = self.recv('prison')
            prison_outflow_rates = self.compute_outflow_rates( message_time, 'prison' )
            self.send( (message_time, prison_outflow_rates), 'prison' )

            # from
            self.send( time, 'prison' )
            (check_time, prison_inflow_rates) = self.recv('prison')
            assert abs(check_time-time) <= 1e-6
            self.ode_params['prison-inflow-rates'] = prison_inflow_rates

            # Interactions in the freedom port
            #------------------------------

            # compute freedom outflow rate

            # Interactions in the visualization port
            #---------------------------------------

            feg = self.population_phase.GetValue('feg')
            self.send( feg, 'visualization' )

            # Evolve prison group population to the next time stamp
            #------------------------------------------------------

            time = self.step( time )

        self.send('DONE', 'visualization') # this should not be needed: TODO

        if state_comm:
            try:
                pickle.dumps(self.state)
            except pickle.PicklingError:
                state_comm.put((idx_comm,None))
            else:
                state_comm.put((idx_comm,self.state))

    def rhs_fn(self, u_vec, t, params):

        feg = u_vec  # parole population groups

        prison_inflow_rates = params['prison-inflow-rates']

        inflow_rates  = prison_inflow_rates

        ce0g = self.ode_params['commit-to-freedom-coeff-grps']
        me0g = self.ode_params['commit-to-freedom-coeff-mod-grps']

        cepg = self.ode_params['commit-to-prison-coeff-grps']
        mepg = self.ode_params['commit-to-prison-coeff-mod-grps']

        outflow_rates = ( ce0g * me0g + cepg * mepg ) * feg

        death_rates = params['parole-death-rates']

        dt_feg = inflow_rates - outflow_rates - death_rates

        return dt_feg

    def step(self, time=0.0):
        r'''
        ODE IVP problem:
        Given the initial data at :math:`t=0`,
        :math:`u = (u_1(0),u_2(0),\ldots)`
        solve :math:`\frac{\text{d}u}{\text{d}t} = f(u)` in the interval
        :math:`0\le t \le t_f`.

        Parameters
        ----------
        time: float
            Time in the droplet unit of time (seconds).

        Returns
        -------
        None
        '''

        u_vec_0 = self.population_phase.GetValue('feg', time)
        t_interval_sec = np.linspace(0.0, self.time_step, num=2)

        (u_vec_hist, info_dict) = odeint(self.rhs_fn,
                                         u_vec_0, t_interval_sec,
                                         args=( self.ode_params, ),
                                         rtol=1e-4, atol=1e-8, mxstep=200,
                                         full_output=True)

        assert info_dict['message'] =='Integration successful.', info_dict['message']

        u_vec = u_vec_hist[1,:]  # solution vector at final time step
        values = self.population_phase.GetRow(time) # values at previous time

        time += self.time_step

        self.population_phase.AddRow(time, values)

        # Update current values
        self.population_phase.SetValue('feg', u_vec, time)

        return time

    def compute_outflow_rates(self, time, name):

        feg = self.population_phase.GetValue('feg',time)

        if name == 'prison':

            cepg = self.ode_params['commit-to-prison-coeff-grps']
            mepg = self.ode_params['commit-to-prison-coeff-mod-grps']

            outflow_rates = cepg * mepg * feg

            return outflow_rates

        if name == 'freedom':

            ce0g = self.ode_params['commit-to-freedom-coeff-grps']
            me0g = self.ode_params['commit-to-freedom-coeff-mod-grps']

            outflow_rates = ce0g * me0g * feg

            return outflow_rates
