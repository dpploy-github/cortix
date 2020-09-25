'''
Models a condenser with steam flowing perpendicular to several horizontal tubes.
Returns the runoff temperature of steam from the condenser.
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment.
# https://cortix.org

import logging
import math
import numpy as np
import matplotlib.pyplot as pyplot

import scipy.constants as unit
import iapws.iapws97 as steam_table

from cortix import Module
from cortix.support.phase_new import PhaseNew as Phase
from cortix import Quantity

class Condenser(Module):
    """Boiling water reactor single-point reactor.

    Notes
    -----
    These are the `port` names available in this module to connect to respective
    modules: `turbine`, and `pump`.
    See instance attribute `port_names_expected`.
    """

    def __init__(self, params):
        """Constructor.

        Parameters
        ----------
        params: dict
            All parameters for the module in the form of a dictionary.

        """

        super().__init__()

        self.port_names_expected = ['inflow-1', 'inflow-2', 'outflow']

        quantities = list()
        self.ode_params = dict()
        self.params = params
        self.initial_time = params['start-time']
        self.end_time = params['end-time']

        self.show_time = (False, 10.0)
        self.time_step = 10.0
        self.log = logging.getLogger('cortix')

        # data required by the condenser
        self.pipe_diameter = 0.1 #m
        self.liquid_velocity = 10 #m/s
        self.cooling_water_flowrate = 100000 #kg/s
        self.heat_transfer_area = 10200 #m2, or 500 4m long, 0.1m diameter pipes
        self.condensation_ht_coeff = 5000 # w/m-k
        self.subcooling_ht_coeff = 1000 # w/m-k


        # Condenser outflow phase history
        quantities = list()

        flowrate = Quantity(name='condenser-runoff-flowrate',
                            formal_name='Condenser Runoff Flowrate', unit='kg/s', value=0.0,
                            info='Condenser Outflow Flowrate', latex_name=r'$Q$')

        quantities.append(flowrate)

        temp = Quantity(name='temp', formal_name='Condenser Runoff Temp.', unit='K',
                        value=params['condenser-runoff-temp'], info='Condenser Outflow Temperature', latex_name=r'$T_o$')

        quantities.append(temp)

        self.outflow_phase = Phase(time_stamp=self.initial_time, time_unit='s',
                                   quantities=quantities)

        self.inflow_state = None

    def run(self, *args):

        #self.__zero_ode_parameters()

        time = self.initial_time
        end_time = self.end_time

        while time < end_time + self.time_step:

            if self.show_time[0] and abs(time%self.show_time[1]-0.0) <= 1.e-1:

                msg = 'time = '+str(round(time/unit.minute, 1))
                self.log.info(msg)

            # Communicate information
            #------------------------
            self.__call_ports(time)

            # Evolve one time step
            #---------------------
            time = self.__step(time)

        return

    def tester(self):
        """Passes a string of test data (temperature, chi) to the condenser to debug the model itself."""
        x = '===================='
        #start with a basic temperature scale
        print(x, '\n', 'TESTING TEMPERATURE', '\n', x)
        temp =  np.linspace(300, 500, 200)
        test_chi = 0
        output_vector = []
        for i in temp:
            output = self.__condenser(0, i, test_chi, self.params)
            print('temp is ', i, ' and outflow temp is ', output)
            output_vector.append(output)
        
        #test how the system responds to different chi's
        print(x, '\n', 'TESTING CHI', '\n', x)
        chi = np.linspace(0, 0.98, 100)
        test_temperature = steam_table._TSat_P(0.005)
        output_vector_2 = []
        for i in chi:
            output = self.__condenser(0, test_temperature, i, self.params)
            print('chi is ', i, ' and outflow temp is ', output)
            output_vector_2.append(output)

        #graph the results
        pyplot.plot(temp, output_vector)
        pyplot.title('inflow temperature vs outflow temperature')
        pyplot.savefig('condenser-inflow-temp-vs-outflow-temp.png')
        pyplot.close()
        pyplot.plot(chi, output_vector_2)
        pyplot.title('inflow quality vs outflow temperature')
        pyplot.savefig('inflow-quality-vs-outflow-temp.png')
        return




    def __call_ports(self, time):

        # Interactions in the inflow port
        #----------------------------------------
        # one way "from" inflow-1

        # from
        self.send(time, 'inflow-1')
        (check_time, inflow_state) = self.recv('inflow-1')
        assert abs(check_time-time) <= 1e-6

        self.inflow_state = inflow_state
        self.inflow_state['time'] = time

        # Interactions in the inflow port
        #----------------------------------------
        # one way "from" inflow-2

        # from
        if self.get_port('inflow-2').connected_port:
            self.send(time, 'inflow-2')
            (check_time, inflow_state) = self.recv('inflow-2')
            assert abs(check_time-time) <= 1e-6

            self.inflow_state = inflow_state
            self.inflow_state['time'] = time

        # Interactions in the outflow port
        #-----------------------------------------
        # one way "to" outflow

        message_time = self.recv('outflow')
        #outflow_state = dict()
        outflow_cool_temp = self.outflow_phase.get_value('temp', time)
        condenser_runoff = dict()
        condenser_runoff['outflow-temp'] = outflow_cool_temp
        self.send((message_time, outflow_cool_temp), 'outflow')

        return

    def __step(self, time):
        assert abs(time-self.inflow_state['time']) <= 1e-6

        temp_in = self.inflow_state['temp']
        chi_in = self.inflow_state['quality']
        pressure = self.inflow_state['pressure']
        flowrate = self.inflow_state['flowrate']

        t_out = self.__condenser(time, temp_in, chi_in, pressure, flowrate, self.params)

        condenser_runoff = self.outflow_phase.get_row(time)

        time = time + self.time_step

        self.outflow_phase.add_row(time, condenser_runoff)
        self.outflow_phase.set_value('temp', t_out)

        return time

    def __condenser(self, time, temp_in, chi_in, p_in, flowrate, params):
        """Simple model to condense a vapor-liquid mixture and subcool it.
        Takes in either superheated steam or a two-phase water mixture, and
        calculates the amount of surface area within a simple condenser required to
        remove all degrees of superheat and condense the mixture.
        """
        if flowrate == 0:
            return 287.15

        #temporary, basic model for condensation, until I can work out a more
        #advanced model that gives realistic values.
        pressure = 0.005
        t_coolant = 14 + 273.15
        h_mixture = steam_table._Region4(pressure, chi_in)['h']
        h_dewpt = steam_table._Region4(pressure, 0)['h']
        heat_pwr = (h_mixture - h_dewpt) * unit.kilo * flowrate
        area_required = heat_pwr/(self.condensation_ht_coeff * (temp_in - t_coolant))
        area_remaining = self.heat_transfer_area - area_required

        if area_remaining < 0:
            return 287.15
        else:
            initial_coolant_temp = 14 + 273.15
            delta_t = heat_pwr/(self.cooling_water_flowrate * unit.kilo * 4.184)
            final_coolant_temp = initial_coolant_temp + delta_t
            subcooling_pwr = area_remaining * self.subcooling_ht_coeff * (temp_in - final_coolant_temp)
            delta_tb = subcooling_pwr/(flowrate * 4.184 * unit.kilo)
            runoff_temp = temp_in - delta_tb
            
            if runoff_temp < initial_coolant_temp:
                runoff_temp = initial_coolant_temp
            return(runoff_temp)

        #anything below this point is not currently in use; I need to figure
        #out a more accurate condensation model, the current one calculates
        #heat transfer coefficients in excess of 100k w/m2-k
        
        #print(time, ' chi_in is ', chi_in)
        critical_temp = steam_table._TSat_P(0.005)
        condenser_runoff = 14 + 273.15
        #if chi_in == -1 and temp_in > critical_temp: #superheated vapor inlet; deprecated
            #return 293.15
            #x = x_in
            #pressure = 0.005
            #h_steam = steam_table._Region2(temp_in, pressure)['h']
            #h_sat = steam_table._Region4(0.005, 1)['h']
            #q = params['steam flowrate'] * (h_steam - h_sat) * sc.kilo

            #log mean temperature difference
            #delta_Ta = temp_in - critical_temp
            #delta_Tb = 20-10
            #LMTD = (delta_Ta - delta_Tb) / (math.log(delta_Ta) - math.log(delta_Tb))

            #low = steam_table.IAPWS97(T=critical_temp+1, P=pressure)
            #high = steam_table.IAPWS97(T=temp_in, P=pressure)

            #low_conductivity = low.k
            #low_rho = low.rho
            #low_prandtl = low.Prandt
            #low_viscosity = low.mu
            #low_reynolds = (low_rho * 0.1 * 40)/low_viscosity
            #low_nusselt = (0.3 + 0.629 * (low_reynolds ** 0.5) * low_prandtl**(1/3))
            #low_nusselt = low_nusselt/(1+(0.4/low_prandtl)**(2/3))**(1/4)
            #low_nusselt = low_nusselt * (1 + (low_reynolds / 282000)**(5/6))**(4/4.5)
            #nusselt number at the crtical temperature according to the
            #churchill-bernsetin correlation
            #low_heat_transfer_coeff = low_nusselt * (low.k/0.1)

            #high_rho = high.rho
            #high_prandtl = high.Prandt
            #high_viscocity = high.mu
            #high_conductivity = high.k
            #high_viscosity = high.mu
            #high_reynolds = (high_rho * 0.1 * 40)/high_viscosity
            #high_nusselt = (0.3 + 0.629 * (high_reynolds ** 0.5) * high_prandtl**(1/3))
            #high_nusselt = high_nusselt/(1+(0.4/high_prandtl)**(2/3))**(1/4)
            #high_nusselt = high_nusselt * (1 + (high_reynolds / 282000)**(5/6))**(4/4.5)
            #high_heat_transfer_coeff = high_nusselt * (high_conductivity/0.1)

            #average_ht_transfer_coeff = (high_heat_transfer_coeff + low_heat_transfer_coeff)/2

        if chi_in > 0:
            pressure = 0.005
            h_mixture = steam_table._Region4(pressure, chi_in)['h']
            h_dewpt = steam_table._Region4(pressure, 0)['h']
            heat_pwr = (h_mixture - h_dewpt) * unit.kilo * flowrate

            critical_temp = temp_in
            critical_temp = steam_table._TSat_P(pressure)
            sat_liquid = steam_table.IAPWS97(T=critical_temp - 1, P=pressure)
            water_cp = steam_table.IAPWS97(T=287.15, P=0.1013).cp
            delta_tb = heat_pwr/(self.cooling_water_flowrate * \
                    water_cp * unit.kilo)
            t_c_in = 14 + 273.15
            t_c_out = 14 + 273.15 + delta_tb

            lmtd = (t_c_out - t_c_in)/(math.log(critical_temp - t_c_in))/\
                    (critical_temp - t_c_out)

            pipe_diameter = self.pipe_diameter
            liquid_velocity = self.liquid_velocity

            #calculate the convective heat transfer coefficient on a liquid basis
            #from the Churchill-Bernstein correlation
            liquid_conductivity = sat_liquid.k
            liquid_rho = sat_liquid.rho
            liquid_prandtl = sat_liquid.Prandt
            liquid_viscosity = sat_liquid.mu
            liquid_reynolds = (liquid_rho * pipe_diameter * liquid_velocity)/\
                    liquid_viscosity

            liquid_nusselt = (0.3 + 0.629 * (liquid_reynolds ** 0.5) * \
                    liquid_prandtl**(1/3))
            liquid_nusselt = liquid_nusselt/(1+(0.4/liquid_prandtl)**(2/3))**(1/4)
            #nusselt number at the critical temperature according to the
            #churchill-bernstein correlation
            liquid_nusselt = liquid_nusselt * (1 + (liquid_reynolds / 282000)**(5/6))**(4/4.5)
            #overall convective heat transfer coefficient
            liquid_heat_transfer_coeff = liquid_nusselt * (liquid_conductivity/pipe_diameter)

            #determine the Martinelli paremeter
            steam = steam_table.IAPWS97(T=critical_temp+1, P=pressure)
            steam_rho = steam.rho
            steam_viscosity = steam.mu

            #print(chi_in)

            xtt = (((1 -chi_in)/chi_in)**1.8 * (steam_rho/liquid_rho) * (liquid_viscosity/
                                                                         steam_viscosity)**0.5)
            xtt = math.sqrt(xtt)

            #determine the overall heat transfer coefficient from the McNaught
            #expression
            alpha_sh = liquid_heat_transfer_coeff * (1/xtt)**0.78
            alpha_sh = alpha_sh
            print(alpha_sh, 'alpha_sh')

            #determine how far the two-phase mixture gets in the condenser
            #before being fully condensed
            condensation_area = (heat_pwr/(alpha_sh * lmtd))

            remaining_area = self.heat_transfer_area - condensation_area
            condenser_runoff = critical_temp
            print(remaining_area)
            #if time > params['malfunction start'] and time < params['malfunction end']:
                #condenser_runoff = 14 + 273.15

            if remaining_area < 0:
                condenser_runoff = critical_temp

            elif remaining_area > 0:
            #subcool the remaining liquid
            #iterative method
            #0. guess an ending value for the coolant and
            #1. calculate high nusselt number
            #2. guess a value for low temperature
            #3. calculate low nusselt number
            #4. calculate lmtd and average heat transfer coefficient
            #5. calculate q
            #6. use q to calculate a new ending temperature for the water and the
            #   vapor
            #7. rinse and repeat

                # initial guess
                final_coolant_temperature_guess = steam_table._TSat_P(0.005) - 10
                final_runoff_temperature_guess = 300 # initial guess
                final_runoff_temperature = 0 #iteration variable

                # Loop until convergeance, +- 1 degree kelvin
                # Compute the runoff temperature (subcooled liquid)
                while abs((final_runoff_temperature_guess -
                           final_runoff_temperature)) > 1:

                    runoff_in = steam_table._TSat_P(0.005)
                    coolant_in = 14 + 273.15 #k
                    #assert(final_coolant_temperature_guess < final_runoff_temperature_guess)
                    #assert(final_coolant_temperature_guess > coolant_in)

                    final_runoff_temperature = final_runoff_temperature_guess

                    #guess the LMTD
                    lmtd = ((runoff_in - final_coolant_temperature_guess) - (final_runoff_temperature - coolant_in)) / math.log(((runoff_in -  final_coolant_temperature_guess) / (final_runoff_temperature - coolant_in)))

                    #average wall temperatures
                    upper_wall_temp = (coolant_in + runoff_in)/2
                    lower_wall_temp = (final_runoff_temperature + \
                            final_coolant_temperature_guess)/2

                    #high temperature heat transfer coefficient calculation
                    high_properties = steam_table.IAPWS97(T=upper_wall_temp,
                                                          P=pressure)
                    high_conductivity = high_properties.k
                    high_rho = high_properties.rho
                    high_prandtl = high_properties.Prandt
                    high_viscosity = high_properties.mu
                    high_reynolds = (high_rho * pipe_diameter * liquid_velocity)/\
                            high_viscosity

                    high_nusselt = (0.3 + 0.629 * (high_reynolds ** 0.5) * \
                            high_prandtl**(1/3))
                    high_nusselt = high_nusselt/(1+(0.4/high_prandtl)**(2/3))**(1/4)
                    high_nusselt = high_nusselt * (1 + (high_reynolds / 282000)**(5/6))**(4/4.5)
                    #overall convective heat transfer coefficient
                    high_heat_transfer_coeff = high_nusselt * (high_conductivity/pipe_diameter)

                    #   low temperature heat transfer coefficient calculation
                    low_properties = steam_table.IAPWS97(T=lower_wall_temp, P=pressure)
                    low_conductivity = low_properties.k
                    low_rho = low_properties.rho
                    low_prandtl = low_properties.Prandt
                    low_viscosity = low_properties.mu
                    low_reynolds = (low_rho * pipe_diameter * \
                            liquid_velocity)/low_viscosity

                    low_nusselt = 0.3 + 0.629 * (low_reynolds ** 0.5) * \
                            low_prandtl**(1/3)
                    low_nusselt = low_nusselt/(1+(0.4/low_prandtl)**(2/3))**(1/4)
                    low_nusselt = low_nusselt * (1 + (low_reynolds / 282000)**(5/6))**(4/4.5)
                    #overall convective heat transfer coefficient
                    low_heat_transfer_coeff = low_nusselt * (low_conductivity/pipe_diameter)

                    #average heat transfer coefficient
                    average_heat_transfer_coeff = ((high_heat_transfer_coeff
                                                    + low_heat_transfer_coeff)/2)

                    #calculate q = UAdeltaTlm
                    q_1 = average_heat_transfer_coeff * remaining_area * lmtd/10

                    #calculate actual coolant and liquid water temperatures

                    coolant_cp = steam_table.IAPWS97(T=coolant_in, P=0.005).cp
                    iterated_coolant_final_temp = ((q_1/(coolant_cp *
                                                         self.cooling_water_flowrate
                                                         * unit.kilo)) + coolant_in)

                    steam_cp = steam_table.IAPWS97(T=runoff_in, P=0.005).cp
                    iterated_steam_final_temp = (runoff_in - (q_1/(steam_cp *
                                                                   flowrate *
                                                                   unit.kilo)))

                    if (iterated_steam_final_temp < coolant_in or
                            iterated_coolant_final_temp > runoff_in or
                            iterated_steam_final_temp < iterated_coolant_final_temp):

                        final_runoff_temperature_guess = coolant_in
                        #final_coolant_runoff_guess = runoff_in use for later implementation
                        break

                    final_runoff_temperature_guess = iterated_steam_final_temp
                    final_coolant_temperature_guess = iterated_coolant_final_temp

                # end of: Loop until convergeance, +- 1 degree kelvin
                condenser_runoff = final_runoff_temperature_guess

        else:
            condenser_runoff = 14 + 273.15

        return condenser_runoff
