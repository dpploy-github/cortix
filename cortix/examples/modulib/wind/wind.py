#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment
# https://cortix.org
#
# All rights reserved, see COPYRIGHT for full restrictions.
# https://github.com/dpploy/cortix/blob/master/COPYRIGHT.txt
#
# Licensed under the University of Massachusetts Lowell LICENSE:
# https://github.com/dpploy/cortix/blob/master/LICENSE.txt
'''
Wind module example in Cortix.
'''
#*********************************************************************************
import os, sys, io, time
import logging
from collections             import namedtuple
import math
import numpy as npy

from cortix.src.utils.xmltree import XMLTree
from cortix.support.quantity import Quantity
from cortix.support.specie   import Specie
from cortix.support.phase    import Phase
#*********************************************************************************

class Wind():
    r'''
    Wind module used example in Cortix.
    '''

#*********************************************************************************
# Construction
#*********************************************************************************

    def __init__( self, 
            slot_id,
            input_full_path_file_name,
            manifesto_full_path_file_name,
            work_dir, 
            ports = list(),
            cortix_start_time = 0.0,
            cortix_final_time = 0.0,
            cortix_time_step  = 0.0,
            cortix_time_unit  = None 
                ):

        #.........................................................................
        # Sanity test 

        assert isinstance(slot_id, int),'-> slot_id type %r is invalid.'%type(slot_id)
        assert isinstance(ports, list),'-> ports type %r is invalid.'%type(ports)
        assert len(ports) > 0
        assert isinstance(cortix_start_time,float),'-> time type %r is invalid.'%\
               type(cortix_start_time)
        assert isinstance(cortix_final_time, float),'-> time type %r is invalid.'%\
               type(cortix_final_time)
        assert isinstance(cortix_time_step, float),'-> time step type %r is invalid.'%\
               type(cortix_time_step)
        assert isinstance(cortix_time_unit, str),'-> time unit type %r is invalid.'%\
               type(cortix_time_unit)

        # Logging: access Cortix Launcher logging facility
        self.__log = logging.getLogger('launcher-wind_'+str(slot_id)+'.cortix_driver.wind')
        self.__log.info('initializing an object of Wind()')

       # Read the manisfesto
        self.__read_manifesto( manifesto_full_path_file_name )
        self.__log.info(self.__port_diagram)

        #.........................................................................
        # Member data 

        self.__slot_id = slot_id
        self.__ports   = ports

        # Convert Cortix's time unit to Wind's internal time unit
        if cortix_time_unit == 'minute':
           self.__time_unit_scale = 60.0
        elif cortix_time_unit == 'second':
           self.__time_unit_scale = 1.0   # Wind time unit
        elif cortix_time_unit == 'hour':
           self.__time_unit_scale = 60.0*60.0
        else:
           assert False, 'Cortix time_unit: %r not acceptable.' % time_unit

        self.__cortix_time_unit = cortix_time_unit

        self.__start_time = cortix_start_time * self.__time_unit_scale # Wind time unit
        self.__final_time = cortix_final_time * self.__time_unit_scale # Wind time unit

        if work_dir[-1] != '/': work_dir = work_dir + '/'
        self.__wrkDir = work_dir

        # Signal to start operation
        self.__goSignal = True    # start operation immediately
        for port in self.__ports: # if there is a signal port, start operation accordingly
            (port_name, port_type, port_file) = port
            if port_name == 'go-signal' and port_type == 'use':
                self.__go_signal = False

        self.__setup_time = 60.0  # time unit; a delay time before starting to run

        #.........................................................................
        # Read input file information if any

        #fin = open(input_full_path_file_name,'r')

        # Configuration member data.

        self.__pyplot_scale = 'linear'
        #self.__ode_integrator = 'scikits.odes' # or 'scipy.integrate' 
        self.__ode_integrator = 'scipy.integrate'

        # Domain specific member data.

        shear_coeff = 0.4  # dimensionless
        Params = namedtuple('Params',['shear_coeff'])
        self.__params = Params( shear_coeff )

        # Setup species in the gas phase.
        species = list()

        air = Specie( name='air', formulaName='Air', phase='gas' )
        air.massCCUnit = 'g/cc'
        air.molarCCUnit = 'mole/cc'
        air.molarMass = 0.3*16*2 + 0.7*14*2

        species.append(air)

        # Quantities in the gas phase.
        quantities = list()

        # Initial position of the wind points; make this zero because data is needed from
        # the use port.
        x_0 = npy.array([0.0,0.0,0.0]) # initial height [m] above ground at 0

        # Initial value for wind velocity at all points. Make this zero too. Use data
        # will change this later.
        # Note: if the z component is positive, wind is blowing upwards.
        #       Gravity points in -z direction.
        self.__box_half_length = 250.0 # wind box [m] 
        self.__box_height      = 100.0 # [m]
        self.__min_core_radius = 2.5 # [m]
        self.__outer_v_theta   = 1.0 # m/s # angular speed
        self.__v_z_0 = 0.50 # [m/s]
        # Wind velocity must be npy.ndarray (see ODE solvers).
        wind_velocity = npy.array([0.0,0.0,0.0])

        # Spatial position of wind points and velocity.
        # One position and velocity quantity per use port to allow for multiple
        # connections to the position use port.
        for port in self.__ports:
            (port_name, port_type, port_file) = port
            if port_name == 'position':

                assert port_type == 'use'

                # Create one position for each use port using the use port_file name
                position = Quantity( name=port_file, formalName='Pos.',
                        unit='m', value=x_0 )
                quantities.append( position )

                # Create a corresponding velocity quantity
                velocity = Quantity( name='velocity@'+port_file, formalName='Veloc.',
                        unit='m/s', value=wind_velocity )
                quantities.append( velocity )

        # Create phase.
        self.__gas_phase = Phase( self.__start_time, time_unit='s', species=species,
                quantities=quantities)

        # Initialize phase.
        air_mass_cc = 0.1 # [g/cc]
        self.__gas_phase.SetValue( 'air', air_mass_cc, self.__start_time )

        return

#*********************************************************************************
# Public member functions
#*********************************************************************************

    def call_ports( self, cortix_time=0.0 ):
        '''
        Transfer data at cortix_time
        '''

        cortix_time *= self.__time_unit_scale  # convert to Wind time unit

        # Provide data to all provide ports.
        self.__provide_data( provide_port_name='velocity', at_time=cortix_time )
        self.__provide_data( provide_port_name='state',  at_time=cortix_time )
        self.__provide_data( provide_port_name='output', at_time=cortix_time )

        # Use data from all use ports.
        self.__use_data( use_port_name='position', at_time=cortix_time )

        return

    def execute( self, cortix_time=0.0, cortix_time_step=0.0 ):
        '''
        Evolve system from cortix_time to cortix_time + cortix_time_step
        '''

        cortix_time      *= self.__time_unit_scale  # convert to Wind time unit
        cortix_time_step *= self.__time_unit_scale  # convert to Wind time unit

        s = 'execute('+str(round(cortix_time,2))+'[min]): '
        self.__log.debug(s)

        self.__evolve( cortix_time, cortix_time_step )

        return

#*********************************************************************************
# Private helper functions (internal use: __)
#*********************************************************************************

    def __provide_data( self, provide_port_name=None, at_time=0.0 ):

        # Access the port file
        port_file = self.__get_port_file( provide_port_name = provide_port_name )

        # Provide data to port files
        if provide_port_name == 'output' and port_file is not None:
            self.__provide_output( port_file, at_time )

        if provide_port_name == 'state' and port_file is not None:
            self.__provide_state( port_file, at_time )

        if provide_port_name == 'velocity' and port_file is not None:
            self.__provide_velocity( port_file, at_time )

        return

    def __use_data( self, use_port_name=None, at_time=0.0 ):

        # Access the port file
        port_file = self.__get_port_file( use_port_name = use_port_name )

        # Use data from port file
        if use_port_name == 'position' and port_file is not None:
            self.__use_position( port_file, at_time )

        return

    def __get_port_file( self, use_port_name=None, provide_port_name=None ):
        '''
        This may return a None port_file
        '''

        port_file = None

        #..........
        # Use ports
        #..........
        if use_port_name is not None:

            assert provide_port_name is None

            for port in self.__ports:

                (port_name,port_type,this_port_file) = port

                if port_name == use_port_name and port_type == 'use':
                    port_file = this_port_file

            if port_file is None:
                return None

            max_n_trials = 50
            n_trials     = 0
            while os.path.isfile(port_file) is False and n_trials <= max_n_trials:
                n_trials += 1
                time.sleep(0.1)

            if n_trials > max_n_trials:
                s = '__get_port_file(): waited ' + str(n_trials) + ' trials for port: ' +\
                    port_file
                self.__log.warn(s)

            assert os.path.isfile(port_file) is True, \
                   'port_file %r not available; stop.' % port_file

        #..............
        # Provide ports
        #..............
        if provide_port_name is not None:

            assert use_port_name is None

            for port in self.__ports:
                (port_name,port_type,this_port_file) = port
                if port_name == provide_port_name and port_type == 'provide':
                    port_file = this_port_file

        return port_file

    def __provide_velocity( self, port_file, at_time ):
        '''
        Provide data while other programs may be trying to read the data. This requires
        a lock. The port file may be completely rewritten or appended to.

        Parameters
        ----------
        port_file: str

        at_time: float

        Returns
        -------
        None
        '''

        import pickle
        from threading import Lock

        lock = Lock()

        with lock:
            # Note that the whole phase is sent out.
            pickle.dump( self.__gas_phase, open(port_file,'wb') )

        print('')
        print('WIND: ALL PHASE PROVIDED')
        print(self.__gas_phase)
        print('')

        s = '__provide_velocity('+str(round(at_time,2))+'[s]): '
        m = 'pickle.dumped velocity.'
        self.__log.debug(s+m)

        return

    def __provide_output_example( self, port_file, at_time ):
        '''
        Provide data that will remain in disk after the simulation ends, persistent
        data. This is an example of how to write a low level plain ASCII table of
        data. This is not used.
        '''
        import datetime

        # If the first time step, write the header of a table data file.
        if at_time == self.__start_time:

            assert os.path.isfile(port_file) is False, \
                'port_file %r exists; stop.' % port_file
            fout = open(port_file,'w')

            # Write file header.
            fout.write('# name:   '+'wind_'+str(self.__slot_id)); fout.write('\n')
            fout.write('# author: '+'cortix.examples.modulib.wind'); fout.write('\n')
            fout.write('# version:'+'0.1'); fout.write('\n')
            today = datetime.datetime.today()
            fout.write('# today:  '+str(today)); fout.write('\n')
            fout.write('#')
            fout.write('%17s'%('Time[sec]'))

            # Mass concentration.
            for specie in self.__gas_phase.GetSpecies():
                fout.write('%18s'%(specie.formulaName+'['+specie.massCCUnit+']'))
            # Quantities.
            for quant in self.__gas_phase.GetQuantities():
                if quant.name == 'position' or quant.name == 'velocity':
                   for i in range(3):
                       fout.write('%18s'%(quant.formalName+str(i+1)+'['+quant.unit+']'))
                else:
                   fout.write('%18s'%(quant.formalName+'['+quant.unit+']'))

            fout.write('\n')
            fout.close()

        # Write history data.
        fout = open(port_file,'a')

        # Wind time.
        fout.write('%18.6e' % (at_time))

        # Mass density.
        for specie in self.__gas_phase.GetSpecies():
            rho = self.__gas_phase.GetValue(specie.name, at_time)
            fout.write('%18.6e'%(rho))

        # Quantities.
        for quant in self.__gas_phase.GetQuantities():
            val = self.__gas_phase.GetValue(quant.name, at_time)
            if quant.name == 'position' or quant.name == 'velocity':
                for v in val:
                    fout.write('%18.6e'%(v))
            else:
                fout.write('%18.6e'%(val))

        fout.write('\n')
        fout.close()

        return

    def __provide_state( self, port_file, at_time ):
        '''
        Provide the internal state of the module. This is typically used by another
        module such as '.cortix.modulib.pyplot'; this XML file type is now deprecated.
        However to have a dynamic plotting option, create this history file in the
        time unit of Cortix; that is, undo the time scaling.
        '''

        import datetime
        import xml.etree.ElementTree as ElementTree
        from threading import Lock

        n_digits_precision = 8

        mutex = Lock()

        # write header
        if at_time == self.__start_time:

            # some thread may be trying to read this output
            mutex.acquire()

            assert os.path.isfile(port_file) is False, 'port_file %r exists; stop.'%\
                    port_file

            a = ElementTree.Element('time-sequence')
            a.set('name','wind_'+str(self.__slot_id)+'-state')

            b = ElementTree.SubElement(a,'comment')
            today = datetime.datetime.today()
            b.set('author','cortix.examples.modulib.wind')
            b.set('version','0.1')

            b = ElementTree.SubElement(a,'comment')
            today = datetime.datetime.today()
            b.set('today',str(today))

            b = ElementTree.SubElement(a,'time')
            b.set('unit',self.__cortix_time_unit)

            # setup the headers
            for specie in self.__gas_phase.species:
                b = ElementTree.SubElement(a,'var')
                formula_name = specie.formulaName
                b.set('name',formula_name)
                unit = specie.massCCUnit
                b.set('unit',unit)
                b.set('legend','Wind_'+str(self.__slot_id)+'-state')
                b.set('scale',self.__pyplot_scale)

            for quant in self.__gas_phase.quantities:
                formal_name = quant.formalName
                if quant.name == 'position' or quant.name == 'velocity':
                    for i in range(3):
                        b = ElementTree.SubElement(a,'var')
                        b.set('name',formal_name+' '+str(i+1))
                        unit = quant.unit
                        b.set('unit',unit)
                        b.set('legend','Wind_'+str(self.__slot_id)+'-state')
                        b.set('scale',self.__pyplot_scale)
                else:
                    b = ElementTree.SubElement(a,'var')
                    b.set('name',formal_name)
                    unit = quant.unit
                    b.set('unit',unit)
                    b.set('legend','Wind_'+str(self.__slot_id)+'-state')
                    b.set('scale',self.__pyplot_scale)

            # write values for all variables
            b = ElementTree.SubElement(a,'timeStamp')
            b.set('value',str(round(at_time/self.__time_unit_scale,n_digits_precision)))

            values = list()

            for specie in self.__gas_phase.species:
                val = self.__gas_phase.GetValue( specie.name, at_time )
                values.append( val )

            for quant in self.__gas_phase.quantities:
                val = self.__gas_phase.GetValue( quant.name, at_time )
                if quant.name == 'position' or quant.name == 'velocity':
                    for v in val:
                        values.append( math.fabs(v) ) # pyplot can't take negative values
                else:
                    values.append( val )

            # flush out data
            text = str()
            for value in values:
                text += str(round(value,n_digits_precision)) + ','

            text = text[:-1]

            b.text = text

            tree = ElementTree.ElementTree(a)

            tree.write( port_file, xml_declaration=True, encoding="unicode",
                    method="xml" )

            mutex.release()

        #-------------------------------------------------------------------------
        # if not the first time step then parse the existing history file and append
        else:

            # some thread may be trying to read this output
            mutex.acquire()

            tree = ElementTree.parse( port_file )
            root_node = tree.getroot()

            a = ElementTree.Element('timeStamp')
            a.set('value',str(round(at_time/self.__time_unit_scale,n_digits_precision)))

            # all variables values
            values = list()

            for specie in self.__gas_phase.species:
                val = self.__gas_phase.GetValue( specie.name, at_time )
                values.append( val )
            for quant in self.__gas_phase.quantities:
                val = self.__gas_phase.GetValue( quant.name, at_time )
                if quant.name == 'position' or quant.name == 'velocity':
                    for v in val:
                        values.append( math.fabs(v) ) # pyplot can't take negative values
                else:
                    values.append( val )

            # flush out data
            text = str()
            for value in values:
                text += str(round(value,n_digits_precision)) + ','

            text = text[:-1]

            a.text = text

            root_node.append(a)

            tree.write( port_file, xml_declaration=True, encoding="unicode", method="xml" )

            mutex.release()

        return

    def __use_position( self, port_file, at_time ):
        '''
        This file usage requires a locking mechanism before the reading operation
        so the program writing the file does not delete it or modify it in the
        middle of the reading process.
        '''

        import pickle
        from threading import Lock
        import numpy as npy

        lock = Lock()

        found = False
        while found is False:

            try:
                lock.acquire()
                # Data specs: tuple(Quantity,float)
                # The Quantity object must have a Pandas.Series as value for the history
                # values.

                (position_history,time_unit) = pickle.load( open(port_file,'rb') )

                assert time_unit == 's'
                assert isinstance(position_history,Quantity)

                loc = position_history.value.index.get_loc(at_time,method='nearest',
                        tolerance=1e-2)
                time_stamp = position_history.value.index[loc]


                assert abs(time_stamp - at_time) <= 1e-2
                found = True
                lock.release()

                print('')
                print('WIND: DROPLET POSITION RECEIVED')
                print(position_history)
                print('')

            except:
                lock.release()
                s = '__use_position('+str(round(at_time,2))+'[s]): '
                m = port_file+' does not have data yet. Retrying ...'
                self.__log.debug(s+m)
                self.__log.debug(s)

        s = '__use_position('+str(round(at_time,2))+'[s]): pickle.loaded position.'
        self.__log.debug(s)

        position = position_history.value.loc[time_stamp]
        assert isinstance(position,npy.ndarray)

        assert self.__gas_phase.has_time_stamp(at_time)

        self.__gas_phase.SetValue( actor=port_file, value=position,
                try_time_stamp=time_stamp )

        # Update the wind velocity at the position
        velocity = self.__vortex_velocity( position )

        self.__gas_phase.SetValue( actor=port_file, value=velocity,
                try_time_stamp=time_stamp )

        return

    def __evolve( self, cortix_time=0.0, cortix_time_step=0.0 ):
        r'''
         '''

        values = self.__gas_phase.GetRow( cortix_time ) # values at previous time

        at_time = cortix_time + cortix_time_step

        self.__gas_phase.AddRow( at_time, values ) # repeat values for current time

        # Compute the wind velocity at the given external position
        # Using a vortex flow model.

        for quant in self.__gas_phase.quantities:
            name = quant.name
            formal_name = quant.formal_name
            if formal_name == 'Pos.':
                position = self.__gas_phase.GetValue(name,cortix_time)

                wind_velocity = self.__vortex_velocity( position )

                velo_name = 'velocity@'+name
                #print(velo_name)
                #print(position)
                #print(wind_velocity)

                #wind_velocity = self.__gas_phase.GetValue(velo_name,cortix_time)

                self.__gas_phase.SetValue(velo_name,wind_velocity,at_time)

        return

    def __vortex_velocity( self, position ):
        '''
        Computes the velocity of the wind at a given position.

        Parameters
        ----------
        position: numpy.ndarray(3)

        Returns
        -------
        wind_velocity: numpy.ndarray(3)
        '''

        # Compute the wind velocity at the given external position
        # Using a vortex flow model.
        box_half_length = self.__box_half_length
        min_core_radius = self.__min_core_radius
        outer_v_theta   = self.__outer_v_theta

        outer_cylindrical_radius = math.hypot(box_half_length,box_half_length)
        circulation = 2*math.pi * outer_cylindrical_radius * outer_v_theta # m^2/s

        core_radius = min_core_radius

        x = position[0]
        y = position[1]
        z = position[2]

        relax_length = self.__box_height/2.0
        z_relax_factor = math.exp(-(self.__box_height-z)/relax_length)
        v_z = self.__v_z_0 * z_relax_factor

        cylindrical_radius = math.hypot(x,y)
        azimuth = math.atan2(y,x)

        v_theta = (1 - math.exp(-cylindrical_radius**2/8/core_radius**2) ) *\
                   circulation/2/math.pi/max(cylindrical_radius,min_core_radius) *\
                   z_relax_factor

        v_x = - v_theta * math.sin(azimuth)
        v_y =   v_theta * math.cos(azimuth)

        wind_velocity = npy.array([v_x,v_y,v_z])

        return wind_velocity

    def __read_manifesto( self, xml_tree_file ):
        '''
        Parse the manifesto
        '''

        assert isinstance(xml_tree_file, str)

        # Read the manifesto
        xml_tree = XMLTree( xml_tree_file=xml_tree_file )

        assert xml_tree.tag == 'module_manifesto'

        assert xml_tree.get_attribute('name') == 'wind'

        # List of (port_name, port_type, port_mode, port_multiplicity)
        __ports = list()

        self.__port_diagram = 'null-module-port-diagram'

        # Get manifesto data  
        for child in xml_tree.children:
            (elem, tag, attributes, text) = child

            if tag == 'port':

                text = text.strip()

                assert len(attributes) == 3, "only <= 3 attributes allowed."

                tmp = dict()  # store port name and three attributes

                for attribute in attributes:
                    key = attribute[0].strip()
                    val = attribute[1].strip()

                    if key == 'type':
                        assert val == 'use' or val == 'provide' or val == 'input' or \
                            val == 'output', 'port attribute value invalid.'
                        tmp['port_name'] = text  # port_name
                        tmp['port_type'] = val   # port_type
                    elif key == 'mode':
                        file_value = val.split('.')[0]
                        assert file_value == 'file' or file_value == 'directory',\
                            'port attribute value invalid.'
                        tmp['port_mode'] = val
                    elif key == 'multiplicity':
                        tmp['port_multiplicity'] = int(val)  # port_multiplicity
                    else:
                        assert False, 'invalid port attribute: %r=%r. fatal.'%\
                                (key,val)

                assert len(tmp) == 4
                store = (tmp['port_name'], tmp['port_type'], tmp['port_mode'],
                         tmp['port_multiplicity'])

                # (port_name, port_type, port_mode, port_multiplicity)
                __ports.append(store)

                # clear
                tmp   = None
                store = None

            if tag == 'diagram':

                self.__port_diagram = text

            if tag == 'ascii_art':

                self.__ascii_art = text

        return

#======================= end class Wind: =========================================
