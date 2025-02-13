#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment
# https://cortix.org
"""
Phase *history* container. When you think of a phase value, think of that value at
a specific point in time. This container holds the historic data of a phase;
its species and quantities. This implementation treats access of time stamps within
a tolerance. All searches for time stamped values are subjected to an approximation
of the time stamp to avoid storing values too close to each other in time, and/or to
return the closest value in time searched or no value if none can be found according
to the tolerance.

Background
----------
TODO: ATTENTION:
The species (list of Species) AND quantities (list of Quantity) data members
have ARBITRARY density values either at an arbitrary point in the history or at
no point in the history. This needs to be removed in the future to avoid confusion.

To obtain history values, associated to the phase, at a particular point in time,
use the GetValue() method to access the history data frame (pandas) via columns and
rows. ALERT: The corresponding values in species and quantities are OVERRIDEN and NOT to
be used through the phase interface.

NOTE: If you find yourself creating an inflow mass flowrate Quantity in your Phase,
call the resulting object a "stream." Outflow rates can be created on the fly from
concentrations and volumetric flowrates, hence they should not be a Quantity. Use
quantity for temperature, pressure, etc.

Author: Valmor F. de Almeida dealmeidav@ornl.gov; vfda
Sat Sep  5 01:26:53 EDT 2015

Cortix: a program for system-level modules coupling, execution, and analysis.
"""

import os, io
from copy import deepcopy
import time
import datetime

import numpy as np
import pandas

import matplotlib
#needed? matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter

from cortix.support.species   import Species
from cortix.support.quantity import Quantity

class PhaseNew:
    """ Phase `history` container.
    A `Phase` consists of `Species` and `Quantities` varying with time. This container is meant to
    reproduce the basic idea of a material phase.
    """

    def __init__(self, name = None,
                 time_stamp = None,
                 time_unit  = None,
                 species    = None,
                 quantities = None
                ):
        #TODO
        """
        Sometimes an empty Phase object is created by user code. This case needs
        adequate logic for None types.
        Note on usage: when passing quantities, do set the value argument explicitly
        to help define the type and avoid set_value() errors with Pandas. This is
        to be investigated later. Also, the usage of a DataFrame needs to be re-evaluated.
        Maybe better to use a Quantity object and a Species object with a Pandas Series
        history as a value to avoid the existance of a value in Quantity and a value
        in Phase that are not in sync.
        """

        if not name:
            self.name = self.__class__.__name__
        else:
            self.name = name

        if time_stamp is None:
            time_stamp = 0.0 # default type is float
        else:
            if type(time_stamp) == int:
                time_stamp = float(time_stamp)
            assert isinstance(time_stamp, float), 'time_stamp=%r, type(time_stamp)=%r'%(time_stamp,
                                                                                        type(time_stamp))
            self.__time_stamp = time_stamp

        if time_unit is None:
            self.__time_unit = 's' # second
        else:
            assert isinstance(time_unit, str)
            self.__time_unit = time_unit

        if species is not None:
            assert isinstance(species, list)
            for each_species in species:
                assert isinstance(each_species, Species)

        if quantities is not None:
            assert isinstance(quantities, list)
            for quant in quantities:
                assert isinstance(quant, Quantity)

        # List of species and quantities objects; columns of data frame are named
        # by objects.
        # A new object held by a Phase() object
        if species is not None:
            self.__species = deepcopy(species)
        else:
            self.__species = None

        # A new object held by a Phase() object
        if quantities is not None:
            self.__quantities = deepcopy(quantities)
        else:
            self.__quantities = None

        names = list()

        if species is not None:
            for spc in self.__species:
                names.append(spc.name)

        if quantities is not None:
            for quant in self.__quantities:
                names.append(quant.name)
                # quant can be a numpy array for example...
                # rethink this...
                #quant.value = 0.0    # clear these values
                                     # todo: eliminate them from Quantity in the future

        # Table data phase without data type assigned; this is left to the user
        # Time stamps will always be float
        self.__df = pandas.DataFrame(index=[float(time_stamp)], columns=names)

        # This is meant to be the value of species concentration; a float type
        if species is not None:
            for spc in species:
                self.__df.loc[time_stamp, spc.name] = 0.0

        # The value of the quantity could be any type
        if quantities is not None:
            for quant in quantities:
                self.__df.loc[time_stamp, quant.name] = quant.value
                #self.__df.fillna(0.0, inplace=True)  # dtype defaults to float

        return

    def has_time_stamp(self, try_time_stamp):
        '''
        Checks to see if try_time_stamp exists in the phase history.

        Parameters
        ----------
        try_time_stamp:
        '''

        time_stamp = self.__get_time_stamp(try_time_stamp)

        if time_stamp is not None:
            return True
        else:
            return False

    def __get_time_unit(self):
        '''
        Returns the time unit of the `Phase.`

        Returns
        -------
        time_unit: str
        '''

        return self.__time_unit
    time_unit = property(__get_time_unit,None,None,None)

    def __get_time_stamps(self):
        '''
        Get all time stamps in the index of the data frame.

        Returns
        -------
        time_stamps: list
        '''

        return list(self.__df.index)  # return all time stamps
    time_stamps = property(__get_time_stamps, None, None, None)

    def __get_species_list(self):
        '''
        Returns every single species in the phase history.

        Returns
        -------
        species: list
        '''

        return self.__species
    species = property(__get_species_list, None, None, None)

    def __get_quantities(self):
        '''
        Returns the list of `Quantities`. The values in each `Quantity` are
        synchronized with the `Phase` data frame.

        Returns
        -------
        quantities: list
        '''

        for quant in self.__quantities:
          tmp = self.get_quantity(quant.name) # handy way to synchronize the whole list

        return self.__quantities
    quantities = property(__get_quantities, None, None, None)

    def __get_actors(self):
        """Returns a list of names of all actors in the phase history.

        Returns
        -------
        list(self.__df.colums): list

        """

        return list(self.__df.columns)  # return all names in order
    actors = property(__get_actors, None, None, None)

    def __get_df(self):
        '''
        Die hard access.
        '''
        return self.__df
    df = property(__get_df,None,None,None)

    def get_species(self, name):
        '''
        Returns the species specified by name if it exists,
        or None if it doesn't.

        Parameters
        ----------
        name: str

        Returns
        -------
        specie: str

        '''

        assert name in self.__df.columns, 'name %r not in %r'%\
                (name,self.__df.columns)

        if self.__species:
            for species in self.__species:
                if species.name == name:
                    return species
        return None

    def get_species_concentration(self, name, try_time_stamp=None):
        '''
        Returns the species concentration at `try_time_stamp`.

        Parameters
        ----------
        name: str

        try_time_time_stamp: float

        Returns
        -------
        concentration: float

        '''
        return self.get_value(name, try_time_stamp)

    def set_species_id(self, name, val):
        '''
        Sets the flag of a species "name" equal to val.

        Parameters
        ----------
        name: str
        val: int

        '''

        assert name in self.__df.columns, 'name %r not in %r'%(name,self.__df.columns)

        for species in self.__species:
            if species.name == name:
                species.flag = val
                return

    def get_quantity(self, name):
        """Get the quantity with `name`.

        Parameters
        ----------
        name: str

        Returns
        -------
        Quantity or None
        """

        assert name in self.__df.columns, 'name %r not in %r'%(name,self.__df.columns)

        if self.__quantities:
            for quant in self.__quantities:
                if quant.name == name:
                   return quant
        return None

    def get_quantity_value(self, name, try_time_stamp=None):
        """Get the quantity value with given `name` at time closest to `try_time_stamp`.

        This is up to a time tolerance. If no time stamp is passed, the value at the last time stamp is
        returned.

        Parameters
        ----------
        name: str
        try_time_stamp: float, int or None
            Time stamp of desired quantity value. Default: None, returns the
            value at the last time stamp.

        Returns
        -------
        quant.value: type of Quantity.value
        """

        assert name in self.__df.columns, 'name %r not in %r'%(name,self.__df.columns)

        time_stamp = self.__get_time_stamp(try_time_stamp)

        if self.__quantities:
            for quant in self.__quantities:
                if quant.name == name:
                    # Update value in quantity with history value
                    quant.value = self.__df.loc[time_stamp, name] # labels' access mode
                    return quant.value  # return quantity value syncronized with the phase

    def get_quantity_history(self, name):
        '''
        Creates a Quantity `name` history. This will create a fully qualified
        Quantity object and return to the caller. The function is typically
        needed for data output to a file through `pickle`. Since the value
        attribute of a quantity can be any data structure, a time-series is
        built on the fly and stored in the value attribute. In addition the
        time unit is added to the final return value as a tuple.

        Parameters
        ----------
        name: str

        Returns
        -------
        quant_history: tuple(Quantity,str) or None
        '''

        assert name in self.__df.columns, 'name %r not in %r'%(name,self.__df.columns)

        if self.__quantities is not None:
            for quant in self.__quantities:
                if quant.name == name:
                    quant_history = deepcopy(quant)
                    quant_history.value = deepcopy(self.__df[name]) # whole data frame index series
                    return (quant_history, self.__time_unit) # return tuple
        else:
            return (None, self.__time_unit)

        return None

    def add_single_species(self, new_species, discard_new_duplicate=False):
        """
        Adds a new specie object to the phase history. See species.py for
        details on the Species class.

        Parameters
        ----------
        new_species: obj

        discard_new_duplicate: if there exists a species with the same name, discard the new one.

        """

        assert isinstance(new_species, Species)

        if not discard_new_duplicate:
            assert new_species.name not in list(self.__df.columns), \
                   'new_species: %r exists. Current names: %r'%(new_species, self.__df.columns)

        if self.__species is not None: # self.__species could be empty
            species_formulae = [specie.formula_name for specie in self.__species]

            if not discard_new_duplicate:
                assert new_species.formula_name not in species_formulae

        if self.__species is not None: # self.__species could be empty
            species_names = [spc.name for spc in self.__species]

            if not discard_new_duplicate:
                assert new_species.name not in species_names

            if new_species.name in species_names:
                return

        if self.__species is not None:
            self.__species.append(deepcopy(new_species))
        else:
            self.__species = [deepcopy(new_species)] # create a list here

        new_name = new_species.name

        new_column = pandas.DataFrame(index=list(self.__df.index), columns=[new_name])
        tmp = self.__df
        df = tmp.join(new_column, how='outer')

        self.__df = df.infer_objects().fillna(0).astype(float) # for species fill concentration with float as default

    def add_quantity(self, new_quant):
        '''
        Adds a new quantity object to the dataframe. See quantity.py for more
        details on the quantity class.

        Parameters
        ----------
        new_quant: object
        '''

        assert isinstance(new_quant, Quantity)
        assert new_quant.name not in list(self.__df.columns), \
               'quantity: %r exists. Current names: %r' % \
               (new_quant, self.__df.columns)
        quant_formal_names = [quant.formal_name for quant in self.__quantities]
        assert new_quant.formal_name not in quant_formal_names
        self.__quantities.append(new_quant)
        new_name = new_quant.name

        # create a col with object data type; user must fill out column
        col = pandas.DataFrame( index=list( self.__df.index), columns=[new_name],
                                dtype=object )
        tmp = self.__df
        df  = tmp.join(col, how='outer')
        #self.__df = df.fillna(new_quant.value)

    def add_row(self, try_time_stamp, row_values):
        '''
        Adds a row to the `DataFrame`, with a `timestamp` equal to `try_time_stamp` and
        row values equal to `row_values`. The length of `row_values` must match the
        number of columns in the data frame.

        Parameters
        ----------
        try_time_stamp: float
        row_values: list

        '''
        assert try_time_stamp not in self.__df.index, 'already used time_stamp: %r'%\
                (try_time_stamp)
        assert isinstance(row_values, list)

        time_stamp = self.__get_time_stamp( try_time_stamp )
        assert time_stamp is None, 'already used time_stamp: %r'%(try_time_stamp)
        time_stamp = try_time_stamp

        assert len(row_values) == self.__df.columns.size

        # create a row with object data type; users row_values data define data type
        row = pandas.DataFrame( index=[time_stamp],
                                columns=list( self.__df.columns ), dtype=object )

        for (col,v) in zip(row.columns, row_values):
            row.loc[time_stamp,col] = v

        frames = [self.__df, row]
        self.__df = pandas.concat(frames)
        return

    def get_row(self, try_time_stamp=None):
        '''
        Returns an entire row of the phase dataframe. A row is a series of
        values that are all at the same time stamp.

        Parameters
        ----------
        try_time_stamp: float

        Returns
        -------
        list(self.__df.loc[time_stamp, :]): list

        '''
        time_stamp = self.__get_time_stamp( try_time_stamp )
        assert time_stamp is not None, 'missing try_time_stamp: %r'%(try_time_stamp)
        return list(self.__df.loc[time_stamp, :])

    def get_column(self, actor):
        '''
        Returns an entire column of data. A column is the entire history
        of data associated with a specific actor.

        Parameters
        ----------
        actor: str

        Returns
        -------
        list(self.__df.loc[:, actor]): list

        '''
        assert isinstance(actor, str)
        assert actor in self.__df.columns, 'actor %r not in %r'%(actor, self.__df.columns)
        return list(self.__df.loc[:, actor])

    def scale_row(self, try_time_stamp, value):
        '''
        Multiplies all of the data in a row (except time stamp) by a scalar
        value.

        Parameters
        ----------
        try_time_stamp: float
        value: float

        '''
        assert isinstance(try_time_stamp, int) or isinstance(try_time_stamp, float)
        time_stamp = self.__get_time_stamp( try_time_stamp )
        assert time_stamp is not None, 'missing try_time_stamp: %r'%(try_time_stamp)
        #assert isinstance(value, int) or isinstance(value, float)
        self.__df.loc[time_stamp, :] *= value
        return

    def ClearHistory(self, value=0.0):
        '''
        Set species and quantities of history to a given value
        (default to zero value), all time stamps are preserved.

        Parameters
        ----------
        value: float

        '''
        assert isinstance(value, int) or isinstance(value, float)
        self.__df.loc[:, :] = value

        return

    def ResetHistory(self, try_time_stamp=None, value=None):
        '''
        Set species and quantities of history to a given value
        (default to zero value) only one time stamp is preserved (default to
        last time stamp).

        Parameters
        ----------
        try_time_stamp: float
        value: float

        '''
        if value is not None:
           assert isinstance(value, int) or isinstance(value, float) or \
                  isinstance(value, np.ndarray)

        if try_time_stamp is not None:
           assert isinstance(try_time_stamp, int) or isinstance(try_time_stamp, float)

        time_stamp = self.__get_time_stamp( try_time_stamp )
        assert time_stamp is not None, 'missing try_time_stamp: %r'%(try_time_stamp)

        values = self.GetRow(time_stamp)  # save values

        columns = list(self.__df.columns)
        assert len(columns) == len(values), 'FATAL: oops internal error.'

        self.__df = pandas.DataFrame( index=[time_stamp], columns=columns )
        self.__df.fillna( 0.0, inplace=True )

        if value is None:
            for v in values:
                idx = values.index(v)
                self.__df.loc[time_stamp, columns[idx]] = v  # restore values
        else:
            self.__df.loc[time_stamp, :] = value   # set user-given value

        return

    def get_value(self, actor, try_time_stamp=None):
        """Returns the value associated with a specified actor at a specified time stamp.

        Parameters
        ----------
        actor: str
        try_time_stamp: float
            Default is None which returns the last time stamp.

        Returns
        -------
        self.__df.loc[time_stamp, actor]: any

        """
        assert isinstance(actor, str)
        assert actor in self.__df.columns, 'actor %r not in %r'%(actor,self.__df.columns)

        if try_time_stamp is not None:
           assert isinstance(try_time_stamp, int) or isinstance(try_time_stamp, float)

        time_stamp = self.__get_time_stamp(try_time_stamp)
        assert time_stamp is not None, 'missing try_time_stamp: %r'%(try_time_stamp)

        return self.__df.loc[time_stamp, actor]

    def set_value(self, actor, value, try_time_stamp=None):
        """Set value of actor in the data frame at a give time stamp.

        Note
        ----
        The value in the original species or quantity container is not modified.
        """
        assert isinstance(actor, str)
        assert actor in self.__df.columns, 'actor = %r not in data frame'%actor

        if try_time_stamp is not None:
            assert isinstance(try_time_stamp, int) or isinstance(try_time_stamp, float)

        time_stamp = self.__get_time_stamp(try_time_stamp)
        assert time_stamp is not None, 'missing try_time_stamp: %r'%(try_time_stamp)

        # Note: user value could have a different type than other column values.
        # If there is a type change, this will not be checked; user has been advised.
        self.__df.loc[time_stamp, actor] = deepcopy(value)

        return

    def write_html(self, filename):
        """Convert the `Phase` container into an HTML file.

        Parameters
        ----------
        filename: str

        """
        assert isinstance(filename, str)

        tmp = pandas.DataFrame(self.__df)
        column_names = tmp.columns

        if self.__species:
            species_names = [species.name for species in self.__species]
        else:
            species_names = list()

        if self.__quantities:
            quantity_names = [quantity.name for quantity in self.__quantities]
        else:
            quantity_names = list()

        for col in column_names:
            if col in species_names:
                idx = species_names.index(col)
                species = self.__species[idx]
                tmp.rename(columns={col: species.formula_name}, inplace=True)
            elif col in quantity_names:
                idx = quantity_names.index(col)
                quant = self.__quantities[idx]
                tmp.rename(columns={col: col + '[' + quant.unit + ']'}, inplace=True)
            else:
                assert False, 'oops fatal.'

        tmp.to_html(filename)

    def __str__(self):
        s = '\n\t **Phase()**: name=%s;' + \
            '\n\t time unit: %s;' + \
            '\n' + \
            '\n\t *quantities*: %s;' + \
            '\n' + \
            '\n\t *species*: %s;' + \
            '\n' + \
            '\n\t *history* # time_stamps=%s;' + \
            '\n\t *history end* @ %s;' + \
            '\n%s'
        return s % (self.name,
                self.__time_unit,
                self.__quantities,
                self.__species,
                len(self.__df.index),
                self.__df.index[-1],
                self.__df.loc[self.__df.index[-1], :])

    def __repr__(self):
        s = '\n\t **Phase()**: name=%s;' + \
            '\n\t time unit: %s;' + \
            '\n' + \
            '\n\t *quantities*: %s;' + \
            '\n' + \
            '\n\t *species*: %s;' + \
            '\n' + \
            '\n\t *history* # time_stamps=%s;' + \
            '\n\t *history end* @ %s;' + \
            '\n%s'
        return s % (self.name,
                self.__time_unit,
                self.__quantities,
                self.__species,
                len(self.__df.index),
                self.__df.index[-1],
                self.__df.loc[self.__df.index[-1], :])

    def __get_time_stamp(self, try_time_stamp=None):
        """ Helper method for finding the closest time stamp to `try_time_stamp` in the phase history.

        The pandas index container used for storing
        float data type time stamps will return the nearest time stamp up to a
        tolerance. Whether the time index has one value, this function will
        inspect for the proximity to that value.

        Parameters
        ----------
        try_time_stamp: float, int or None
            Default: None will return the last time stamp.

        Returns
        -------
        self.__df.index[loc]: float or None
            Will return None if no time stamp within tolerance is found.

        """
        import numpy as np

        tol = 1.0e-3

        if try_time_stamp is None:
            return self.__df.index[-1]
        else:
            time_stamps = np.array(self.__df.index)
            if time_stamps.size >= 2:
               tol = 1.0e-3 * np.diff(time_stamps).mean() # 1e-3 * the mean delta t

            # deprecated (no exception thrown)
            #try: # abs(index_value - try_time_stamp) <= tolerance
            #    loc = self.__df.index.get_loc(try_time_stamp, method='nearest', tolerance=tol)
            #except KeyError: # no value found within tol
            #    return None
            #else:
            #    return  self.__df.index[loc]

            # abs(index_value - try_time_stamp) <= tolerance
            loc = self.__df.index.get_indexer([try_time_stamp], method='nearest', tolerance=tol)[0]
            if loc == -1:
                return None
            else:
                return  self.__df.index[loc]

    def plot_species(self, name, scaling=[1.0, 1.0] , title=None, xlabel='Time [s]',
                     ylabel='y', legend=None, filename_tag=None, figsize=[6,5],
                     dpi=100 ):

        if legend is not None:
            assert isinstance(legend, str)

        fig,ax=plt.subplots(1, figsize=figsize)

        x = np.array([t for t in self.__df.index])
        x *= float(scaling[0])

        y = np.array(self.get_column(name), dtype=np.float64)
        y *= float(scaling[1])

        yformatter = ScalarFormatter(useMathText=True,useOffset=True)
        yformatter.set_powerlimits((15, 5))
        ax.yaxis.set_major_formatter(yformatter)

        if legend:
            ax.plot(x, y, 'b-', label=legend)
        else:
            ax.plot(x, y, 'b-')

        ax.set_xlabel(r''+xlabel,fontsize=16)
        ax.set_ylabel(r''+ylabel,fontsize=16,color='black')
        ax.tick_params(axis='y',labelsize=14)
        ax.tick_params(axis='x',labelsize=14)

        if legend:
            ax.legend(loc='best',fontsize=12)

        if title:
            ax.set_title(title)
        elif self.get_species(name).info:
            ax.set_title(self.get_species(name).info,fontsize=14)

        ax.grid(True)

        fig_name = name+'-'+self.name+'-phase-plot-'

        if filename_tag:
            fig_name += filename_tag

        fig.savefig(fig_name+'.png', dpi=dpi, format='png')
        plt.close(fig)

        return

    def plot(self, actors=None, title=None, name='phase-plot-null-name',
             var_unit=None,
             x_scaling=1,
             legend=None, nrows=2, ncols=2, figsize=[6,5], show=False, dpi=300):
        """Plot assistant for all phase container actors.

        This needs a full review...it was copied here in a hurry.

        Parameters
        ----------
        actors: list(str)
            List of names of quantities or species in the phase. Defaults to all actors being plotted.

        title: str
            Title of the dashboard.

        name: str
            Stem of the filename of the plots saved to file.

        var_unit: str
            An alternative unit for the species concentration. The quantities variable in the
            phase will have their own unit. Default: g/L

        show: boolean
            When used in an interactive session, show the plots on display.
        """

        if len(self.__df.columns) == 0:
            return

        if actors is None:
            actors = self.__df.columns
        else:
            assert isinstance(actors, list)

        if title is not None:
            assert isinstance(title, str)

        if var_unit is not None:
            assert isinstance(var_unit, str)

        num_var = len(actors)

        today = datetime.datetime.today().strftime("%d%b%y %H:%M:%S")

        lead_name = name

        fig_num = None

        # Loop over variables and assign to the dashboards
        i_dash = 0

        for i_var in range(num_var):

            col_name = actors[i_var]

            # if multiple of nrows*ncols start new dashboard
            if i_var % (nrows*ncols) == 0:

                if i_var != 0:  # flush any current figure
                    fig_name = lead_name+'-'+self.name+'-phase-plot-' + \
                            str(i_dash).zfill(2)
                    fig.savefig(fig_name+'.png', dpi=dpi, format='png')
                    #plt.close(fig_num)

                    #pickle.dump( fig, open(fig_name+'.pickle','wb') )

                    i_dash += 1

                fig_num = str(np.random.random()) + '.' + str(i_dash)
                fig = plt.figure(num=fig_num, figsize=figsize)

                gs = gridspec.GridSpec(nrows, ncols)
                #gs.update(left=0.08, right=0.98, wspace=0.4, hspace=0.4)
                gs.update(left=0.13, right=0.98, wspace=0.4, hspace=0.6)

                axlst = list()

                nPlotsNeeded = num_var - i_var
                count = 0
                for i in range(nrows):
                    for j in range(ncols):
                        axlst.append(fig.add_subplot(gs[i, j]))
                        count += 1
                        if count == nPlotsNeeded:
                            break
                    if count == nPlotsNeeded:
                        break

                axes = np.array(axlst)

                if title is None:
                    text = today + ': Cortix.Phase.Plot'
                else:
                    text = title
                fig.text(.5, .95, text, horizontalalignment='center', fontsize=16)

                axs = axes.flat

                axId = 0

            # end of: if i_var % nrows*ncols == 0: # if a multiple of nrows*ncols
            # start a new dashboard

            if var_unit is None:
                var_unit = 'g/L'

            ax = axs[axId]
            axId += 1

            species = self.get_species(col_name)

            if species:
                #latex_name = species.latex_name  # not working
                latex_name = species.name
                var_unit = var_unit
                info_str = species.info

            quantity = self.get_quantity(col_name)

            if quantity:
                latex_name = quantity.latex_name
                var_unit = quantity.unit
                info_str = quantity.info

            # Sanity check assumes all species come before all quantities in the data frame.
            if species:
                if i_var <= len(self.__species):
                    assert self.__species[i_var].name == self.__df.columns[i_var]
            if quantity and species:
                if i_var > len(self.__species):
                    assert self.__quantities[i_var].name == self.__df.columns[i_var]
            elif quantity: # only quantities exist and species is empty
                if i_var > len(self.__species):
                    assert self.__quantities[i_var].name == self.__df.columns[i_var], \
                       'ivar=%r; __quant[i]=%r; __df.col[i]=%r; __quant=%r; __df.col=%r'%(i_var,
                       self.__quantities[i_var].name, self.__df.columns[i_var], self.__quantities,
                       self.__df.columns)

            '''
            if varUnit == 'gram':
                varUnit = 'g'
            if varUnit == 'gram/min':
                varUnit = 'g/min'
            if varUnit == 'gram/s':
                varUnit = 'g/s'
            if varUnit == 'gram/m3':
                varUnit = 'g/m3'
            if varUnit == 'gram/L':
                varUnit = 'g/L'
            if varUnit == 'sec':
                varUnit = 's'
            '''

            varLegend = legend
            varScale  = 'linear-linear'

            assert varScale == 'log' or varScale == 'linear' or varScale == 'log-linear' \
                or varScale == 'linear-log' or varScale == 'linear-linear' or \
                varScale == 'log-log'

            time_unit = 's'

            if time_unit == 'minute':
                time_unit = 'min'

            x = np.array([i for i in self.__df.index])
            #x *= x_scaling

            if (varScale == 'linear' or varScale == 'linear-linear' or \
                varScale == 'linear-log') and x.max() >= 60.0:
                if x.max() >= 60.0 * 60.0:
                    x /= 60*60
                    if time_unit == 's' or time_unit == 'second':
                        time_unit = 'h'
                elif x.max() >= 60.0:
                    x /= 60.0
                    if time_unit == 'min':
                        time_unit = 'h'
                    if time_unit == 'second' or time_unit=='s':
                        time_unit = 'min'

            y = np.array(self.__df[col_name]) # convert to numpy ndarray

            '''
            if (y.max() >= 1e3 or y.min() <= -1e3) and varScale != 'linear-log' and \
                    varScale != 'log-log' and varScale != 'log':
                y /= 1e3
                if varUnit == 'gram' or varUnit == 'g':
                    varUnit = 'kg'
                if varUnit == 'L':
                    varUnit = 'kL'
                if varUnit == 'cc':
                    varUnit = 'L'
                if varUnit == 'Ci':
                    varUnit = 'kCi'
                if varUnit == 'W':
                    varUnit = 'kW'
                if varUnit == 'gram/min' or varUnit == 'g/min':
                    varUnit = 'kg/min'
                if varUnit == 'gram/s' or varUnit == 'g/s':
                    varUnit = 'kg/s'
                if varUnit == 'gram/m3' or varUnit == 'g/m3':
                    varUnit = 'kg/m3'
                if varUnit == 'gram/L' or varUnit == 'g/L':
                    varUnit = 'kg/L'
                if varUnit == 'W/L':
                    varUnit = 'kW/L'
                if varUnit == 'Ci/L':
                    varUnit = 'kCi/L'
                if varUnit == '':
                    varUnit = 'x1e3'
                if varUnit == 'L/min':
                    varUnit = 'kL/min'
                if varUnit == 'Pa':
                    varUnit = 'kPa'
                if varUnit == 's':
                    varUnit = 'ks'
                if varUnit == 'm':
                    varUnit = 'km'
                if varUnit == 'm/s':
                    varUnit = 'km/s'

            if (y.max() < 1e-6 and y.min() > -1e-6) and varScale != 'linear-log' and \
                    varScale != 'log-log' and varScale != 'log':
                y *= 1e9
                if varUnit == 'gram' or varUnit == 'g':
                    varUnit = 'ng'
                if varUnit == 'cc':
                    varUnit = 'n-cc'
                if varUnit == 'L':
                    varUnit = 'nL'
                if varUnit == 'W':
                    varUnit = 'nW'
                if varUnit == 'Ci':
                    varUnit = 'nCi'
                if varUnit == 'gram/min' or varUnit == 'g/min':
                    varUnit = 'ng/min'
                if varUnit == 'gram/s' or varUnit == 'g/s':
                    varUnit = 'ng/s'
                if varUnit == 'gram/m3' or varUnit == 'g/m3':
                    varUnit = 'ng/m3'
                if varUnit == 'gram/L' or varUnit == 'g/L':
                    varUnit = 'ng/L'
                if varUnit == 'W/L':
                    varUnit = 'nW/L'
                if varUnit == 'Ci/L':
                    varUnit = 'nCi/L'
                if varUnit == 'L/min':
                    varUnit = 'nL/min'
                if varUnit == 'Pa':
                    varUnit = 'nPa'
                if varUnit == 's':
                    varUnit = 'ns'
                if varUnit == 'm/s':
                    varUnit = 'nm/s'

            if (y.max() >= 1e-6 and y.max()  <  1e-3) or \
               (y.min() > -1e-3 and y.min() <= -1e-6) and varScale != 'linear-log' and \
                    varScale != 'log-log' and varScale != 'log':
                y *= 1e6
                if varUnit == 'gram' or varUnit == 'g':
                    varUnit = 'ug'
                if varUnit == 'cc':
                    varUnit = 'u-cc'
                if varUnit == 'L':
                    varUnit = 'uL'
                if varUnit == 'W':
                    varUnit = 'uW'
                if varUnit == 'Ci':
                    varUnit = 'uCi'
                if varUnit == 'gram/min' or varUnit == 'g/min':
                    varUnit = 'ug/min'
                if varUnit == 'gram/s' or varUnit == 'g/s':
                    varUnit = 'ug/s'
                if varUnit == 'gram/m3' or varUnit == 'g/m3':
                    varUnit = 'ug/m3'
                if varUnit == 'gram/L' or varUnit == 'g/L':
                    varUnit = 'ug/L'
                if varUnit == 'W/L':
                    varUnit = 'uW/L'
                if varUnit == 'Ci/L':
                    varUnit = 'uCi/L'
                if varUnit == 'L/min':
                    varUnit = 'uL/min'
                if varUnit == 'Pa':
                    varUnit = 'uPa'
                if varUnit == 's':
                    varUnit = 'us'
                if varUnit == 'm/s':
                    varUnit = 'um/s'

            if (y.max() >= 1e-3 and y.max()  < 1e-1) or \
               (y.min() <= -1e-3 and y.min() > -1e-1) and varScale != 'linear-log' and \
                    varScale != 'log-log' and varScale != 'log':
                y *= 1e3
                if varUnit == 'gram' or varUnit == 'g':
                    varUnit = 'mg'
                if varUnit == 'cc':
                    varUnit = 'm-cc'
                if varUnit == 'L':
                    varUnit = 'mL'
                if varUnit == 'W':
                    varUnit = 'mW'
                if varUnit == 'Ci':
                    varUnit = 'mCi'
                if varUnit == 'gram/min' or varUnit == 'g/min':
                    varUnit = 'mg/min'
                if varUnit == 'gram/s' or varUnit == 'g/s':
                    varUnit = 'mg/s'
                if varUnit == 'gram/m3' or varUnit == 'g/m3':
                    varUnit = 'mg/m3'
                if varUnit == 'gram/L' or varUnit == 'g/L':
                    varUnit = 'mg/L'
                if varUnit == 'W/L':
                    varUnit = 'mW/L'
                if varUnit == 'Ci/L':
                    varUnit = 'mCi/L'
                if varUnit == 'L/min':
                    varUnit = 'mL/min'
                if varUnit == 'Pa':
                    varUnit = 'mPa'
                if varUnit == 's':
                    varUnit = 'ms'
                if varUnit == 'm/s':
                    varUnit = 'mm/s'
            '''

            ax.set_xlabel('Time [' + time_unit + ']', fontsize=12)
            #ax.set_ylabel('$'+latex_name+'$' + ' [' + var_unit + ']', fontsize=12)
            ax.set_ylabel(latex_name+' [' + var_unit + ']', fontsize=12)

            '''
            ymax = y.max()
            dy = ymax * .1
            ymax += dy
            ymin = y.min()
            ymin -= dy

            if abs(ymin - ymax) <= 1.e-4:
                ymin = -1.0
                ymax = 1.0

            ax.set_ylim(ymin, ymax)
            '''

            if ncols >= 4:
                for l in ax.get_xticklabels():
                    l.set_fontsize(8)
            else:
                for l in ax.get_xticklabels():
                    l.set_fontsize(10)
            for l in ax.get_yticklabels():
                l.set_fontsize(10)

            if time_unit == 'h' and x.max() - x.min() <= 5.0:
                majorLocator = MultipleLocator(1.0)
                minorLocator = MultipleLocator(0.5)

                ax.xaxis.set_major_locator(majorLocator)
                ax.xaxis.set_minor_locator(minorLocator)

            if varScale == 'log' or varScale == 'log-log':
                ax.set_xscale('log')
                ax.set_yscale('log')
                positiveX = x > 0.0
                x = np.extract(positiveX, x)
                y = np.extract(positiveX, y)
                positiveY = y > 0.0
                x = np.extract(positiveY, x)
                y = np.extract(positiveY, y)
                if y.size > 0:
                    if y.min() > 0.0 and y.max() > y.min():
                        ymax = y.max()
                        dy = ymax * .1
                        ymax += dy
                        ymin = y.min()
                        ymin -= dy
                        if ymin < 0.0 or ymin > ymax / 1000.0:
                            ymin = ymax / 1000.0
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.set_ylim(1.0, 10.0)
                else:
                    ax.set_ylim(1.0, 10.0)

            if varScale == 'log-linear':
                ax.set_xscale('log')
                positiveX = x > 0.0 # True if > 0.0
                x = np.extract(positiveX, x)
                y = np.extract(positiveX, y)

            if varScale == 'linear-log':
                ax.set_yscale('log')
                positiveY = y > 0.0 # True if > 0.0
                x = np.extract(positiveY, x)
                y = np.extract(positiveY, y)
                #assert x.size == y.size, 'size error; stop.'
                if y.size > 0:
                    if y.min() > 0.0 and y.max() > y.min():
                        ymax = y.max()
                        dy = ymax * .1
                        ymax += dy
                        ymin = y.min()
                        ymin -= dy
                        if ymin < 0.0 or ymin > ymax / 1000.0:
                            ymin = ymax / 1000.0
                        ax.set_ylim(y.min(), ymax)
                    else:
                        ax.set_ylim(1.0, 10.0)
                else:
                    ax.set_ylim(1.0, 10.0)

            # ...................
            # make the plot here

            yformatter = ScalarFormatter(useMathText=True,useOffset=True)
            yformatter.set_powerlimits((15, 5))
            ax.yaxis.set_major_formatter(yformatter)

            if varLegend:

                ax.plot(x, y, 's-', color='black', linewidth=0.5, markersize=2,
                        markeredgecolor='black', label=varLegend)
            else:

                ax.plot(x, y, 's-', color='black', linewidth=0.5, markersize=2,
                        markeredgecolor='black')

            # ...................

            ax.set_title(info_str, fontsize=14)
            if varLegend:
                ax.legend(loc='best', prop={'size': 7})
            ax.grid()

        # end of: for i_var in range(num_var):

        fig_name = lead_name+'-'+self.name+'-phase-plot-'+str(i_dash).zfill(2)
        fig.savefig(fig_name+'.png', dpi=dpi, format='png')

        #plt.close(fig_num)

        if show:
            plt.show()

        #pickle.dump( fig, open(fig_name+'.pickle','wb') )

if __name__ == '__main__':
    tbp_org = Species(name='TBP', formula_name='(C4H9O)_3PO(o)',
                      atoms=['12*C','27*H','4*O','P'] )
    quant = Quantity( name='volume' )
    phase = PhaseNew(name='solvent',species=[tbp_org],quantities=[quant])
    print(phase)
