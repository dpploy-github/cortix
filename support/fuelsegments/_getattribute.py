#!/usr/bin/env python
"""
Valmor F. de Almeida dealmeidav@ornl.gov; vfda

Fuel segment support class

Thu Jun 25 18:16:06 EDT 2015
"""
#*********************************************************************************
import os, sys, io, time, datetime
import math, random
from ..periodictable import ELEMENTS
from ..periodictable import SERIES
#*********************************************************************************

# Get stored fuel segment property either overall or on a nuclide basis 

#---------------------------------------------------------------------------------
def _GetAttribute(self, attributeName, nuclide=None, series=None ):

  assert attributeName in self.attributeNames, ' attribute name: %r; options: %r; fail.' % (attributeName,self.attributeNames)

  if nuclide is not None: assert series is None, 'fail.'
  if series is not None: assert nuclide is None, 'fail.'

  if series is not None: assert False,' not implemented.'

#.................................................................................
# # of segments

  if attributeName == 'nSegments':  return 1

#.................................................................................
# segmentId   

  if attributeName == 'segmentId': return self.geometry['segment id'] 

#.................................................................................
# fuel volume

  if attributeName == 'fuelVolume':   return  __GetFuelSegmentVolume( self )
#.................................................................................
# segment volume

  if attributeName == 'segmentVolume': 
 
    claddingLength = self.geometry['cladding length [cm]'] 
    claddingDiam   = self.geometry['OD [cm]']
    volume = claddingLength * math.pi * (claddingDiam/2.0)**2
    return volume

#.................................................................................
# fuel segment overall quantities
  if nuclide is None and series is None:

# mass or mass concentration
     if attributeName == 'massCC' or attributeName == 'massDens' or attributeName == 'mass': 
        massCC = 0.0
        for spc in self._species:
            massCC += spc.massCC
        if attributeName == 'massCC' or attributeName == 'massDens': 
          return massCC
        else:
          volume = __GetFuelSegmentVolume( self )
          return massCC * volume
# radioactivity 
     if attributeName == 'radioactivtyDens' or attributeName == 'radioactivity':
        radDens = 0.0
        for spc in self._species:
            radDens += spc.molarRadioactivity * spc.molarCC
        if attributeName == 'radioactivityDens': 
          return radDens
        else:
          volume = __GetFuelSegmentVolume( self )
          return radDens * volume
# gamma          
     if attributeName == 'gammaDens' or attributeName == 'gamma':
        gammaDens = 0.0
        for spc in self._species:
            gammaDens += spc.molarGammaPwr * spc.molarCC
        if attributeName == 'gammaDens': 
          return gammaDens
        else:
          volume = __GetFuelSegmentVolume( self )
          return gammaDens * volume
# heat           
     if attributeName == 'heatDens' or attributeName == 'heat':
        heatDens = 0.0
        for spc in self._species:
            heatDens += spc.molarHeatPwr * spc.molarCC
        if attributeName == 'heatDens': 
          return heatDens
        else:
          volume = __GetFuelSegmentVolume( self )
          return heatDens * volume

#.................................................................................
# radioactivity               

  if attributeName == 'radioactivityDens' or attributeName == 'radioactivity': 
     colName = 'Radioactivity Dens. [Ci/cc]'

#.................................................................................
# thermal                     

  if attributeName == 'thermalDens' or attributeName == 'thermal' or  \
     attributeName == 'heatDens'    or attributeName == 'heat': 
     colName = 'Thermal Dens. [W/cc]'

#.................................................................................
# gamma                       

  if attributeName == 'gammaDens' or attributeName == 'gamma': 
     colName = 'Gamma Dens. [W/cc]'

#.................................................................................
##################################################################################
#.................................................................................

  if attributeName[-4:] == 'Dens' or attributeName[-2:] == 'CC': 
     attributeDens = True
  else:
     attributeDens = False 

#.................................................................................
# all nuclide content of the fuel added

  if nuclide is None and series is None:

     density = 0.0

     density = self.propertyDensities[ colName ].sum()

     if attributeDens is False:  
        volume = __GetFuelSegmentVolume( self )
        prop = density * volume
        return prop
     else:
        return density

#.................................................................................
# get chemical element series

  if series is not None:
 
     density = 0.0

     for isotope in isotopes:
       density += self.propertyDensities.loc[isotope,colName]

     if attributeDens is False:  
        volume = __GetFuelSegmentVolume( self )
        prop = density * volume
        return prop
     else:
        return density
   
#.................................................................................
# get specific nuclide (either the isotopes of the nuclide or the specific isotope) property

  if nuclide is not None:

#    print('nuclide = ', nuclide)


    # case with a particular isotope
    if len(nuclide.split('-')) == 2: 

       nuclideMolarMass = ELEMENTS[nuclide.split('-')[0]].isotopes[int(nuclide.split('-')[1].strip('m'))].mass

       massCC = 0.0

       for spc in self._species:

         nuclides = spc.atoms

         moleFraction = 0.0
         for nucl in nuclides:

#           print('nucl       = ', nucl)
#           print('nucl.split = ', nucl.split('*'))

           if len(nucl.split('*')) == 1: 
             if nucl.split('*')[0].strip() == nuclide: 
                moleFraction = 1.0
             else:
                moleFraction = 0.0
           elif len(nucl.split('*')) == 2: 
             if nucl.split('*')[1].strip() == nuclide: 
#                print('made here')
                moleFraction = float(nucl.split('*')[0].strip()) 
             else:
                moleFraction = 0.0
           else:
             assert False

           massCC += spc.molarCC * moleFraction * nuclideMolarMass

#       print( 'moleFraction = ', moleFraction )
#       print( 'return mass = ', massCC * __GetFuelSegmentVolume( self ) )
       return massCC * __GetFuelSegmentVolume( self )

    density = 0.0

  # nuclide 
    if len(nuclide.split('-')) == 2:
       density = self.propertyDensities.loc[nuclide,colName]

  # chemical element 
    else:
      nuclidesNames = self.propertyDensities.index
#    print(self.propertyDensities)
      isotopes = [x for x in nuclidesNames if x.split('-')[0].strip()==nuclide]
#    print(isotopes)
      for isotope in isotopes:
        density += self.propertyDensities.loc[isotope,colName]

    if attributeDens is False:  
       volume = __GetFuelSegmentVolume( self )
       prop = density * volume
       return prop
    else:
       return density

#---------------------------------------------------------------------------------
def __GetFuelSegmentVolume(self):
 
  fuelLength = self.geometry['fuel length [cm]'] 
  fuelDiam   = self.geometry['fuel diameter [cm]']

  volume = fuelLength * math.pi * (fuelDiam/2.0)**2

  return volume

#*********************************************************************************
