"""
Valmor F. de Almeida dealmeidav@ornl.gov; vfda

Cortix: a program for system-level modules
        coupling, execution, and analysis.

Tue Dec 10 11:21:30 EDT 2013
"""
#*********************************************************************************
import os, sys, io
import datetime
from   src.configtree import ConfigTree
from   ._setup import _setup
#*********************************************************************************

#*********************************************************************************
# network class constructor

def _network(self, netConfigNode):  

  assert type(netConfigNode) is ConfigTree, '-> netConfigNode is invalid.' 

  self.configNode = netConfigNode

  self.name = self.configNode.GetNodeName()

  _setup( self )

  return

#*********************************************************************************
