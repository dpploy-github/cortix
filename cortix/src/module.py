#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment
# https://cortix.org

from cortix.src.port import Port
from cortix.src.port import PortType

class Module:
    """
    The representation of a Cortix module. This class is to be inherited by every
    Cortix module. It provides facilities for creating and connecting modules within
    the Cortix network.
    """
    # Global list of ports (must be populated by the module)
    ports =  []
    def __init__(self):
        pass

    def send(self, data, port):
        """
        Send data through a given provide port
        """
        if isinstance(port, string):
            assert port in [p.name for p in self.ports], "Unknown port!"
        elif isinstance(port, Port):
            assert port in self.ports, "Unknown port!"
        else:
            raise TypeError("port must be of Port or String type")

        # TODO: Implement MPI calls here

    def recv(self, port):
        """
        Receive data from a given use port
        """
        if isinstance(port, string):
            assert port in [p.name for p in self.ports], "Unknown port!"
        elif isinstance(port, Port):
            assert port in self.ports, "Unknown port!"
        else:
            raise TypeError("port must be of Port or String type")

        # TODO: Implement MPI calls here

    def add_port(self, port_name, port_type):
        p = Port(port_name, port_type)
        if p not in self.ports:
            self.ports.append(p)
