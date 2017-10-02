#
# CoCoInitVarsWithOdesProvided.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

from pynestml.src.main.python.org.nestml.cocos.CoCo import CoCo
from pynestml.src.main.python.org.nestml.ast.ASTNeuron import ASTNeuron
from pynestml.src.main.python.org.nestml.visitor.NESTMLVisitor import NESTMLVisitor
from pynestml.src.main.python.org.utils.Logger import Logger, LOGGING_LEVEL
from pynestml.src.main.python.org.nestml.symbol_table.symbols.Symbol import SymbolKind


class CoCoInitVarsWithOdesProvided(CoCo):
    """
    This CoCo ensures that all variables which have been declared in the "initial_values" block are provided 
    with a corresponding ode.
    Allowed:
        initial_values:
            V_m mV = E_L
        end
        ...
        equations:
            V_m' = ...
        end
    Not allowed:        
        initial_values:
            V_m mV = E_L
        end
        ...
        equations:
            # no ode declaration given
        end
    """

    @classmethod
    def checkCoCo(cls, _neuron=None):
        """
        Checks this coco on the handed over neuron.
        :param _neuron: a single neuron instance.
        :type _neuron: ASTNeuron
        """
        assert (_neuron is not None and isinstance(_neuron, ASTNeuron)), \
            '(PyNestML.CoCo.VariablesDefined) No or wrong type of neuron provided (%s)!' % type(_neuron)
        _neuron.accept(InitVarsVisitor())
        return


class InitVarsVisitor(NESTMLVisitor):
    """
    This visitor checks that all variables as provided in the init block have been provided with an ode.
    """

    def visitDeclaration(self, _declaration=None):
        """
        Checks the coco on the current node.
        :param _declaration: a single declaration.
        :type _declaration: ASTDeclaration
        """
        for var in _declaration.getVariables():
            symbol = _declaration.getScope().resolveToSymbol(var.getNameOfLhs(), SymbolKind.VARIABLE)
            # first check that all initial value variables have a lhs
            if symbol is not None and symbol.isInitValues() and not _declaration.hasExpression():
                Logger.logMessage(
                    'No rhs of initial value of variable "%s" at %s!'
                    % (var.getName(), var.getSourcePosition().printSourcePosition()),
                    LOGGING_LEVEL.ERROR)
            # now check that they have been provided with an ODE
            if symbol is not None and symbol.isInitValues() and not symbol.isOdeDefined():
                Logger.logMessage(
                    'Variable "%s" at %s not provided with an ODE!'
                    % (var.getName(), var.getSourcePosition().printSourcePosition()),
                    LOGGING_LEVEL.ERROR)
            if symbol is not None and symbol.isInitValues() and not symbol.hasInitialValue():
                Logger.logMessage(
                    'Initial value of ode variable "%s" at %s not provided!'
                    % (var.getName(), var.getSourcePosition().printSourcePosition()),
                    LOGGING_LEVEL.ERROR)
        return
