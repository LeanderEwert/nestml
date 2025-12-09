# -*- coding: utf-8 -*-
#
# ast_vector_parameter_setter_and_printer_factory.py
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

from pynestml.utils.ast_vector_parameter_setter_and_printer import ASTPreAndSuffixSetterAndPrinter
from pynestml.visitors.ast_visitor import ASTVisitor

from pynestml.utils.model_parser import ModelParser
from pynestml.visitors.ast_symbol_table_visitor import ASTSymbolTableVisitor
from pynestml.symbol_table.scope import Scope, ScopeType, Symbol, SymbolKind
from pynestml.symbols.variable_symbol import VariableSymbol


class ASTPreAndSuffixSetterAndPrinterFactory:
    """
    This file is part of the compartmental code generation process.

    Part of vectorized printing.
    """
    def __init__(self, model, printer):
        self.printer = printer
        self.model = model

    def create_ast_pre_and_suffix_setter_and_printer(self, prefix=None, suffix=None):
        my_printer = ASTPreAndSuffixSetterAndPrinter()
        my_printer.printer = self.printer
        my_printer.model = self.model
        my_printer.suffix = suffix
        my_printer.prefix = prefix
        return my_printer
