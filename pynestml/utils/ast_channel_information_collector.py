# -*- coding: utf-8 -*-
#
# ast_channel_information_collector.py
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

from collections import defaultdict
import copy

from pynestml.frontend.frontend_configuration import FrontendConfiguration
from pynestml.meta_model.ast_block_with_variables import ASTBlockWithVariables
from pynestml.meta_model.ast_inline_expression import ASTInlineExpression
from pynestml.meta_model.ast_neuron import ASTNeuron
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.messages import Messages
from pynestml.visitors.ast_visitor import ASTVisitor


class ASTChannelInformationCollector(object):
    """
    This class is used to enforce constraint conditions on a compartmental model neuron

    While checking compartmental model constraints it also builds a nested
    data structure (chan_info) that can be used for code generation later

    Constraints:

    It ensures that all variables x as used in the inline expression named {channelType}
    (which has no kernels and is inside ASTEquationsBlock)
    have the following compartmental model functions defined

        x_inf_{channelType}(v_comp real) real
        tau_x_{channelType}(v_comp real) real


    Example:
        equations:
            inline Na real = m_Na_**3 * h_Na_**1
        end

        # triggers requirements for functions such as
        function h_inf_Na(v_comp real) real:
            return 1.0/(exp(0.16129032258064516*v_comp + 10.483870967741936) + 1.0)
        end

        function tau_h_Na(v_comp real) real:
            return 0.3115264797507788/((-0.0091000000000000004*v_comp - 0.68261830000000012)/(1.0 - 3277527.8765015295*exp(0.20000000000000001*v_comp)) + (0.024*v_comp + 1.200312)/(1.0 - 4.5282043263959816e-5*exp(-0.20000000000000001*v_comp)))
        end

    Moreover it checks
    -if all expected sates are defined,
    -that at least one gating variable exists (which is recognize when variable name ends with _{channel_name} )
    -that no gating variable repeats inside the inline expression that triggers cm mechanism
    Example:
        inline Na real = m_Na**3 * h_Na**1

    #causes the requirement for following entries in the state block

        gbar_Na
        e_Na
        m_Na
        h_Na

    Other allowed examples:
        # any variable that does not end with _Na is allowed
        inline Na real = m_Na**3 * h_Na**1 + x
        # gbar and e variables will not be counted as gating variables
        inline Na real = gbar_Na * m_Na**3 * h_Na**1 * (e_Na - v_comp) # gating variables detected: m and h

    Not allowed examples:
        inline Na real = p_Na **3 + p_Na **1  # same gating variable used twice
        inline Na real = x**2                 # no gating variables

    """

    padding_character = "_"
    inf_string = "inf"
    tau_sring = "tau"
    gbar_string = "gbar"
    equilibrium_string = "e"

    first_time_run = defaultdict(lambda: True)
    chan_info = defaultdict()

    def __init__(self, params):
        '''
        Constructor
        '''

    """
    detect_cm_inline_expressions

    analyzes any inline without kernels and returns

    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables": [ASTVariable, ASTVariable, ASTVariable, ...], # potential gating variables

        },
        "K":
        {
            ...
        }
    }
    """

    @classmethod
    def detect_cm_inline_expressions(cls, neuron):
        if not FrontendConfiguration.target_is_compartmental():
            return defaultdict()

        # search for inline expressions inside equations block
        inline_expressions_inside_equations_block_collector_visitor = ASTInlineExpressionInsideEquationsCollectorVisitor()
        neuron.accept(
            inline_expressions_inside_equations_block_collector_visitor)
        inline_expressions_dict = inline_expressions_inside_equations_block_collector_visitor.inline_expressions_to_variables

        # filter for any inline that has no kernel
        relevant_inline_expressions_to_variables = defaultdict(lambda: list())
        for expression, variables in inline_expressions_dict.items():
            inline_expression_name = expression.variable_name
            if not inline_expressions_inside_equations_block_collector_visitor.is_synapse_inline(
                    inline_expression_name):
                relevant_inline_expressions_to_variables[expression] = variables

        # create info structure
        chan_info = defaultdict()
        for inline_expression, inner_variables in relevant_inline_expressions_to_variables.items():
            info = defaultdict()
            channel_name = cls.cm_expression_to_channel_name(inline_expression)
            info["ASTInlineExpression"] = inline_expression
            info["gating_variables"] = inner_variables
            chan_info[channel_name] = info

        return chan_info

    # extract channel name from inline expression name
    # i.e  Na_ -> channel name is Na
    @classmethod
    def cm_expression_to_channel_name(cls, expr):
        assert isinstance(expr, ASTInlineExpression)
        return expr.variable_name.strip(cls.padding_character)

    # extract pure variable name from inline expression variable name
    # i.e  p_Na -> pure variable name is p
    @classmethod
    def extract_pure_variable_name(cls, varname, ic_name):
        varname = varname.strip(cls.padding_character)
        assert varname.endswith(ic_name)
        return varname[:-len(ic_name)].strip(cls.padding_character)

    # generate gbar variable name from ion channel name
    # i.e  Na -> gbar_Na
    @classmethod
    def get_expected_gbar_name(cls, ion_channel_name):
        return cls.gbar_string + cls.padding_character + ion_channel_name

    # generate equilibrium variable name from ion channel name
    # i.e  Na -> e_Na
    @classmethod
    def get_expected_equilibrium_var_name(cls, ion_channel_name):
        return cls.equilibrium_string + cls.padding_character + ion_channel_name

    # generate tau function name from ion channel name
    # i.e  Na, p -> tau_p_Na
    @classmethod
    def get_expected_tau_result_var_name(
            cls, ion_channel_name, pure_variable_name):
        return cls.padding_character + \
            cls.get_expected_tau_function_name(ion_channel_name, pure_variable_name)

    # generate tau variable name (stores return value)
    # from ion channel name and pure variable name
    # i.e  Na, p -> _tau_p_Na
    @classmethod
    def get_expected_tau_function_name(
            cls, ion_channel_name, pure_variable_name):
        return cls.tau_sring + cls.padding_character + \
            pure_variable_name + cls.padding_character + ion_channel_name

    # generate inf function name from ion channel name and pure variable name
    # i.e  Na, p -> p_inf_Na
    @classmethod
    def get_expected_inf_result_var_name(
            cls, ion_channel_name, pure_variable_name):
        return cls.padding_character + \
            cls.get_expected_inf_function_name(ion_channel_name, pure_variable_name)

    # generate inf variable name (stores return value)
    # from ion channel name and pure variable name
    # i.e  Na, p -> _p_inf_Na
    @classmethod
    def get_expected_inf_function_name(
            cls, ion_channel_name, pure_variable_name):
        return pure_variable_name + cls.padding_character + \
            cls.inf_string + cls.padding_character + ion_channel_name

    # calculate function names that must be implemented
    # i.e
    # m_Na**3 * h_Na**1
    # expects
    # m_inf_Na(v_comp real) real
    # tau_m_Na(v_comp real) real
    """
    analyzes cm inlines for expected function names
    input:
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables": [ASTVariable, ASTVariable, ASTVariable, ...]

        },
        "K":
        {
            ...
        }
    }

    output:
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": str,
                        "inf": str
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": str,
                        "inf": str
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

    """

    @classmethod
    def calc_expected_function_names_for_channels(cls, chan_info):
        variables_procesed = defaultdict()

        for ion_channel_name, channel_info in chan_info.items():
            cm_expression = channel_info["ASTInlineExpression"]
            variables = channel_info["gating_variables"]
            variable_names_seen = set()

            variables_info = defaultdict()
            channel_parameters_exclude = cls.get_expected_equilibrium_var_name(
                ion_channel_name), cls.get_expected_gbar_name(ion_channel_name)

            for variable_used in variables:
                variable_name = variable_used.name.strip(cls.padding_character)
                if not variable_name.endswith(ion_channel_name):
                    # not a gating variable
                    continue

                # exclude expected channel parameters
                if variable_name in channel_parameters_exclude:
                    continue

                # enforce unique variable names per channel, i.e n and m , not
                # n and n
                if variable_name in variable_names_seen:
                    code, message = Messages.get_cm_inline_expression_variable_used_mulitple_times(
                        cm_expression, variable_name, ion_channel_name)
                    Logger.log_message(
                        code=code,
                        message=message,
                        error_position=variable_used.get_source_position(),
                        log_level=LoggingLevel.ERROR,
                        node=variable_used)
                    continue
                else:
                    variable_names_seen.add(variable_name)

                pure_variable_name = cls.extract_pure_variable_name(
                    variable_name, ion_channel_name)
                expected_inf_function_name = cls.get_expected_inf_function_name(
                    ion_channel_name, pure_variable_name)
                expected_tau_function_name = cls.get_expected_tau_function_name(
                    ion_channel_name, pure_variable_name)

                variables_info[pure_variable_name] = defaultdict(
                    lambda: defaultdict())
                variables_info[pure_variable_name]["expected_functions"][cls.inf_string] = expected_inf_function_name
                variables_info[pure_variable_name]["expected_functions"][cls.tau_sring] = expected_tau_function_name
                variables_info[pure_variable_name]["ASTVariable"] = variable_used

            variables_procesed[ion_channel_name] = copy.copy(variables_info)

        for ion_channel_name, variables_info in variables_procesed.items():
            chan_info[ion_channel_name]["gating_variables"] = variables_info

        return chan_info

    """
    generate Errors on invalid variable names
    and add channel_parameters section to each channel

    input:
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

    output:

    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "channel_parameters":
            {
                "gbar":{"expected_name": "gbar_Na"},
                "e":{"expected_name": "e_Na"}
            }
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

    """
    @classmethod
    def add_channel_parameters_section_and_enforce_proper_variable_names(
            cls, node, chan_info):
        ret = copy.copy(chan_info)

        channel_parameters = defaultdict()
        for ion_channel_name, channel_info in chan_info.items():
            channel_parameters[ion_channel_name] = defaultdict()
            channel_parameters[ion_channel_name][cls.gbar_string] = defaultdict()
            channel_parameters[ion_channel_name][cls.gbar_string]["expected_name"] = cls.get_expected_gbar_name(
                ion_channel_name)
            channel_parameters[ion_channel_name][cls.equilibrium_string] = defaultdict(
            )
            channel_parameters[ion_channel_name][cls.equilibrium_string]["expected_name"] = cls.get_expected_equilibrium_var_name(
                ion_channel_name)

            if len(channel_info["gating_variables"]) < 1:
                cm_inline_expr = channel_info["ASTInlineExpression"]
                code, message = Messages.get_no_gating_variables(
                    cm_inline_expr, ion_channel_name)
                Logger.log_message(
                    code=code,
                    message=message,
                    error_position=cm_inline_expr.get_source_position(),
                    log_level=LoggingLevel.ERROR,
                    node=cm_inline_expr)
                continue

        for ion_channel_name, channel_info in chan_info.items():
            ret[ion_channel_name]["channel_parameters"] = channel_parameters[ion_channel_name]

        return ret

    """
    checks if all expected functions exist and have the proper naming and signature
    also finds their corresponding ASTFunction objects

    input
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": str,
                        "inf": str
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": str,
                        "inf": str
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

    output
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }
    """
    @classmethod
    def check_and_find_functions(cls, neuron, chan_info):
        ret = copy.copy(chan_info)
        # get functions and collect their names
        declared_functions = neuron.get_functions()

        function_name_to_function = {}
        for declared_function in declared_functions:
            function_name_to_function[declared_function.name] = declared_function

        # check for missing functions
        for ion_channel_name, channel_info in chan_info.items():
            for pure_variable_name, variable_info in channel_info["gating_variables"].items(
            ):
                if "expected_functions" in variable_info.keys():
                    for function_type, expected_function_name in variable_info["expected_functions"].items(
                    ):
                        if expected_function_name not in function_name_to_function.keys():
                            code, message = Messages.get_expected_cm_function_missing(
                                ion_channel_name, variable_info["ASTVariable"].name, expected_function_name)
                            Logger.log_message(
                                code=code,
                                message=message,
                                error_position=neuron.get_source_position(),
                                log_level=LoggingLevel.ERROR,
                                node=neuron)
                        else:
                            ret[ion_channel_name]["gating_variables"][pure_variable_name]["expected_functions"][function_type] = defaultdict()
                            ret[ion_channel_name]["gating_variables"][pure_variable_name]["expected_functions"][
                                function_type]["ASTFunction"] = function_name_to_function[expected_function_name]
                            ret[ion_channel_name]["gating_variables"][pure_variable_name][
                                "expected_functions"][function_type]["function_name"] = expected_function_name

                            # function must have exactly one argument
                            astfun = ret[ion_channel_name]["gating_variables"][pure_variable_name][
                                "expected_functions"][function_type]["ASTFunction"]
                            if len(astfun.parameters) != 1:
                                code, message = Messages.get_expected_cm_function_wrong_args_count(
                                    ion_channel_name, variable_info["ASTVariable"].name, astfun)
                                Logger.log_message(
                                    code=code,
                                    message=message,
                                    error_position=astfun.get_source_position(),
                                    log_level=LoggingLevel.ERROR,
                                    node=astfun)

                            # function must return real
                            if not astfun.get_return_type().is_real:
                                code, message = Messages.get_expected_cm_function_bad_return_type(
                                    ion_channel_name, astfun)
                                Logger.log_message(
                                    code=code,
                                    message=message,
                                    error_position=astfun.get_source_position(),
                                    log_level=LoggingLevel.ERROR,
                                    node=astfun)

                            if function_type == "tau":
                                ret[ion_channel_name]["gating_variables"][pure_variable_name]["expected_functions"][function_type][
                                    "result_variable_name"] = cls.get_expected_tau_result_var_name(ion_channel_name, pure_variable_name)
                            elif function_type == "inf":
                                ret[ion_channel_name]["gating_variables"][pure_variable_name]["expected_functions"][function_type][
                                    "result_variable_name"] = cls.get_expected_inf_result_var_name(ion_channel_name, pure_variable_name)
                            else:
                                raise RuntimeError(
                                    'This should never happen! Unsupported function type ' + function_type + ' from variable ' + pure_variable_name)

        return ret


#----------------------- New collection functions for generalized ODE Descriptions

    """
        detect_cm_inline_expressions_ode

        analyzes any inline without kernels and returns

        {
            "Na":
            {
                "ASTInlineExpression": ASTInlineExpression,
                "ode_variables": [ASTVariable, ASTVariable, ASTVariable, ...], # potential ode variables

            },
            "K":
            {
                ...
            }
        }
        """

    @classmethod
    def detect_cm_inline_expressions_ode(cls, neuron):
        if not FrontendConfiguration.target_is_compartmental():
            return defaultdict()

        inline_expressions_inside_equations_block_collector_visitor = ASTInlineExpressionInsideEquationsCollectorVisitor()
        neuron.accept(
            inline_expressions_inside_equations_block_collector_visitor)
        inline_expressions_dict = inline_expressions_inside_equations_block_collector_visitor.inline_expressions_to_variables

        # filter for any inline that has no kernel
        relevant_inline_expressions_to_variables = defaultdict(lambda: list())
        for expression, variables in inline_expressions_dict.items():
            inline_expression_name = expression.variable_name
            if not inline_expressions_inside_equations_block_collector_visitor.is_synapse_inline(
                    inline_expression_name):
                relevant_inline_expressions_to_variables[expression] = variables

        # create info structure
        chan_info = defaultdict()
        for inline_expression, inner_variables in relevant_inline_expressions_to_variables.items():
            info = defaultdict()
            channel_name = cls.cm_expression_to_channel_name(inline_expression)
            info["ASTInlineExpression"] = inline_expression
            info["ode_variables"] = inner_variables
            chan_info[channel_name] = info

        return chan_info


    """
        analyzes cm inlines for expected odes
        input:
        {
            "Na":
            {
                "ASTInlineExpression": ASTInlineExpression,
                "ode_variables": [ASTVariable, ASTVariable, ASTVariable, ...]

            },
            "K":
            {
                ...
            }
        }

        output:
        {
            "Na":
            {
                "ASTInlineExpression": ASTInlineExpression,
                "ode_variables":
                {
                    "m":
                    {
                        "ASTVariable": ASTVariable
                        "describing_ode": ASTOdeEquation
                    },
                    "h":
                    {
                        "ASTVariable": ASTVariable
                        "describing_ode": ASTOdeEquation
                    },
                    ...
                },
            }
            "K":
            {
                ...
            }
        }

        """

    @classmethod
    def search_related_ode(cls, neuron, chan_info):
        ret = copy.copy(chan_info)


    @classmethod
    def check_related_declarations_and_definitions(cls, neuron, chan_info):


#----------------------- Test function for building a chan_info prototype
    @classmethod
    def create_chan_info_ode_prototype_HH(cls, chan_info):
        ret = copy.copy(chan_info)



#----------------------- Collector root functions

    @classmethod
    def get_chan_info(cls, neuron: ASTNeuron):
        """
        returns previously generated chan_info
        as a deep copy so it can't be changed externally
        via object references
        :param neuron: a single neuron instance.
        :type neuron: ASTNeuron
        """

        # trigger generation via check_co_co
        # if it has not been called before
        if cls.first_time_run[neuron]:
            cls.check_co_co(neuron)

        return copy.deepcopy(cls.chan_info[neuron])

    @classmethod
    def check_co_co(cls, neuron: ASTNeuron):
        """
        :param neuron: a single neuron instance.
        :type neuron: ASTNeuron
        """
        # make sure we only run this a single time
        # subsequent calls will be after AST has been transformed
        # where kernels have been removed
        # and inlines therefore can't be recognized by kernel calls any more
        if cls.first_time_run[neuron]:
            chan_info = cls.detect_cm_inline_expressions(neuron)

            # further computation not necessary if there were no cm neurons
            if not chan_info:
                cls.chan_info[neuron] = dict()
                # mark as done so we don't enter here again
                cls.first_time_run[neuron] = False
                return True

            chan_info = cls.calc_expected_function_names_for_channels(
                chan_info)
            chan_info = cls.check_and_find_functions(neuron, chan_info)
            chan_info = cls.add_channel_parameters_section_and_enforce_proper_variable_names(
                neuron, chan_info)

            # now check for existence of expected state variables
            # and add their ASTVariable objects to chan_info
            missing_states_visitor = VariableMissingVisitor(chan_info)
            neuron.accept(missing_states_visitor)

            cls.chan_info[neuron] = chan_info
            cls.first_time_run[neuron] = False

        return True


# ------------------- Helper classes
"""
    Finds the actual ASTVariables in state block
    For each expected variable extract their right hand side expression
    which contains the desired state value


    chan_info input
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "channel_parameters":
            {
                "gbar":{"expected_name": "gbar_Na"},
                "e":{"expected_name": "e_Na"}
            }
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "expected_functions":
                    {
                        "tau": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str},
                        "inf": {"ASTFunction": ASTFunction, "function_name": str, "result_variable_name": str}
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

    chan_info output
    {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "channel_parameters":
            {
                "gbar": {
                            "expected_name": "gbar_Na",
                            "parameter_block_variable": ASTVariable,
                            "rhs_expression": ASTSimpleExpression or ASTExpression
                        },
                "e":  {
                            "expected_name": "e_Na",
                            "parameter_block_variable": ASTVariable,
                            "rhs_expression": ASTSimpleExpression or ASTExpression
                        }
            }
            "gating_variables":
            {
                "m":
                {
                    "ASTVariable": ASTVariable,
                    "state_variable": ASTVariable,
                    "expected_functions":
                    {
                        "tau":  {
                                    "ASTFunction": ASTFunction,
                                    "function_name": str,
                                    "result_variable_name": str,
                                    "rhs_expression": ASTSimpleExpression or ASTExpression
                                },
                        "inf":  {
                                    "ASTFunction": ASTFunction,
                                    "function_name": str,
                                    "result_variable_name": str,
                                    "rhs_expression": ASTSimpleExpression or ASTExpression
                                }
                    }
                },
                "h":
                {
                    "ASTVariable": ASTVariable,
                    "state_variable": ASTVariable,
                    "expected_functions":
                    {
                        "tau":  {
                                    "ASTFunction": ASTFunction,
                                    "function_name": str,
                                    "result_variable_name": str,
                                    "rhs_expression": ASTSimpleExpression or ASTExpression
                                },
                        "inf":  {
                                    "ASTFunction": ASTFunction,
                                    "function_name": str,
                                    "result_variable_name": str,
                                    "rhs_expression": ASTSimpleExpression or ASTExpression
                                }
                    }
                },
                ...
            }
        },
        "K":
        {
            ...
        }
    }

"""


class VariableMissingVisitor(ASTVisitor):

    def __init__(self, chan_info):
        super(VariableMissingVisitor, self).__init__()
        self.chan_info = chan_info

        # store ASTElement that causes the expecation of existence of state value
        # needed to generate sufficiently informative error message
        self.expected_to_object = defaultdict()

        self.values_expected_from_channel = set()
        for ion_channel_name, channel_info in self.chan_info.items():
            for channel_variable_type, channel_variable_info in channel_info["channel_parameters"].items(
            ):
                self.values_expected_from_channel.add(
                    channel_variable_info["expected_name"])
                self.expected_to_object[channel_variable_info["expected_name"]
                                        ] = channel_info["ASTInlineExpression"]

        self.values_expected_from_variables = set()
        for ion_channel_name, channel_info in self.chan_info.items():
            for pure_variable_type, variable_info in channel_info["gating_variables"].items(
            ):
                self.values_expected_from_variables.add(
                    variable_info["ASTVariable"].name)
                self.expected_to_object[variable_info["ASTVariable"]
                                        .name] = variable_info["ASTVariable"]

        self.not_yet_found_variables = set(
            self.values_expected_from_channel).union(
            self.values_expected_from_variables)

        self.inside_state_block = False
        self.inside_parameter_block = False
        self.inside_declaration = False
        self.current_block_with_variables = None
        self.current_declaration = None

    def visit_declaration(self, node):
        self.inside_declaration = True
        self.current_declaration = node

    def endvisit_declaration(self, node):
        self.inside_declaration = False
        self.current_declaration = None

    def visit_variable(self, node):
        if self.inside_state_block and self.inside_declaration:
            varname = node.name
            if varname in self.not_yet_found_variables:
                Logger.log_message(message="Expected state variable '" + varname + "' found inside state block", log_level=LoggingLevel.INFO)
                self.not_yet_found_variables.difference_update({varname})

                # make a copy because we can't write into the structure directly
                # while iterating over it
                chan_info_updated = copy.copy(self.chan_info)

                # now that we found the satate defintion, extract information
                # into chan_info

                # state variables
                if varname in self.values_expected_from_variables:
                    for ion_channel_name, channel_info in self.chan_info.items():
                        for pure_variable_name, variable_info in channel_info["gating_variables"].items(
                        ):
                            if variable_info["ASTVariable"].name == varname:
                                chan_info_updated[ion_channel_name]["gating_variables"][pure_variable_name]["state_variable"] = node
                                rhs_expression = self.current_declaration.get_expression()
                                if rhs_expression is None:
                                    code, message = Messages.get_cm_variable_value_missing(
                                        varname)
                                    Logger.log_message(
                                        code=code,
                                        message=message,
                                        error_position=node.get_source_position(),
                                        log_level=LoggingLevel.ERROR,
                                        node=node)

                                chan_info_updated[ion_channel_name]["gating_variables"][
                                    pure_variable_name]["rhs_expression"] = rhs_expression
                self.chan_info = chan_info_updated

        if self.inside_parameter_block and self.inside_declaration:
            varname = node.name
            if varname in self.not_yet_found_variables:
                Logger.log_message(message="Expected variable '" + varname + "' found inside parameter block", log_level=LoggingLevel.INFO)
                self.not_yet_found_variables.difference_update({varname})

                # make a copy because we can't write into the structure directly
                # while iterating over it
                chan_info_updated = copy.copy(self.chan_info)
                # now that we found the defintion, extract information into
                # chan_info

                # channel parameters
                if varname in self.values_expected_from_channel:
                    for ion_channel_name, channel_info in self.chan_info.items():
                        for variable_type, variable_info in channel_info["channel_parameters"].items(
                        ):
                            if variable_info["expected_name"] == varname:
                                chan_info_updated[ion_channel_name]["channel_parameters"][
                                    variable_type]["parameter_block_variable"] = node
                                rhs_expression = self.current_declaration.get_expression()
                                if rhs_expression is None:
                                    code, message = Messages.get_cm_variable_value_missing(
                                        varname)
                                    Logger.log_message(
                                        code=code,
                                        message=message,
                                        error_position=node.get_source_position(),
                                        log_level=LoggingLevel.ERROR,
                                        node=node)

                                chan_info_updated[ion_channel_name]["channel_parameters"][
                                    variable_type]["rhs_expression"] = rhs_expression
                self.chan_info = chan_info_updated

    def endvisit_neuron(self, node):
        missing_variable_to_proper_block = {}
        for variable in self.not_yet_found_variables:
            if variable in self.values_expected_from_channel:
                missing_variable_to_proper_block[variable] = "parameters block"
            elif variable in self.values_expected_from_variables:
                missing_variable_to_proper_block[variable] = "state block"

        if self.not_yet_found_variables:
            code, message = Messages.get_expected_cm_variables_missing_in_blocks(
                missing_variable_to_proper_block, self.expected_to_object)
            Logger.log_message(
                code=code,
                message=message,
                error_position=node.get_source_position(),
                log_level=LoggingLevel.ERROR,
                node=node)

    def visit_block_with_variables(self, node):
        if node.is_state:
            self.inside_state_block = True
        if node.is_parameters:
            self.inside_parameter_block = True
        self.current_block_with_variables = node

    def endvisit_block_with_variables(self, node):
        if node.is_state:
            self.inside_state_block = False
        if node.is_parameters:
            self.inside_parameter_block = False
        self.current_block_with_variables = None


"""
for each inline expression inside the equations block,
collect all ASTVariables that are present inside
"""


class ASTInlineExpressionInsideEquationsCollectorVisitor(ASTVisitor):

    def __init__(self):
        super(ASTInlineExpressionInsideEquationsCollectorVisitor, self).__init__()
        self.inline_expressions_to_variables = defaultdict(lambda: list())
        self.inline_expressions_with_kernels = set()
        self.inside_equations_block = False
        self.inside_inline_expression = False
        self.inside_kernel_call = False
        self.inside_simple_expression = False
        self.current_inline_expression = None

    def is_synapse_inline(self, inline_name):
        return inline_name in self.inline_expressions_with_kernels

    def visit_variable(self, node):
        if self.inside_equations_block and self.inside_inline_expression and self.current_inline_expression is not None:
            self.inline_expressions_to_variables[self.current_inline_expression].append(
                node)

    def visit_inline_expression(self, node):
        self.inside_inline_expression = True
        self.current_inline_expression = node

    def endvisit_inline_expression(self, node):
        self.inside_inline_expression = False
        self.current_inline_expression = None

    def visit_equations_block(self, node):
        self.inside_equations_block = True

    def endvisit_equations_block(self, node):
        self.inside_equations_block = False

    def visit_function_call(self, node):
        if self.inside_equations_block:
            if self.inside_inline_expression and self.inside_simple_expression:
                if node.get_name() == "convolve":
                    inline_name = self.current_inline_expression.variable_name
                    self.inline_expressions_with_kernels.add(inline_name)

    def visit_simple_expression(self, node):
        self.inside_simple_expression = True

    def endvisit_simple_expression(self, node):
        self.inside_simple_expression = False
