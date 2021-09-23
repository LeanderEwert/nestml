      
import copy
from pynestml.meta_model.ast_expression import ASTExpression
from pynestml.meta_model.ast_inline_expression import ASTInlineExpression
from pynestml.meta_model.ast_neuron import ASTNeuron
from pynestml.utils.model_parser import ModelParser
from pynestml.visitors.ast_symbol_table_visitor import ASTSymbolTableVisitor
import sympy


class ChanInfoEnricher():
    

    """
    Adds derivative of inline expression to chan_info
    This needs to be done used from within nest_codegenerator
    because the import of ModelParser will otherwise cause 
    a circular dependency when this is used 
    inside CmProcessing
    
    input:
    
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
    
    output:
    
        {
        "Na":
        {
            "ASTInlineExpression": ASTInlineExpression,
            "inline_derivative": ASTInlineExpression,
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

    @classmethod
    def enrich_with_additional_info(cls, neuron: ASTNeuron, chan_info: dict):
        chan_info_copy = copy.copy(chan_info)
        for ion_channel_name, ion_channel_info in chan_info_copy.items():
            chan_info[ion_channel_name]["inline_derivative"] = cls.computeExpressionDerivative(chan_info[ion_channel_name]["ASTInlineExpression"])
        return chan_info
    
    @classmethod
    def computeExpressionDerivative(cls, inline_expression: ASTInlineExpression) -> ASTExpression:
        expr_str = str(inline_expression.get_expression())
        sympy_expr = sympy.parsing.sympy_parser.parse_expr(expr_str)
        sympy_expr = sympy.diff(sympy_expr, "v_comp")
        
        ast_expression_d = ModelParser.parse_expression(str(sympy_expr))
        # copy scope of the original inline_expression into the the derivative
        ast_expression_d.update_scope(inline_expression.get_scope())
        ast_expression_d.accept(ASTSymbolTableVisitor())  
        
        return ast_expression_d
  
    
    
    
    
    
    
    
    
    
        