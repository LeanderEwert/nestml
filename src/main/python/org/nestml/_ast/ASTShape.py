"""
@author kperun
TODO header
"""
import ASTVariable
import ASTExpr


class ASTShape:
    """
    This class is used to store shapes. 
    Grammar:
        shape : 'shape' lhs=variable '=' rhs=expr;
    """
    __lhs = None
    __rhs = None

    def __init__(self, _lhs: ASTVariable = None, _rhs: ASTExpr = None):
        """
        Standard constructor of ASTShape.
        :param _lhs: the variable corresponding to the shape 
        :type _lhs: ASTVariable
        :param _rhs: the right-hand side expression
        :type _rhs: ASTExpr
        """
        self.__lhs = _lhs
        self.__rhs = _rhs

    @classmethod
    def makeASTShape(cls,_lhs: ASTVariable = None, _rhs: ASTExpr = None):
        """
        Factory method of ASTShape.
        :param _lhs: the variable corresponding to the shape
        :type _lhs: ASTVariable
        :param _rhs: the right-hand side expression
        :type _rhs: ASTExpr
        :return: a new ASTShape object
        :rtype: ASTShape
        """
        return cls(_lhs,_rhs)

    def getVariable(self) -> ASTVariable:
        """
        Returns the variable of the left-hand side.
        :return: the variable
        :rtype: ASTVariable
        """
        return self.__lhs

    def getExpression(self)-> ASTExpr:
        """
        Returns the right-hand side expression.
        :return: the expression
        :rtype: ASTExpr
        """
        return self.__rhs

