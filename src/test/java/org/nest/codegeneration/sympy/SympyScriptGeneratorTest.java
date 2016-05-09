/*
 * Copyright (c) 2015 RWTH Aachen. All rights reserved.
 *
 * http://www.se-rwth.de/
 */
package org.nest.codegeneration.sympy;

import org.junit.Ignore;
import org.junit.Test;
import org.nest.base.ModelbasedTest;
import org.nest.commons._ast.ASTFunctionCall;
import org.nest.nestml._ast.ASTNESTMLCompilationUnit;
import org.nest.nestml._parser.NESTMLParser;
import org.nest.nestml._symboltable.NESTMLScopeCreator;
import org.nest.ode._ast.ASTEquation;
import org.nest.symboltable.predefined.PredefinedFunctions;
import org.nest.utils.ASTNodes;

import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.nest.codegeneration.sympy.SympyScriptGenerator.generateSympyODEAnalyzer;

/**
 * Tests that the solver script is generated from an ODE based model.
 *
 * @author plotnikov
 */
public class SympyScriptGeneratorTest extends ModelbasedTest {
  private static final String PATH_TO_PSC_MODEL
      = "src/test/resources/codegeneration/iaf_neuron.nestml";
  private static final String PATH_TO_COND_MODEL
      = "src/test/resources/codegeneration/iaf_cond_alpha.nestml";
  private static final String PATH_TO_COND_IMPLICIT_MODEL
      = "src/test/resources/codegeneration/iaf_cond_alpha_implicit.nestml";

  private static final String OUTPUT_FOLDER = "target";
  private static final Path OUTPUT_SCRIPT_DIRECTORY = Paths.get(OUTPUT_FOLDER, "sympy");

  @Test
  public void generateSymPySolverForPSCModel() throws IOException {
    generateAndCheck(PATH_TO_PSC_MODEL);
  }

  @Test
  public void generateSymPySolverForCondModel() throws IOException {
    generateAndCheck(PATH_TO_COND_MODEL);
  }

  @Ignore("Enable as soon as the script can handle it")
  @Test
  public void generateSymPySolverForCondImplicitModel() throws IOException {
    generateAndCheck(PATH_TO_COND_IMPLICIT_MODEL);
  }

  @Test
  public void testReplacement() throws IOException {
    final NESTMLParser p = new NESTMLParser(TEST_MODEL_PATH);
    final String ODE_DECLARATION = "V' = -1/Tau * V + 1/C_m * (I_sum(G, spikes) + I_e + ext_currents)";
    final Optional<ASTEquation> ode = p.parseEquation(new StringReader(ODE_DECLARATION));
    assertTrue(ode.isPresent());

    boolean i_sum = ASTNodes.getAll(ode.get(), ASTFunctionCall.class)
        .stream()
        .anyMatch(astFunctionCall -> astFunctionCall.getCalleeName().equals(PredefinedFunctions.I_SUM));
    assertTrue(i_sum);

    final ASTEquation testant = SympyScriptGenerator.replace_I_sum(ode.get());
    i_sum = ASTNodes.getAll(testant, ASTFunctionCall.class)
        .stream()
        .anyMatch(astFunctionCall -> astFunctionCall.getCalleeName().equals(PredefinedFunctions.I_SUM));
    assertFalse(i_sum);
  }

  private void generateAndCheck(final String pathToModel) throws IOException {
    final NESTMLParser p = new NESTMLParser(TEST_MODEL_PATH);
    final Optional<ASTNESTMLCompilationUnit> root = p.parse(pathToModel);
    assertTrue(root.isPresent());

    final NESTMLScopeCreator nestmlScopeCreator = new NESTMLScopeCreator(TEST_MODEL_PATH);
    nestmlScopeCreator.runSymbolTableCreator(root.get());

    final Optional<Path> generatedScript = generateSympyODEAnalyzer(
            root.get().getNeurons().get(0), OUTPUT_SCRIPT_DIRECTORY);

    assertTrue(generatedScript.isPresent());
  }


}
