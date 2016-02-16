/*
 * Copyright (c)  RWTH Aachen. All rights reserved.
 *
 * http://www.se-rwth.de/
 */
package org.nest.cli;

import org.junit.Assert;
import org.junit.Test;
import org.nest.base.ModelbasedTest;
import org.nest.codegeneration.NESTGenerator;
import org.nest.mocks.PSCMock;
import org.nest.nestml._symboltable.NESTMLScopeCreator;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.nest.utils.FileHelper.collectNESTMLModelFilenames;

/**
 * Mocks the codegenerator and tests executor functions.
 *
 * @author plotnikov
 */
public class CLIConfigurationExecutorTest extends ModelbasedTest {
  private static final Path TEST_INPUT_PATH = Paths.get("src/test/resources/command_line_base/");
  private static final Path TARGET_FOLDER = Paths.get("target/build");
  private final PSCMock pscMock = new PSCMock();
  private final Configuration testConfig;
  private final CLIConfigurationExecutor executor = new CLIConfigurationExecutor();
  private final NESTMLScopeCreator scopeCreator = new NESTMLScopeCreator(TEST_INPUT_PATH);

  public CLIConfigurationExecutorTest() {
    testConfig = new Configuration.Builder()
        .withCoCos()
        .withInputBasePath(TEST_INPUT_PATH)
        .withTargetPath(TARGET_FOLDER)
        .build();
  }

  @Test
  public void testExecutionTestConfiguration() {
    final NESTGenerator generator = new NESTGenerator(scopeCreator, pscMock);
    executor.execute(generator, testConfig);
  }

  @Test
  public void testArtifactCollection() {
    final List<Path> collectedFiles = collectNESTMLModelFilenames(TEST_INPUT_PATH);
    Assert.assertEquals(2, collectedFiles.size());
  }

}
