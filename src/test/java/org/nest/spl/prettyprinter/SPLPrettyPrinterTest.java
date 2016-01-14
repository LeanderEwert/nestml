/* * Copyright (c) 2015 RWTH Aachen. All rights reserved. * * http://www.se-rwth.de/ */package org.nest.spl.prettyprinter;import org.junit.Test;import org.nest.spl._ast.ASTSPLFile;import org.nest.spl._parser.SPLParser;import java.io.File;import java.io.IOException;import java.io.StringReader;import java.util.Optional;import static org.junit.Assert.assertTrue;/** * Checks that the pretty printed result can be parsed again. * * @author plotnikov */public class SPLPrettyPrinterTest {  private final SPLParser splFileParser = new SPLParser();  private final ExpressionsPrettyPrinter prettyPrinter = new ExpressionsPrettyPrinter();  private Optional<ASTSPLFile> parseStringAsSPLFile(final String fileAsString) throws IOException {    return splFileParser.parse(new StringReader(fileAsString));  }  @Test  public void testThatPrettyPrinterProducesParsableOutput() throws IOException {    final SPLPrettyPrinter splPrettyPrinter = new SPLPrettyPrinter(prettyPrinter);    final Optional<ASTSPLFile> root = splFileParser.parse        //("src/test/resources/org/nest/spl/parsing/modelContainingAllLanguageElements.simple");        ("src/test/resources/org/nest/spl/parsing/complexExpressions.simple");    assertTrue(root.isPresent());    root.get().accept(splPrettyPrinter); // starts prettyPrinter    Optional<ASTSPLFile> prettyPrintedRoot = parseStringAsSPLFile(splPrettyPrinter.getResult());    assertTrue(prettyPrintedRoot.isPresent());    System.out.println(splPrettyPrinter.getResult());  }  @Test  public void testAllModelsForParser() throws IOException {    parseAndCheckSPLModelsFromFolder("src/test/resources/org/nest/spl/parsing");  }  @Test  public void testAllModelsForCocos() throws IOException {    parseAndCheckSPLModelsFromFolder("src/test/resources/org/nest/spl/cocos");  }  private void parseAndCheckSPLModelsFromFolder(final String folderPath) throws IOException {    final File parserModelsFolder = new File(folderPath);    for (File splModelFile : parserModelsFolder.listFiles()) {      System.out.println("Handles the model: " + splModelFile.getPath());      final SPLPrettyPrinter splPrettyPrinter = new SPLPrettyPrinter(prettyPrinter);      final Optional<ASTSPLFile> splModelRoot = splFileParser.parse(splModelFile.getPath());      assertTrue("Cannot parse the model: " + splModelFile.getName(), splModelRoot.isPresent());      splModelRoot.get().accept(splPrettyPrinter);      System.out.println(splPrettyPrinter.getResult());      Optional<ASTSPLFile> prettyPrintedRoot = parseStringAsSPLFile(splPrettyPrinter.getResult());      assertTrue(prettyPrintedRoot.isPresent());    }  }}