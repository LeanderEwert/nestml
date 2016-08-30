<#--Dynamics implements BodyElement = "dynamics" (MinDelay | TimeStep) "(" Parameter ")"
                                        BLOCK_OPEN! Block BLOCK_CLOSE!;-->


for ( long lag = from ; lag < to ; ++lag ) {

  ${tc.include("org.nest.spl.Block", ast.getBlock())}

  // voltage logging
  B_.logger_.record_data(origin.get_steps()+lag);
}
