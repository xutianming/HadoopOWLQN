package lb.hadoop.optimization.termcrit;

import lb.hadoop.optimization.solution.OptimizationSolution;

public interface TerminationCriterion {
	public double getTermCritValue(OptimizationSolution state);
}
