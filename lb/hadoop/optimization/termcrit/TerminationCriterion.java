package lb.hadoop.optimization.termcrit;

import lb.hadoop.optimization.state.IterationState;

public interface TerminationCriterion {
	public double getTermCritValue(IterationState state);
}
