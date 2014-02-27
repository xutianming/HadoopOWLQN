package lb.hadoop.optimization.termcrit;

import java.util.LinkedList;

import lb.hadoop.optimization.solution.OWLQN;
import lb.hadoop.optimization.solution.OptimizationSolution;

public class RelativeMeanImprovementCriterion implements TerminationCriterion{
	
	private int numItersToAvg;
	private LinkedList<Double> prevVals;
	
	public RelativeMeanImprovementCriterion() {
		this.numItersToAvg = 5;
	}
	
	public RelativeMeanImprovementCriterion(int numItersToAvg) {
		this.numItersToAvg = numItersToAvg;
	}

	@Override
	public double getTermCritValue(OptimizationSolution state) {
		
		// TODO Auto-generated method stub
		double retVal = Double.POSITIVE_INFINITY;
		if(prevVals.size()>5) {
			double prevVal = prevVals.getFirst();
			if(prevVals.size()==10) {
				prevVals.removeFirst();
			}
			double averageImprovement = (prevVal-state.GetValue())/prevVals.size();
			double relAvgImpr = averageImprovement / Math.abs(state.GetValue());
			retVal = relAvgImpr;
			
		}
		return 0;
	}

}
