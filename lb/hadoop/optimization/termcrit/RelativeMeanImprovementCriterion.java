package lb.hadoop.optimization.termcrit;

import java.util.LinkedList;

import lb.hadoop.optimization.state.IterationState;

public class RelativeMeanImprovementCriterion implements TerminationCriterion{
	
	@SuppressWarnings("unused")
	private int numItersToAvg;
	private LinkedList<Double> prevVals = new LinkedList<Double>();
	
	public RelativeMeanImprovementCriterion() {
		this.numItersToAvg = 5;
	}
	
	public RelativeMeanImprovementCriterion(int numItersToAvg) {
		this.numItersToAvg = numItersToAvg;
	}

	@Override
	public double getTermCritValue(IterationState state) {
		
		double retVal = Double.POSITIVE_INFINITY;
		if(prevVals.size()>5) {
			double prevVal = prevVals.getFirst();
			if(prevVals.size()==10) {
				prevVals.removeFirst();
			}
			double averageImprovement = (prevVal-state.getIterStateValue())/prevVals.size();
			double relAvgImpr = averageImprovement / Math.abs(state.getIterStateValue());
			retVal = relAvgImpr;
		}
		prevVals.addLast(state.getIterStateValue());
		return retVal;
	}

}
