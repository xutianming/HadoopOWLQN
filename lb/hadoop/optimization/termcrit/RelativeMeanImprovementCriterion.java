package lb.hadoop.optimization.termcrit;

public class RelativeMeanImprovementCriterion implements TerminationCriterion{
	
	private int numItersToAvg;
	
	public RelativeMeanImprovementCriterion() {
		this.numItersToAvg = 5;
	}
	
	public RelativeMeanImprovementCriterion(int numItersToAvg) {
		this.numItersToAvg = numItersToAvg;
	}

	@Override
	public double getTermCritValue() {
		// TODO Auto-generated method stub
		return 0;
	}

}
