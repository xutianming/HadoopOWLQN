package lb.hadoop.optimization.solution;

import lb.hadoop.model.DifferentiableFunction;
import lb.hadoop.optimization.termcrit.RelativeMeanImprovementCriterion;

public class OWLQN implements OptimizationSolution{
	
	private boolean quiet=false;
	private boolean responsibleForTermCrit;
	
	public RelativeMeanImprovementCriterion termCrit;
	
	public OWLQN(boolean quiet) {
		this.quiet=quiet;
		termCrit = new RelativeMeanImprovementCriterion();
		responsibleForTermCrit = true;
	}

	public OWLQN(RelativeMeanImprovementCriterion termCrit, boolean quiet) {
		this.termCrit = termCrit;
		this.quiet = quiet;
		responsibleForTermCrit = false;
	}
	
	public void setQuiet(boolean quiet) {
		this.quiet = quiet;
	}
	@Override
	public void minimize(DifferentiableFunction func, double[] init,
			double[] res, double tor, double memoryLimit) {
		// TODO Auto-generated method stub
		
	}

}
