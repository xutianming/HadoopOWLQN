package lb.hadoop.model.logistic;

import lb.hadoop.model.DifferentiableFunction;

public class LogisticRegression implements DifferentiableFunction {
	private double l2reg;
	
	public LogisticRegression(double l2reg) {
		this.l2reg = l2reg;
	}
	
	@Override
	public double eval(double[] input, double[] grad) {
		
		double loss = 1.0;
		//TODO: use int here ?
		for(int i=0; i<input.length; i++) {
			loss += 0.5 * input[i] * input[i] * l2reg;
			grad[i] = l2reg * input[i];
		}
		
		double instanceLoss = updateLossAndGrad(input, grad);
		loss += instanceLoss;
		return loss;
	}
	
	private double updateLossAndGrad(double[] input, double[] grad) {
		return (new LRUpdateLoss().run(input, grad));
	}

}
