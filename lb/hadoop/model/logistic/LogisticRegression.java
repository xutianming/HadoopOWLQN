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
		
		double instanceLoss = 0.0;
		try {
			instanceLoss = updateLossAndGrad(input, grad);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		loss += instanceLoss;
		return loss;
	}
	
	private double updateLossAndGrad(double[] input, double[] grad) throws Exception {
		return (new LRUpdateLoss().run(input, grad));
	}

}
