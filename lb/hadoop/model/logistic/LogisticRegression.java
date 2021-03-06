package lb.hadoop.model.logistic;

import java.util.ArrayList;

import lb.hadoop.model.DifferentiableFunction;

public class LogisticRegression implements DifferentiableFunction {
	private double l2reg;
	private ArrayList<Double> gradient;
	
	public LogisticRegression(double l2reg) {
		this.l2reg = l2reg;
	}
	
	@Override
	public double eval(ArrayList<Double> input, ArrayList<Double> grad) {
		
		double loss = 1.0;
		//TODO: use int here ?
		for(int i=0; i<input.size(); i++) {
			loss += 0.5 * input.get(i) * input.get(i) * l2reg;
			grad.set(i, l2reg * input.get(i));
		}
		
		double instanceLoss = 0.0;
		try {
			instanceLoss = updateLossAndGrad(input, grad);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		this.gradient = new ArrayList<Double>(grad);
		loss += instanceLoss;
		return loss;
	}
	
	private double updateLossAndGrad(ArrayList<Double> input, ArrayList<Double> grad) throws Exception {
		return (new LRUpdateLoss().run(input, grad));
	}

	@Override
	public ArrayList<Double> getGrad() {
		return this.gradient;
	}

	@Override
	public double getDirDeriv(ArrayList<Double> dir) {
		double res = 0.0;
		if(dir.size() != this.gradient.size()) {
			System.err.println("dir and grad different dimension");
			System.exit(1);
		}
		for(int i=0; i<dir.size(); i++) {
			res += dir.get(i) * this.gradient.get(i);
		}
		return res;
	}

}
