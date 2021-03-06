package lb.hadoop.model;

import java.util.ArrayList;

import lb.hadoop.model.logistic.LogisticRegression;

public class L1RegularizedLR implements DifferentiableFunction {
	// TODO L1REG must be positive
	private double l1reg;
	private double l2reg;
	private ArrayList<Double> gradient;
	
	public L1RegularizedLR(double l1reg, double l2reg) {
		this.l1reg = l1reg;
		this.l2reg = l2reg;
	}
	
	@Override
	public double eval(ArrayList<Double> input, ArrayList<Double> grad) {
		LogisticRegression lr = new LogisticRegression(l2reg);
		double loss = lr.eval(input, grad);
		for(int i=0; i<input.size(); i++) {
			loss += Math.abs(input.get(i)) * l1reg;
		}
		this.gradient = new ArrayList<Double>(grad);
		updateGrad(input);
		return loss;
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
	
	private void updateGrad( ArrayList<Double> input ) {
		for(int i=0; i<input.size(); i++) {
			if(input.get(i) < 0) {
				this.gradient.set(i, this.gradient.get(i)-l1reg);
			} else if(input.get(i) > 0) {
				this.gradient.set(i, this.gradient.get(i)+l1reg);
			} else {
				if(this.gradient.get(i) + l1reg < 0) {
					this.gradient.set(i, this.gradient.get(i)+l1reg);
				} else if(this.gradient.get(i) -l1reg >0) {
					this.gradient.set(i, this.gradient.get(i)-l1reg);
				} else {
					this.gradient.set(i, 0.0);
				}
			}
		}
	}

}
