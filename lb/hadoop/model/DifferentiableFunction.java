package lb.hadoop.model;

import java.util.ArrayList;

public interface DifferentiableFunction {
	public double eval(ArrayList<Double> input, ArrayList<Double> grad);
	public ArrayList<Double> getGrad();
	public double getDirDeriv(ArrayList<Double> dir);
}
