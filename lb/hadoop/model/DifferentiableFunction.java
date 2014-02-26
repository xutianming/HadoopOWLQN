package lb.hadoop.model;

public interface DifferentiableFunction {
	public double eval(double[] input, double[] grad);
}
