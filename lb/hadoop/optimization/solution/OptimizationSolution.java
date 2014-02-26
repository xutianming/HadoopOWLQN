package lb.hadoop.optimization.solution;

import lb.hadoop.model.DifferentiableFunction;

public interface OptimizationSolution {
	public void minimize(DifferentiableFunction func, double[] init, double[] res, 
			double tor, double memoryLimit);
}
