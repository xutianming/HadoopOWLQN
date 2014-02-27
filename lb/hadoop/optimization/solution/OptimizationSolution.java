package lb.hadoop.optimization.solution;

import java.util.ArrayList;

public interface OptimizationSolution {
	public void minimize(ArrayList<Double> init, ArrayList<Double> res);
}
