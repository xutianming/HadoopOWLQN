package lb.hadoop.optimization.solution;

import java.util.ArrayList;
import java.util.Collections;

import lb.hadoop.model.L1RegularizedLR;

public class Script {

	public static void main(String[] args) {
		L1RegularizedLR lr = new L1RegularizedLR(1.0, 0.0);
		double tor = 1.0e-4;
		int memoryLimit = 10;
		int dimension = 80;
		ArrayList<Double> init = new ArrayList<Double>(Collections.nCopies(dimension, new Double(0)));
		ArrayList<Double> res = new ArrayList<Double>(Collections.nCopies(dimension, new Double(0)));
		OWLQN owlqn = new OWLQN(false, lr, dimension, memoryLimit, tor);
		owlqn.minimize(init, res);
		System.out.println("Print vector");
		for(int i=0; i<res.size(); i++) {
			System.out.println(res.get(i));
		}
		System.out.println("Script end");
	}

}
