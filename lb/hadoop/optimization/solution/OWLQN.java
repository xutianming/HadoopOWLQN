package lb.hadoop.optimization.solution;

import java.util.ArrayList;
import java.util.LinkedList;

import lb.hadoop.model.DifferentiableFunction;
import lb.hadoop.model.L1RegularizedLR;
import lb.hadoop.optimization.termcrit.RelativeMeanImprovementCriterion;

public class OWLQN implements OptimizationSolution{
	
	private boolean quiet=false;
	private boolean responsibleForTermCrit;
	
	private ArrayList<Double> x, grad, newX, newGrad, dir;
	private ArrayList<Double> steepestDescDir; // references newGrad to save memory, since we don't ever use both at the same time
	private LinkedList<ArrayList<Double>> sList, yList;
	private LinkedList<Double> roList;
	private ArrayList<Double> alphas;
	
	private double value;
	private int iter, m;
	private int dim;
	L1RegularizedLR func;
	double l1weight;


	public void UpdateDir() {
		
	}
	
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
	
	public void MakeSteepestDescDir() {
		if (this.l1weight == 0) {
			scaleInto(dir, grad, -1);
		} else {
			ArrayList<Double> grads = func.getGrad();
			for (int i=0; i<dim; i++) {
				dir.set(i,0-grads.get(i));
			}
		}
		steepestDescDir = dir;
	}
	
	public void MapDirByInverseHessian() {
		int count = sList.size();

		if (count != 0) {
			for (int i = count - 1; i >= 0; i--) {
				alphas.set(i, 0-dotProduct(sList.get(i),dir)/roList.get(i));
				addMult(dir, yList.get(i), alphas.get(i));
			}

			ArrayList<Double> lastY = yList.get(count-1);
			double yDotY = dotProduct(lastY, lastY);
			double scalar = roList.get(count-1) / yDotY;
			scale(dir, scalar);

			for (int i = 0; i < count; i++) {
				double beta = dotProduct(yList.get(i), dir) / roList.get(i);
				addMult(dir, sList.get(i), 0-alphas.get(i)- beta);
			}
		}
	}
	
	public void FixDirSigns() {
		if (l1weight > 0) {
			for (int i = 0; i<dim; i++) {
				if (dir.get(i) * steepestDescDir.get(i) <= 0) {
					dir.set(i,0.0);
				}
			}
		}
	}
	
	public void GetNextPoint(double alpha) {
		addMultInto(newX, x, dir, alpha);
		if (l1weight > 0) {
			for (int i=0; i<dim; i++) {
				if (x.get(i) * newX.get(i) < 0.0) {
					newX.set(i,0.0);
				}
			}
		}
	}
	
	public void BackTrackingLineSearch() {
		double origDirDeriv = func.getDirDeriv(dir);
		// if a non-descent direction is chosen, the line search will break anyway, so throw here
		// The most likely reason for this is a bug in your function's gradient computation
		if (origDirDeriv >= 0) {
			System.exit(1);
		}

		double alpha = 1.0;
		double backoff = 0.5;
		if (iter == 1) {
			//alpha = 0.1;
			//backoff = 0.5;
			double normDir = Math.sqrt(dotProduct(dir, dir));
			alpha = (1 / normDir);
			backoff = 0.1;
		}

		double c1 = 1e-4;
		double oldValue = value;

		while (true) {
			GetNextPoint(alpha);
			value = func.eval(newX, newGrad);

			if (value <= oldValue + c1 * origDirDeriv * alpha) break;

			alpha *= backoff;
		}

	}
	
	public void Shift() {
		ArrayList<Double> nextS,nextY;

		int listSize = (int)sList.size();

		if (listSize < m) {
			nextS = new ArrayList<Double>(dim);
			nextY = new ArrayList<Double>(dim);
		}

		if (nextS == null) {
			nextS = sList.getFirst();
			sList.;
			nextY = yList.front();
			yList.pop_front();
			roList.pop_front();
		}

		addMultInto(*nextS, newX, x, -1);
		addMultInto(*nextY, newGrad, grad, -1);
		double ro = dotProduct(*nextS, *nextY);

		sList.push_back(nextS);
		yList.push_back(nextY);
		roList.push_back(ro);

		x.swap(newX);
		grad.swap(newGrad);

		iter++;
	}
	
	public static double dotProduct(ArrayList<Double> a, ArrayList<Double> b) {
		double result = 0;
		for (int i=0; i<a.size(); i++) {
			result += a.get(i) * b.get(i);
		}
		return result;
	}

	public static void addMult(ArrayList<Double>a, ArrayList<Double> b, double c) {
		for (int i=0; i<a.size(); i++) {
			a.set(i, a.get(i)+b.get(i)*c);
		}
	}

	public static void add(ArrayList<Double> a, ArrayList<Doule> b) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,a.get(i)+b.get(i));
		}
	}

	public static void addMultInto(ArrayList<Double> a, ArrayList<Double>  b, ArrayList<Double> c, double d) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,b.get(i)+c.get(i)*d);
		}
	}

	public static void scale(ArrayList<Double> a, double b) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,a.get(i)*b);
		}
	}

	public static void scaleInto(ArrayList<Double> a, ArrayList<Double> b, double c) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,b.get(i)*c);
		}
	}

	public ArrayList<Double> GetX() {
		return this.newX;
	}

	public ArrayList<Double> GetLastX() {
		return this.x;
	}

	public ArrayList<Double> GetGrad() {
		return this.newGrad;
	}

	public ArrayList<Double> GetLastGrad() {
		return this.grad;
	}

	public ArrayList<Double> GetLastDir() {
		return this.dir;
	}

	public double GetValue() {
		return this.value;
	}
	
	public int GetIter() {
		return this.iter;
	}
	
	public int GetDim() {
		return this.dim;
	}
	
	@Override
	public void minimize(DifferentiableFunction func, double[] init,
			double[] res, double tor, double memoryLimit) {
		// TODO Auto-generated method stub
		
	}

}
