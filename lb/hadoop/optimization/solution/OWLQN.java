package lb.hadoop.optimization.solution;

import java.util.ArrayList;
import java.util.LinkedList;

import lb.hadoop.model.DifferentiableFunction;
import lb.hadoop.optimization.termcrit.RelativeMeanImprovementCriterion;

public class OWLQN implements OptimizationSolution{
	
	private boolean quiet=false;
	private boolean responsibleForTermCrit;
	
	private ArrayList<Double> x, grad, newX, newGrad, dir;
	private ArrayList<Double> steepestDescDir; // references newGrad to save memory, since we don't ever use both at the same time
	private LinkedList<Double> sList, yList;
	private LinkedList<Double> roList;
	private ArrayList<Double> alphas;
	
	private double value;
	private int iter, m;
	private int dim;
	DifferentiableFunction func;
	double l1weight;


	void MapDirByInverseHessian();
	void UpdateDir();
	double DirDeriv() const;
	void GetNextPoint(double alpha);
	void BackTrackingLineSearch();
	void Shift();
	void MakeSteepestDescDir();
	double EvalL1();
	void FixDirSigns();
	void TestDirDeriv();
	
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
