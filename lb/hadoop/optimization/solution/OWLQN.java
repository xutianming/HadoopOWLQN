package lb.hadoop.optimization.solution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;

import lb.hadoop.model.DifferentiableFunction;
import lb.hadoop.optimization.state.OWLQNIterationState;
import lb.hadoop.optimization.termcrit.RelativeMeanImprovementCriterion;
import lb.hadoop.optimization.termcrit.TerminationCriterion;

public class OWLQN implements OptimizationSolution {
	
	@SuppressWarnings("unused")
	private boolean quiet = false;
	@SuppressWarnings("unused")
	private boolean responsibleForTermCrit;
	
	private ArrayList<Double> x, grad, newX, newGrad;
	private ArrayList<Double> steepestDescDir, dir; 
	private LinkedList<ArrayList<Double>> sList, yList;
	private LinkedList<Double> roList;
	private ArrayList<Double> alphas;
	private int iter, m;
	private TerminationCriterion termCrit;
	private DifferentiableFunction func;
	private OWLQNIterationState state;
	private double value;
	private int dim;
	private double tor;

	public OWLQN(boolean quiet, DifferentiableFunction func, int dimension, int memoryLimit, double tor) {
		this.quiet=quiet;
		this.termCrit = new RelativeMeanImprovementCriterion();
		this.state = new OWLQNIterationState();
		this.responsibleForTermCrit = true;
		this.func = func;
		this.dim = dimension;
		this.m = memoryLimit;
		this.x = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.grad = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.newX = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.newGrad = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.steepestDescDir = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.sList = new LinkedList<ArrayList<Double>>();
		this.yList = new LinkedList<ArrayList<Double>>();
		this.roList = new LinkedList<Double>();
		this.alphas = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.dir = new ArrayList<Double>(Collections.nCopies(dimension, 0.0));
		this.tor = tor;
	}
	/** temporary unused
	public OWLQN(TerminationCriterion termCrit, DifferentiableFunction func, boolean quiet) {
		this.termCrit = termCrit;
		this.quiet = quiet;
		this.responsibleForTermCrit = false;
		this.func = func;
	}
	*/
	
	private void updateDir() {
		makeSteepestDescDir();
		mapDirByInverseHessian();
		fixDirSigns();
	}
	
	private void makeSteepestDescDir() {
		ArrayList<Double> grads = func.getGrad();
		for (int i=0; i<dim; i++) {
			dir.set(i,0-grads.get(i));
		}
		steepestDescDir = new ArrayList<Double>(dir);
	}
	
	private void mapDirByInverseHessian() {
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
	
	private void fixDirSigns() {
		for (int i = 0; i<dim; i++) {
			if (dir.get(i) * steepestDescDir.get(i) <= 0) {
				dir.set(i,0.0);
			}
		}
	}
	
	private void getNextPoint(double alpha) {
		addMultInto(newX, x, dir, alpha);
		for (int i=0; i<dim; i++) {
			if (x.get(i) * newX.get(i) < 0.0) {
					newX.set(i,0.0);
			}
		}
	}
	
	public void backTrackingLineSearch() {
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
			getNextPoint(alpha);
			value = func.eval(newX, newGrad);
			if (value <= oldValue + c1 * origDirDeriv * alpha) {
				state.setIterStateValue(value);
				break;
			}

			alpha *= backoff;
		}

	}
	
	public void shift() {
		ArrayList<Double> nextS = null,nextY = null;

        int listSize = (int)sList.size();

        if (listSize < m-1) {
                nextS = new ArrayList<Double>(dim);
                nextY = new ArrayList<Double>(dim);
        }

        if (nextS == null) {
                nextS = sList.getFirst();
                sList.removeFirst();
                nextY = yList.getFirst();
                yList.removeFirst();
                roList.removeFirst();
        }

        addMultInto(nextS, newX, x, -1);
        addMultInto(nextY, newGrad, grad, -1);
        double ro = dotProduct(nextS, nextY);

        sList.addLast(nextS);
        yList.addLast(nextY);
        roList.addLast(ro);

        x = new ArrayList<Double>(newX);
        grad = new ArrayList<Double>(newGrad);

        state.increaseIter();
	}
	
	@Override
	public void minimize(ArrayList<Double> init, ArrayList<Double> res) {
		value = func.eval(init, grad);  // Loss
		state.setIterStateValue(value); // Use loss as iteration state value
        System.out.println("Iter:" + state.getIter());
        System.out.println("Loss:" + state.getIterStateValue());
		termCrit.getTermCritValue(state);
        while (true) {
        	updateDir();
            backTrackingLineSearch();
            double termCritVal = termCrit.getTermCritValue(state);
            if(termCritVal < tor)
            	break;
            shift();
            System.out.println("Iter:" + state.getIter());
            System.out.println("Loss:" + state.getIterStateValue());
            System.out.println("TermCrit:" + termCritVal);
        }
        res = newX;
		
	}

	private double dotProduct(ArrayList<Double> a, ArrayList<Double> b) {
		double result = 0;
		for (int i=0; i<a.size(); i++) {
			result += a.get(i) * b.get(i);
		}
		return result;
	}

	private void addMult(ArrayList<Double>a, ArrayList<Double> b, double c) {
		for (int i=0; i<a.size(); i++) {
			a.set(i, a.get(i)+b.get(i)*c);
		}
	}

	@SuppressWarnings("unused")
	private void add(ArrayList<Double> a, ArrayList<Double> b) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,a.get(i)+b.get(i));
		}
	}

	private void addMultInto(ArrayList<Double> a, ArrayList<Double>  b, ArrayList<Double> c, double d) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,b.get(i)+c.get(i)*d);
		}
	}

	private void scale(ArrayList<Double> a, double b) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,a.get(i)*b);
		}
	}

	@SuppressWarnings("unused")
	private void scaleInto(ArrayList<Double> a, ArrayList<Double> b, double c) {
		for (int i=0; i<a.size(); i++) {
			a.set(i,b.get(i)*c);
		}
	}

	public void setQuiet(boolean quiet) {
		this.quiet = quiet;
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
		return this.x.size();
	}
}
