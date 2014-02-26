package lb.hadoop.optimization.state;

public abstract class IterationState {
	private double lossFuncValue;
	private int iter;
	
	public double getIterStateValue() {
		return this.lossFuncValue;
	}
	
	public void setIterStateValue(double value) {
		this.lossFuncValue = value;
	}
	
	public void increaseIter() {
		this.iter += 1;
	}
	
	public int getIter() {
		return this.iter;
	}
}
