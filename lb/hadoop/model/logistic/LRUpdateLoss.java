package lb.hadoop.model.logistic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class LRUpdateLoss extends Configured implements Tool {
	
	public static class MyMapper extends
		Mapper<LongWritable, Text, IntWritable, DoubleWritable> {		
		private IntWritable outKey = new IntWritable();
		private DoubleWritable outVal = new DoubleWritable();
		private ArrayList<Double> weights = new ArrayList<Double>();
		
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			String filenameWeight = conf.get("filename.weight");
			try {
				FileSystem fs = FileSystem.get(conf);
				BufferedReader reader = new BufferedReader(
										new InputStreamReader(
										fs.open(new Path(filenameWeight))));
				String line;
				while((line = reader.readLine()) != null) {
					line = line.trim();
					if(line.length() > 0)
						weights.add(Double.parseDouble(line));
				}
				reader.close();
				//fs.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public void map(LongWritable key, Text value, Context context) 
				throws IOException, InterruptedException {
			
			String line = value.toString().trim();
			if(line.length() <=0)
				return;
			String[] parts = line.split("\t");
			
			int label = Integer.parseInt(parts[0].trim());
			// TODO To validate our program ,we do not permit invalid data here
			if(label != -1 && label != 1)
				System.exit(1);
			
			// Calculate sum_(y_i * w_i * x_i) 
			double score = 0.0;
			for(int i=1;i<parts.length;i++) {
				String[] pairs = parts[i].split(":");
				if(pairs.length != 2)
					System.exit(1);
				int index = Integer.parseInt(pairs[0].trim());
				double val = Double.parseDouble(pairs[1].trim());
				score += weights.get(index).doubleValue() * val;
			}
			score *= label;
			
			// Calculate instance loss
			double insLoss, insProb;
			if(score < -30) {
				insLoss = -score;
				insProb = 0;
			} else if(score > 30) {
				insLoss = 0;
				insProb = 1;
			} else {
				double temp = 1 + Math.exp(-score);
				insLoss = Math.log(temp);
				insProb = 1 / temp;
			}
			
			// Generate changes on every index
			for(int i=1;i<parts.length;i++) {
				String[] pairs = parts[i].split(":");
				if(pairs.length != 2)
					System.exit(1);
				int index = Integer.parseInt(pairs[0].trim());
				double val = Double.parseDouble(pairs[1].trim());
				outKey.set(index);
				outVal.set(label * -1 * (1-insProb) * val);
				context.write(outKey, outVal);
			}
			
			outKey.set(-1);
			outVal.set(insLoss);
			context.write(outKey, outVal);
		}
	}
	
	public static class MyReducer extends
		Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
		
		private IntWritable key = new IntWritable();
		private DoubleWritable val = new DoubleWritable();
		private ArrayList<Double> grad = new ArrayList<Double>();
		
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			String filenameGrad = conf.get("filename.grad");
			try {
				FileSystem fs = FileSystem.get(conf);
				BufferedReader br = new BufferedReader(
										new InputStreamReader(
										fs.open(new Path(filenameGrad))));
				String line;
				while((line = br.readLine()) != null) {
					line = line.trim();
					if(line.length() > 0)
						grad.add(Double.parseDouble(line));
				}
				br.close();
				//fs.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public void reduce(IntWritable k, Iterable<DoubleWritable> vs, Context context) 
				throws IOException, InterruptedException {
			
			if(k.get() == -1) {
				double loss = 0.0;
				for(DoubleWritable v : vs) {
					loss += v.get();
				}
				key.set(-1);
				val.set(loss);
				context.write(key, val);
			} else {
				int index = k.get();
				for(DoubleWritable v : vs) {
					grad.set(index, grad.get(index) +  v.get());
				}
				key.set(index);
				val.set(grad.get(index));
				context.write(key, val);
			}
		}
	}

	private String filenameTrainSet;
	private String filenameWeight;
	private String filenameLossAndGrad;
	private String filenameGrad;
	
	@Override
	public int run(String[] arg0) throws Exception {
		
		Configuration conf = getConf();
		Job job = new Job(conf, "Calculate Loss");
		job.setJarByClass(LRUpdateLoss.class);
		job.setMapperClass(MyMapper.class);
		job.setReducerClass(MyReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(DoubleWritable.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job, new Path(this.filenameTrainSet));
		FileOutputFormat.setOutputPath(job, new Path(this.filenameLossAndGrad));
		
		conf = job.getConfiguration();
		conf.set("filename.trainset", this.filenameTrainSet);
		conf.set("filename.weight", this.filenameWeight);
		conf.set("filename.grad", this.filenameGrad);
		conf.set("filename.lossandgrad", this.filenameLossAndGrad);
		
		if (!job.waitForCompletion(true))
        {
        	return 1;
        } else {
        	return 0;
        }
	}
	
	public double run(ArrayList<Double> input, ArrayList<Double> grad) throws Exception {
		
		this.filenameTrainSet = "data_matrix.txt";
		this.filenameWeight = "input.txt";
		this.filenameGrad = "grad.txt";
		this.filenameLossAndGrad  = "result";
		
		//  Write input and grad into file as input of hadoop
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		Path weightPath = new Path(this.filenameWeight);
		if(fs.exists(weightPath))
			fs.delete(weightPath, true);
		
		BufferedWriter bw1 = new BufferedWriter(new OutputStreamWriter(
				fs.create(weightPath)));
		for(int i=0; i<input.size(); i++) {
			bw1.write(String.valueOf(input.get(i)));
			bw1.newLine();
		}
		bw1.flush();
		bw1.close();
		
		Path gradPath = new Path(this.filenameGrad);
		if(fs.exists(gradPath))
			fs.delete(gradPath, true);
		BufferedWriter bw2 = new BufferedWriter(new OutputStreamWriter(
				fs.create(gradPath)));
		for(int i=0; i<grad.size(); i++) {
			bw2.write(String.valueOf(grad.get(i)));
			bw2.newLine();
		}
		bw2.flush();
		bw2.close();
		fs.close();
		
		// Run hadoop job		
		int retcd = ToolRunner.run(this, null);
		if(retcd != 0) {
			System.err.println("Hadoop job failed.");
		}
		// Update grad using hadoop output and return loss
		fs = FileSystem.get(conf);
		BufferedReader br = new BufferedReader(new InputStreamReader(
				fs.open(new Path(this.filenameLossAndGrad+"/part-r-00000"))));
		String line;
		double loss = 0.0;
		while((line = br.readLine()) != null) {
			line = line.trim();
			if(line.length() > 0) {
				String[] pairs = line.split("\t");
				int index = Integer.parseInt(pairs[0].trim());
				double val = Double.parseDouble(pairs[1].trim());
				if(index == -1) {
					loss = val;
				} else {
					grad.set(index, val);
				}
			}
		}
		fs.deleteOnExit(new Path(this.filenameLossAndGrad));
		br.close();
		fs.close();
		return loss;
	}

}
