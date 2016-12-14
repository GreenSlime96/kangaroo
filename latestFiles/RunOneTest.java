package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.pair.IntFloatPair;

/**
 * Class RunOneTest - Used to calculate the best K for use in the K-nearest-neighbour algorithm
 * for the "tiny image" feature. All values of K from 1 to "maxClusters" are considered and the
 * accuracy is average over "maxExecution" independent runs of the algorithm.
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunOneTest {
	
	public static void main(String[] args) throws FileSystemException {
		
		//values of K to consider
		int maxClusters=50;
		
		//average the results over that many executions
		int maxExecution=5;
		
		// ----------------------------------- Load the data -----------------------------------------
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		// ------------------------------------   Execution   ----------------------------------------- 
		
		//Will hold the pairs <Number of Neighbours, Accuracy>
		List<IntFloatPair> pairs=new ArrayList<IntFloatPair>();
		
		for(int i=0;i<maxClusters;i++){
			pairs.add(new IntFloatPair(i, 0));
		}
		
		//loop for as many executions as we require
		for(int execution=0;execution<maxExecution;execution++){
			
			//provide new sets of data at every execution
			final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
			final GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			final GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			// create our FeatureExtractor with the square size being 16
			FeatureExtractor<DoubleFV, FImage> featureExtractor=new TinyImageFeatureExtractor(16);

			// create a KNNAnnotator with no K set
			KNNAnnotator<FImage, String, DoubleFV> annotator=new KNNAnnotator<FImage, String, DoubleFV>(featureExtractor, DoubleFVComparison.EUCLIDEAN);
			
			//train the annotator on the data
			annotator.train(training);
			
			
			System.out.println("Execution:"+(execution+1));
			System.out.println("Calculation for _ neighbours:");
			
			for(int K=1;K<=maxClusters;K++){
				System.out.print(K+" ");
				annotator.setK(K);
				
				//classifier which will give us the accuracy
				ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
						new ClassificationEvaluator<CMResult<String>, String, FImage>(
							annotator, testing, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
				
				//perform the evaluation
				Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
				
				//get the accuracy 
				CMResult<String> result = eval.analyse(guesses);
				pairs.get(K-1).second+=result.getMatrix().getAccuracy();
			}
			System.out.println("");
		}
		
		//Sort the pairs so the one with the highest accuracy comes first
		pairs.sort(new Comparator<IntFloatPair>(){

			@Override
			public int compare(IntFloatPair o1, IntFloatPair o2) {
				float c=o2.second-o1.second;
				if(c>0){
					c=1;
				} else {
					c=-1;
				}
				return (int) c;
			}});
		
		System.out.println("Accuracy:");
		//Display all accuracies while not forgetting to average them
		for(int i=0;i<maxClusters;i++){
			System.out.println((pairs.get(i).first+1)+" "+(pairs.get(i).second/maxExecution));
		}
	}
	

}
