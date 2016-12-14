package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Class RunTwoTest - Test class for the Bag-of-visual-words linear classifiers 
 * using fixed size densely-sampled pixel patches algorithm.
 * It is used to estimate the best parameters to use when classifying the test dataset.
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunTwoTest {
	
	public static void main(String[] args) throws FileSystemException {
		
		// ----------------------------------- Load the data -----------------------------------------
		
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		
		// ------------------------------------   Execution   -----------------------------------------
		
		System.out.println("Images loaded!");
		
		//instance to provide the fixed-sized densely-sampled pixel patches for an image
		DensePatch densePatch=new DensePatch(4, 4, 8, 8);
		
		//Hard assigner for visual words learned from the pixel patches of a subset of the training data
		HardAssigner<float[], float[], IntFloatPair> assigner = 
				RunTwo.trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 100), densePatch);
	
		System.out.println("Assigner trained!");
		
		//FeatureExtractor based on bag-of-visual-words built from dense pixel patches
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWDensePatchExtractor(densePatch, assigner);
		
		//train the annotator
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(splits.getTrainingDataset());
		
		System.out.println("Linear classifiers trained!");

		//------------------------------------ Evaluate --------------------------------------
		
		//Construct an evaluator
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
				new ClassificationEvaluator<CMResult<String>, String, FImage>(
					ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
					
		//perform the evaluation
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);
		
		//print the report
		System.out.println(result.getDetailReport());
	}


}
