package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class RunTwoTest {
	
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		
		System.out.println("Images loaded!");
		DensePatch densePatch=new DensePatch(4, 4, 8, 8);
		
		HardAssigner<float[], float[], IntFloatPair> assigner = 
				RunTwo.trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 100), densePatch);
	
		System.out.println("Assigner trained!");
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWDensePatchExtractor(densePatch, assigner);
		
		//------------------------------- Perform classification -----------------------------
		//Perform linear classification
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
