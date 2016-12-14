package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
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
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Class RunThreeTestDensePatch - Modified implementation for the Bag-of-visual-words linear classifiers 
 * using fixed size densely-sampled pixel patches algorithm.
 * 
 * The introduced improvements are non-linearity through HomogeneousKernelMap and better sampling
 * by taking a fixed amount of images from each group.
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunThreeTestDensePatch {
	
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
		
		//Hard assigner for visual words learned from the pixel patches
		HardAssigner<float[], float[], IntFloatPair> assigner = 
				trainQuantiserGrouped(splits.getTrainingDataset(), densePatch, 10, 10000, 1000);
		
		System.out.println("Assigner trained!");
	
		
		//HomogeneousKernelMap to provide for non-linearity
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		
		//FeatureExtractor based on bag-of-visual-words built from dense pixel patches
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new BOVWDensePatchExtractor(densePatch, assigner));
		
		//train the annotator
		
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(splits.getTrainingDataset());
		
		System.out.println("Linear classifiers trained!");
		
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
	
	/*
	 * This method takes an equal amount of features from each image for the clustering and the same amount of images from each group
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiserGrouped(
			GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset,  DensePatch densePatch, 
			int imagesFromGroup, int clusters, int featuresPerImage)
	{
		//list to contain all pixel patches from our images
		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();
		
		//go over all groups
		for(Entry<String, ListDataset<FImage>> entry:groupedDataset.entrySet()){
			
			//for each group choose a fixed amount of images
			for (int i=0;i<imagesFromGroup;i++) {
				
				FImage img=entry.getValue().getRandomInstance();
				
			    //compute the dense pixel patches
			    densePatch.analyseImage(img);
			   
			    LocalFeatureList<FloatKeypoint> keypoints=densePatch.getFloatKeypoints();
			    
			    //add N random keypoints to the list
			    allkeys.add((LocalFeatureList<FloatKeypoint>) keypoints.randomSubList(Math.min(featuresPerImage,keypoints.size())));
			}
		}
		
		//we get an object ready to perform approximate K-means clustering based on KD Trees
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(clusters);
		
		//get all dense patches  in a Data Source
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		
		//perform the clustering
		FloatCentroidsResult result = km.cluster(datasource);
		
		//get a hard assigner, which can assign new keypoint to a cluster
		return result.defaultHardAssigner();
	}

}
