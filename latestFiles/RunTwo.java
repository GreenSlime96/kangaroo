package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.Collections;
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
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class RunTwo {
	
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		
		DensePatch densePatch=new DensePatch(4, 4, 8, 8);
		
		HardAssigner<float[], float[], IntFloatPair> assigner = 
				trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 100), densePatch);
	
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWDensePatchExtractor(densePatch, assigner);
		
		//------------------------------- Perform classification -----------------------------
		//Perform linear classification
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(allData);

		for (int i = 0; i < query.size(); i++) {
			FImage image = query.get(i);
			String name = query.getID(i);
			
			// get the maximum confidence annotation
			ScoredAnnotation<String> max = Collections.max(ann.annotate(image));			
			
			System.out.println(name + " " + max.annotation);
		}
	}

	/*
	 * Method to perform K-means clustering on a training set. It returns a Hard Assigner that can
	 * be used on the testing set.
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
	        Dataset<FImage> sample, DensePatch densePatch)
	{
		//list to contain all features from our images
		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();
		
		for (FImage rec : sample) {
		
		    //compute the DSIFT descriptors
		    densePatch.analyseImage(rec);
		    //add all SIFT descriptors to the list
		    allkeys.add(densePatch.getFloatKeypoints());
		}
		
		//get the first max 10000 elements
		if (allkeys.size() > 10000)
		    allkeys = allkeys.subList(0, 10000);
		
		//we get an object ready to perform approximate K-means clustering based on KD Trees
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(2000);
		
		//get all SIFT features in a Data Source
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		
		//perform the clustering
		FloatCentroidsResult result = km.cluster(datasource);
		
		for(float[] centroid:result.centroids){
			for(int j=0;j<centroid.length;j++){
				System.out.println(centroid[j]+" ");
			}
			System.out.println("");
		}
		
		//get a hard assigner, which can assign new points to a cluster
		return result.defaultHardAssigner();
	}

}
