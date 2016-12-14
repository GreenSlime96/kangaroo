package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
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

/**
 * Class RunTwo - Main implementation for the Bag-of-visual-words linear classifiers 
 * using fixed size densely-sampled pixel patches algorithm.
 * The class RunTwoTest was used to the best parameters to use when classifying the test dataset.
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunTwo {
	
	public static void main(String[] args) throws FileSystemException {
		
		// ----------------------------------- Load the data -----------------------------------------
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		// ------------------------------------   Execution   -----------------------------------------
		
		//instance to provide the fixed-sized densely-sampled pixel patches for an image
		DensePatch densePatch=new DensePatch(4, 4, 8, 8);
		
		//Hard assigner for visual words learned from the pixel patches of a subset of all training data
		HardAssigner<float[], float[], IntFloatPair> assigner = 
				trainQuantiser(GroupedUniformRandomisedSampler.sample(allData, 100), densePatch, 500);
	
		//FeatureExtractor based on bag-of-visual-words built from dense pixel patches
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWDensePatchExtractor(densePatch, assigner);
		
		//train the annotator
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(allData);

		//Annotate testing data
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
			Dataset<FImage> sample, DensePatch densePatch, int clusters)
	{
		//list to contain all pixel patches from our images
		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();
		
		
		for (FImage rec : sample) {
		
		    //compute the dense pixel patches
		    densePatch.analyseImage(rec);
		    
		    //add them to the list
		    allkeys.add(densePatch.getFloatKeypoints());
		}
		
		//we get an object ready to perform approximate K-means clustering based on KD Trees
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(clusters);
		
		//get all Ddense patches  in a Data Source
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		
		//perform the clustering
		FloatCentroidsResult result = km.cluster(datasource);
		
		//get a hard assigner, which can assign new keypoint to a cluster
		return result.defaultHardAssigner();
	}

}
