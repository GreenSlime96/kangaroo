package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.ByteFVComparison;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.feature.MultidimensionalFloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.VLAD;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class RunThree_4 {
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training3", ImageUtilities.FIMAGE_READER);
		
		
		//final VFSListDataset<FImage> query = 
		//		new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		DensePatch dpatch = new DensePatch(4,4,8,8);

		Map<String,FloatCentroidsResult> results=new HashMap<String,FloatCentroidsResult>();
		for(Entry<String, ListDataset<FImage>> entry:splits.getTrainingDataset().entrySet()){
			results.put(entry.getKey(), trainQuantiser(entry.getValue(), dpatch));
		}
		
		//HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(results, dpatch, 200f);
		
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(splits.getTrainingDataset());
		
		System.out.println("here3");

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

	/*
	 * Method to perform K-means clustering on a training set. It returns a Hard Assigner that can
	 * be used on the testing set.
	 */
	static FloatCentroidsResult trainQuantiser(Dataset<FImage> sample,
			DensePatch dpatch) {
		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();

		
		for (int i=0;i<15;i++) {
			FImage image=sample.getRandomInstance();
			dpatch.analyseImage(image);
			LocalFeatureList<FloatKeypoint> t=dpatch.getFloatKeypoints();
			t=(LocalFeatureList<FloatKeypoint>) t.randomSubList(Math.min(100,t.size()));
			allkeys.add(t);
			System.out.println(allkeys.size());
		}
		System.out.println("here");
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		System.out.println(datasource.size());
		FloatCentroidsResult result = km.cluster(datasource);
		return result;
	}
	
	
	static class Extractor implements FeatureExtractor<DoubleFV, FImage> {
		Map<String,FloatCentroidsResult> classCentroids;
		DensePatch dpatch;
		float accuracy;
		float energy;
		int length;
		static int i=0;
		
		public Extractor(Map<String,FloatCentroidsResult> classCentroids, DensePatch dpatch, float accuracy) {
			this.classCentroids = classCentroids;
			this.dpatch=dpatch;
			this.accuracy=accuracy;
		}

		public DoubleFV extractFeature(FImage object) {
			i++;
			System.out.println(i);
			FImage image = object.getImage();
			dpatch.analyseImage(image);
			
			DoubleFV result=new DoubleFV(1);
			
			LocalFeatureList<FloatKeypoint> keypoints=dpatch.getFloatKeypoints();

			int index=0;
			for(Entry<String, FloatCentroidsResult> t:classCentroids.entrySet()){
				DoubleFV c=new DoubleFV(t.getValue().numClusters());
				index=0;
				for(float[] centroid:t.getValue().getCentroids()){
					int t1=0;
					for(FloatKeypoint point:keypoints){
						if(point.getFeatureVector().compare(new FloatFV(centroid), FloatFVComparison.EUCLIDEAN)<accuracy){
							t1++;
						}
					}
					c.setFromDouble(index, t1);
					index++;
				}
				result=result.concatenate(c);
			}

			return result;
		}
	}

}
