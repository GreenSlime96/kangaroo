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

public class RunThree_3 {
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training3", ImageUtilities.FIMAGE_READER);
		
		
		//final VFSListDataset<FImage> query = 
		//		new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		DenseSIFT dsift = new DenseSIFT(5, 7);

		Map<String,ByteCentroidsResult> results=new HashMap<String,ByteCentroidsResult>();
		for(Entry<String, ListDataset<FImage>> entry:splits.getTrainingDataset().entrySet()){
			results.put(entry.getKey(), trainQuantiser(entry.getValue(), dsift));
		}
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(results, dsift, 200f, 0.4f);
		
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
	static ByteCentroidsResult trainQuantiser(Dataset<FImage> sample,
			DenseSIFT dsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		
		for (int i=0;i<15;i++) {
			FImage image=sample.getRandomInstance();
			dsift.analyseImage(image);
			LocalFeatureList<ByteDSIFTKeypoint> t=dsift.getByteKeypoints(0.4f);
			t=(LocalFeatureList<ByteDSIFTKeypoint>) t.randomSubList(Math.min(100,t.size()));
			allkeys.add(t);
			System.out.println(allkeys.size());
		}
		System.out.println("here");
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println(datasource.size());
		ByteCentroidsResult result = km.cluster(datasource);
		return result;
	}
	
	
	static class Extractor implements FeatureExtractor<DoubleFV, FImage> {
		Map<String,ByteCentroidsResult> classCentroids;
		DenseSIFT dsift;
		float accuracy;
		float energy;
		int length;
		static int i=0;
		
		public Extractor(Map<String,ByteCentroidsResult> classCentroids, DenseSIFT dsift, float accuracy, float energy) {
			this.classCentroids = classCentroids;
			this.dsift=dsift;
			this.accuracy=accuracy;
			this.energy=energy;
		}

		public DoubleFV extractFeature(FImage object) {
			i++;
			System.out.println(i);
			FImage image = object.getImage();
			dsift.analyseImage(image);
			
			DoubleFV result=new DoubleFV(classCentroids.size());
			
			LocalFeatureList<ByteDSIFTKeypoint> keypoints=dsift.getByteKeypoints(energy);

			int index=0;
			for(Entry<String, ByteCentroidsResult> t:classCentroids.entrySet()){
				int t1=0;
				for(ByteDSIFTKeypoint point:keypoints){
					for(byte[] centroid:t.getValue().getCentroids()){
						if(point.getFeatureVector().compare(new ByteFV(centroid), ByteFVComparison.EUCLIDEAN)<accuracy){
							t1++;
							break;
						}
					}
				}
				result.setFromDouble(index++, t1);
			}

			return result;
		}
	}

}
