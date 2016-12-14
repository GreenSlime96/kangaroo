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

public class RunThree_2 {
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training2", ImageUtilities.FIMAGE_READER);
		
		
		
		//final VFSListDataset<FImage> query = 
		//		new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		DenseSIFT dsift = new DenseSIFT(5, 7);
		//PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

		Map<String,ByteCentroidsResult> results=new HashMap<String,ByteCentroidsResult>();
		for(Entry<String, ListDataset<FImage>> entry:splits.getTrainingDataset().entrySet()){
			results.put(entry.getKey(), trainQuantiser(entry.getValue(), dsift));
		}

		float desiredAccuracy=200f;
		System.out.println(desiredAccuracy);
		Map<String, Integer> accuracies=new HashMap<String,Integer>();
		for(Entry<String, ListDataset<FImage>> entry:splits.getTestDataset().entrySet()){
			System.out.println(entry.getKey());
			int c=0;
			for(FImage img:entry.getValue()){
				dsift.analyseImage(img);
				String result="";
				int max=0;
				for(Entry<String, ByteCentroidsResult> t:results.entrySet()){
					int t1=0;

					for(ByteDSIFTKeypoint point:dsift.getByteKeypoints(0.05f)){
						for(byte[] centroid:t.getValue().getCentroids()){
							if(point.getFeatureVector().compare(new ByteFV(centroid), ByteFVComparison.EUCLIDEAN)<desiredAccuracy){
								t1++;
								break;
							}
						}
					}
					System.out.print(t.getKey()+" "+t1+" ");
					if(t1>max){
						max=t1;
						result=t.getKey();
					}
				}
				System.out.println("");
				if(result.equals(entry.getKey())){
					c++;
				}
			}
			accuracies.put(entry.getKey(), c);
		}
		
		float overall=0;
		for(Entry<String, Integer> accuracy:accuracies.entrySet()){
			System.out.println(accuracy.getKey()+" "+accuracy.getValue()+" "+((float)accuracy.getValue()/25));
			overall+=accuracy.getValue();
		}
		System.out.println("Accuracy:"+(overall/(allData.getGroups().size()*25)));
		System.out.println("");
		
		/*
		HardAssigner<byte[], float[], IntFloatPair> assigner = 
				trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);
	
		System.out.println("here2");
		//PHOW feature extractor (more detailed explanation in the class)
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		*/
		//------------------------------- Perform classification -----------------------------
		/*
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new GISTExtractor());
		//Perform linear classification
		System.out.println("here");
		//LinearSVMAnnotator<FImage, String> ann = new LinearSVMAnnotator<FImage, String>(extractor);
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
		*/
		}

	/*
	 * Method to perform K-means clustering on a training set. It returns a Hard Assigner that can
	 * be used on the testing set.
	 */
	/*
	static ByteCentroidsResult trainQuantiser(Dataset<FImage> sample,
			PyramidDenseSIFT<FImage> pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		
		for (FImage image:sample) {
			pdsift.analyseImage(image);
			LocalFeatureList<ByteDSIFTKeypoint> t=pdsift.getByteKeypoints(0.05f);
			t=(LocalFeatureList<ByteDSIFTKeypoint>) t.randomSubList(Math.min(500,t.size()));
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
	}*/
	
	static ByteCentroidsResult trainQuantiser(Dataset<FImage> sample,
			DenseSIFT dsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		
		for (FImage image:sample) {
			dsift.analyseImage(image);
			LocalFeatureList<ByteDSIFTKeypoint> t=dsift.getByteKeypoints(0.05f);
			t=(LocalFeatureList<ByteDSIFTKeypoint>) t.randomSubList(Math.min(200,t.size()));
			allkeys.add(t);
			System.out.println(allkeys.size());
		}
		System.out.println("here");
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(2000);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println(datasource.size());
		ByteCentroidsResult result = km.cluster(datasource);
		return result;
	}
	
	
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;

		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.pdsift = pdsift;
			this.assigner = assigner;
		}

		public DoubleFV extractFeature(FImage object) {
			FImage image = object.getImage();
			pdsift.analyseImage(image);

			BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

			BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw,
					2, 2);

			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
	}
	
	static class GISTExtractor implements FeatureExtractor<DoubleFV, FImage> {
		Gist<FImage> g;
		static int i=0;

		public GISTExtractor() {
			this.g=new Gist<FImage>();
		}

		public DoubleFV extractFeature(FImage object) {
			i++;
			System.out.println(i);
			FImage image = object.getImage();
			g.analyseImage(image);

			return g.getResponse().normaliseFV();
		}
	}

}
