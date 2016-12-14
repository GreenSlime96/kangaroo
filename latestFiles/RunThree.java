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
import org.openimaj.feature.FloatFV;
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

public class RunThree {
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training2", ImageUtilities.FIMAGE_READER);
		
		
		
		//final VFSListDataset<FImage> query = 
		//		new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		
		/*
		HardAssigner<byte[], float[], IntFloatPair> assigner = 
				trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);
	
		System.out.println("here2");
		//PHOW feature extractor (more detailed explanation in the class)
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		*/
		//------------------------------- Perform classification -----------------------------
		
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
		
		}

	/*
	 * Method to perform K-means clustering on a training set. It returns a Hard Assigner that can
	 * be used on the testing set.
	 */
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample,
			PyramidDenseSIFT<FImage> pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (FImage rec : sample) {
			pdsift.analyseImage(rec);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
		}

		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(100);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
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
