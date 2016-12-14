package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Class RunThreeTestGIST- Test implementation for our run three with the GIST feature.
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunThreeTestGIST {
	public static void main(String[] args) throws FileSystemException {
		
		// ----------------------------------- Load the data -----------------------------------------
		
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
		
		// ------------------------------------   Execution   -----------------------------------------
		
		//HomogeneousKernelMap to provide for non-linearity
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		
		//feature extractor for our class
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new GISTExtractor());
		
		
		//Perform linear classification
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(splits.getTrainingDataset());

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
	
	/**
	 * Class GISTExtractor - a simple extractor for the GIST feature
	 * @author nb4g14 and kbp2g14
	 */
	static class GISTExtractor implements FeatureExtractor<DoubleFV, FImage> {
		Gist<FImage> g;

		public GISTExtractor() {
			this.g=new Gist<FImage>();
		}

		public DoubleFV extractFeature(FImage object) {
			// get the image
			FImage image = object.getImage();
			
			// analyse it with GIST
			g.analyseImage(image);

			// return a normalised DoubleFV
			return g.getResponse().normaliseFV();
		}
	}

}
