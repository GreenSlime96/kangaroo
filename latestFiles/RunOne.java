package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.Collections;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

/**
 * Class RunOne - Main implementation of the k-nearest-neighbor "tiny image" algorithm.
 * The class RunOneTest was used to determine the best K parameter (7 although other values were very close).
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunOne {
	public static void main(String[] args) throws FileSystemException {
		
		// ----------------------------------- Load the data -----------------------------------------
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> trainingData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> testingData = 
				new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		// ------------------------------------   Execution   ----------------------------------------- 
		
		// create our FeatureExtractor with the square size being 16
		FeatureExtractor<DoubleFV, FImage> featureExtractor = new TinyImageFeatureExtractor(16);

		// create a KNNAnnotator with K = 7
		KNNAnnotator<FImage, String, DoubleFV> ann = 
				new KNNAnnotator<FImage, String, DoubleFV>(featureExtractor, DoubleFVComparison.EUCLIDEAN, 7);
		
		// train the annotator
		ann.train(trainingData);

		//Annotate testing data
		for (int i = 0; i < testingData.size(); i++) {
			FImage image = testingData.get(i);
			String name = testingData.getID(i);
			
			// get the maximum confidence annotation
			ScoredAnnotation<String> max = Collections.max(ann.annotate(image));			
			
			System.out.println(name + " " + max.annotation);
		}
	}

}
