import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.pair.IntFloatPair;

public class RunOne_3 {
	public static void main(String[] args) throws FileSystemException {
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>(directory  + "testing", ImageUtilities.FIMAGE_READER);
		
		// create our FeatureExtractor with the square size being 16
		FeatureExtractor<DoubleFV, FImage> featureExtractor = new TinyImageFeatureExtractor(32);

		// create a KNNAnnotator with K = 15
		KNNAnnotator<FImage, String, DoubleFV> ann = 
				new KNNAnnotator<FImage, String, DoubleFV>(featureExtractor, DoubleFVComparison.EUCLIDEAN, 15);
		
		// train the annotator
		ann.train(allData);
		
		for (int i = 0; i < query.size(); i++) {
			FImage image = query.get(i);
			String name = query.getID(i);
			
			// get the maximum confidence annotation
			ScoredAnnotation<String> max = Collections.max(ann.annotate(image));			
			
			System.out.println(name + " " + max.annotation);
		}
	}
	
	static class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
		final ResizeProcessor processor;
		final int imageLength;
		
		public TinyImageFeatureExtractor(int length) {
			processor = new ResizeProcessor(length, length);
			imageLength = length;			
		}
		
		@Override
		public DoubleFV extractFeature(FImage image) {
			// crop  the image along the shortest axis
			final int length = Math.min(image.getHeight(), image.getWidth());				
			final FImage processed = image.extractCenter(length, length);
			
			// resize the image to a 16x16 image
			processor.processImage(processed);
							
			// get the underlying pixel array
			final double[] pixels = processed.getDoublePixelVector();
			
			// calculate means and standard deviation
			final double mean = mean(pixels);
			final double std = std(pixels, mean);

			// subtract mean and divide by std to standardise
			for (int i = 0; i < pixels.length; i++) {
				pixels[i] -= mean;
				pixels[i] /= std;
			}
			
			return new DoubleFV(pixels);
		}
		
		
		private static double mean(double[] array) {
			double average = 0;
			
			for (int i = 0; i < array.length; i++) {
				average += array[i];				
			}
			
			average /= array.length;
			
			return average;
		}
		
		public static double std(double[] array, double mean) {
			double variance = 0;
			
			for (int i = 0; i < array.length; i++) {
				variance += Math.pow(array[i] - mean, 2);
			}
			
			variance /= array.length;
			
			return Math.sqrt(variance);
		}		
	}

}
