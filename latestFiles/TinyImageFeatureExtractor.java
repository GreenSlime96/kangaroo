package uk.ac.soton.ecs.nb4g14.coursework3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * Class TinyImageFeatureExtractor - Extracts a "tiny image" representation of the image.
 * A tiny image is a scaled down version of the square about the center of the image.
 * The parameter passed in the constructor of the class specifies down to what size the image should be scaled.
 * 
 * @author nb4g14 and kbp2g14
 */
public class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, FImage>{
	
	final ResizeProcessor processor;
	
	public TinyImageFeatureExtractor(int length) {
		processor = new ResizeProcessor(length, length);
	}
	
	@Override
	public DoubleFV extractFeature(FImage image) {
		// crop  the image along the shorter axis
		final int length = Math.min(image.getHeight(), image.getWidth());				
		final FImage processed = image.extractCenter(length, length);
		
		// resize the image to the size specified in the constructor
		processor.processImage(processed);
						
		// get the underlying pixel array
		final double[] pixels = processed.getDoublePixelVector();
		
		// calculate means and standard deviation
		final double mean = processed.sum() / pixels.length;
		final double std = std(pixels, mean);

		// subtract mean and divide by std to standardise
		for (int i = 0; i < pixels.length; i++) {
			pixels[i] -= mean;
			pixels[i] /= std;
		}
		
		return new DoubleFV(pixels);
	}	
	
	/**
	 * Gives you STDs through an array
	 * 
	 * @param array the double array representing values
	 * @param mean the mean of the array
	 * @return the standard deviation of values
	 */
	public static double std(double[] array, double mean) {
		double variance = 0;
		
		// sum the differences from mean squared
		for (int i = 0; i < array.length; i++) {
			variance += Math.pow(array[i] - mean, 2);
		}
		
		// divide by N
		variance /= array.length;
		
		return Math.sqrt(variance);
	}

}
