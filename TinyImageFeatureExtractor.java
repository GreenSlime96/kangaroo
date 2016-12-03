package uk.ac.soton.ecs.nb4g14.coursework;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImageFeatureExtractor implements FeatureExtractor<FloatFV, FImage>{

	static final int TINYIMAGE_LENGTH = 16;
	static final int TINYIMAGE_AREA = TINYIMAGE_LENGTH * TINYIMAGE_LENGTH;
	
	static final ResizeProcessor TINYIMAGE_PROCESSOR = new ResizeProcessor(TINYIMAGE_LENGTH, TINYIMAGE_LENGTH);
	
	/*
	@Override
	public FloatFV extractFeature(FImage image) {
		// crop  the image along the shortest axis
				final int length = Math.min(image.getHeight(), image.getWidth());				
				final FImage processed = image.extractCenter(length, length);
				
				// resize the image to a 16x16 image
				TINYIMAGE_PROCESSOR.processImage(processed);
				// get the underlying pixel array
				final float[] pixels = processed.getFloatPixelVector();
				
				
				// calculate means and standard deviation
				final float mean = processed.sum() / pixels.length;

				// subtract mean and get magnitude
				float m=0;
				for (int i = 0; i < pixels.length; i++) {
					pixels[i] = pixels[i]-mean;
					m+=pixels[i]*pixels[i];
				}
				m=(float) Math.sqrt(m);
				if(m!=0){
					for (int i = 0; i < pixels.length; i++) {
						pixels[i] = pixels[i]/m;
					}
				}
				
				FloatFV floats=new FloatFV(pixels);
				return floats;
	}
	*/
	
	@Override
	public FloatFV extractFeature(FImage image) {
		// crop  the image along the shortest axis
		final int length = Math.min(image.getHeight(), image.getWidth());				
		final FImage processed = image.extractCenter(length, length);
		
		// resize the image to a 16x16 image
		TINYIMAGE_PROCESSOR.processImage(processed);
						
		// get the underlying pixel array
		final float[] pixels = processed.getFloatPixelVector();
		
		// calculate means and standard deviation
		final float mean = processed.sum() / pixels.length;
		final double std = std(pixels, mean);

		// subtract mean and divide by std to standardise
		for (int i = 0; i < pixels.length; i++) {
			pixels[i] -= mean;
			pixels[i] /= std;
		}
		FloatFV floats=new FloatFV(pixels);
		return floats;
	}
	
	/**
	 * Gives you STDs through an array
	 * 
	 * @param array the float array representing values
	 * @param mean the mean of the array
	 * @return the standard deviation of values
	 */
	public static double std(float[] array, float mean) {
		float variance = 0;
		
		// sum the differences from mean squared
		for (int i = 0; i < array.length; i++) {
			variance += Math.pow(array[i] - mean, 2);
		}
		
		// divide by N
		variance /= array.length;
		
		return Math.sqrt(variance);
	}

}
