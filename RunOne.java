import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;

public class RunOne {
	static final int TINYIMAGE_LENGTH = 16;
	static final int TINYIMAGE_AREA = TINYIMAGE_LENGTH * TINYIMAGE_LENGTH;
	
	static final ResizeProcessor TINYIMAGE_PROCESSOR = new ResizeProcessor(TINYIMAGE_LENGTH, TINYIMAGE_LENGTH);
	
	public static void main(String[] args) throws FileSystemException {
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>("/Users/khengboonpek/Downloads/training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>("/Users/khengboonpek/Downloads/testing", ImageUtilities.FIMAGE_READER);
		
		// Training and Testing Data
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 90, 0, 10);
		final GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		final GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
		
		// Nearest Neighbour Classifiers
		final float[][] trainingData = new float[training.numInstances()][]; 
		final FloatNearestNeighbours nn = new FloatNearestNeighboursExact(trainingData);
		
		// Store classes and names in array
		final String[] classes = new String[trainingData.length];
		int counter = 0;
			
		// iterate through the classes and their constituent images
		for (String key : training.getGroups()) {
			for (FImage image : training.get(key)) {							
				// get the underlying pixel array
				final float[] pixels = toFeatureVector(image);
				
				// add to KNN Classifier
				classes[counter] = key;
				trainingData[counter] = pixels;
				
				// increment counter
				counter++;
			}
		}
		
		// parameters
		final int K = 30;
		counter = 0;
		
		// test against the trained data
		for (String key : testing.getGroups()) {
			for (FImage image : testing.get(key)) {
				// get the underlying pixel array
				final float[] pixels = toFeatureVector(image);
				
				List<IntFloatPair> value = nn.searchKNN(pixels, K);
				String[] strings = new String[K];
				
				// build list of strings corresponding to class
				for (int i = 0; i < K; i++) {
					strings[i] = classes[value.get(i).first];
				}
				
				// find the most occurring string
				String classified = findPopular(strings);
				
				// count number of corrects
				if (key.equals(classified))
					counter++;				
				
				System.out.println("Image is " + key + ", predicted " + classified);
				
				for (IntFloatPair pair : value) {
					System.out.println(" - " + classes[pair.first] + "\t" + pair.second);
				}
			}
		}
		
		// accuracy readout
		System.out.println(counter + " / " + testing.numInstances() + "\t" + ((double) counter / testing.numInstances()));
		
		// build answer
		for (int i = 0; i < query.size(); i++) {
			final FImage image = query.get(i);
			final String id = query.getID(i);
			
			// get the underlying pixel array
			final float[] pixels = toFeatureVector(image);
			
			List<IntFloatPair> value = nn.searchKNN(pixels, K);
			String[] strings = new String[K];
			
			// build list of strings corresponding to class
			for (int j = 0; j < K; j++) {
				strings[j] = classes[value.get(j).first];
			}
			
			// find the most occurring string
			String classified = findPopular(strings);
			
			System.out.println(id + " " + classified);
		}
	}
	
	/**
	 * Finds the most popular occurence of a string in an array
	 * 
	 * @param strings an array of Strings to find the most popular one of
	 * @return the String which occurs the most, or null if ambiguous
	 */
	public static String findPopular(String[] strings) {
		Map<String, Integer> stringCount = new HashMap<String, Integer>();
		String mostPopular = null;
		int mostOccurring = Integer.MIN_VALUE;
		
		for (String string : strings) {
			final int count = stringCount.getOrDefault(string, 0) + 1;
						
			// update most popular or remove if ambiguous
			if (count > mostOccurring) {
				mostPopular = string;
				mostOccurring = count;				
			} else if (count == mostOccurring) {
				mostPopular = null;
			}
			
			stringCount.put(string, count);
		}
		
		return mostPopular;
	}
	
	/**
	 * Converts the input image into a feature vector. The image is first
	 * cropped into a square along its longest edge and then resized to a 16x16
	 * image. The image's underlying vector is extracted and standardised by the
	 * zero-mean and unit-length by subtracting the means from the image and
	 * dividing the array by the standard deviation.
	 * 
	 * @param image the input image
	 * @return the featurevector corresponding to the image
	 */
	public static float[] toFeatureVector(FImage image) {
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
		
		return pixels;
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
