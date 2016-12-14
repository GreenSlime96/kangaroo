package uk.ac.soton.ecs.nb4g14.coursework3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

/**
 * Class BOVWDensePatchExtractor - Simple FeatureExtractor, which aggregates fixed-sized densely-sampled pixel patches
 * into a bag-of-visual-words.
 * 
 * @author nb4g14 and kbp2g14
 */
public class BOVWDensePatchExtractor implements FeatureExtractor<DoubleFV,FImage>{

	DensePatch densePatch;
	BagOfVisualWords<float[]> bovw;
	
	public BOVWDensePatchExtractor(DensePatch densePatch, HardAssigner<float[], float[], IntFloatPair> assigner) {
		this.densePatch=densePatch;
		
		//the bag-of-visual-word has to be given an assigner, which will take FloatKeypoints to clusters/visual words
		bovw = new BagOfVisualWords<float[]>(assigner);
	}
	
	@Override
	public DoubleFV extractFeature(FImage img) {
		densePatch.analyseImage(img);
		return bovw.aggregate(densePatch.getFloatKeypoints()).normaliseFV();
	}

}
