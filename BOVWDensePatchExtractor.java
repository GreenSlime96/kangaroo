package uk.ac.soton.ecs.nb4g14.coursework3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;


public class BOVWDensePatchExtractor implements FeatureExtractor<DoubleFV,FImage>{

	DensePatch densePatch;
	BagOfVisualWords<float[]> bovw;
	
	public BOVWDensePatchExtractor(DensePatch densePatch, HardAssigner<float[], float[], IntFloatPair> assigner) {
		this.densePatch=densePatch;
		bovw = new BagOfVisualWords<float[]>(assigner);
	}
	
	@Override
	public DoubleFV extractFeature(FImage img) {
		densePatch.analyseImage(img);
		return bovw.aggregate(densePatch.getFloatKeypoints()).normaliseFV();
	}

}
