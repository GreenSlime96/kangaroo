package uk.ac.soton.ecs.nb4g14.coursework3;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.analyser.ImageAnalyser;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;

public class DensePatch implements ImageAnalyser<FImage>{
	
	private int stepX, stepY, binX, binY;
	
	private LocalFeatureList<FloatKeypoint> features;

	public DensePatch(int stepX, int stepY, int binX, int binY){
		//TODO: check that no value is <= 0
		this.stepX=stepX;
		this.stepY=stepY;
		this.binX=binX;
		this.binY=binY;
		features=new MemoryLocalFeatureList<FloatKeypoint>();
	}

	@Override
	public void analyseImage(FImage image) {
		features=new MemoryLocalFeatureList<FloatKeypoint>();
		
		for(int x=0;x+binX<image.getWidth();x+=stepX){
			for(int y=0;y+binY<image.getHeight();y+=stepY){
				float[] feature= new float[binX*binY];
				int c=0;
				for(int i=x;i<x+binX;i++){
					for(int j=y;j<y+binY;j++){
						feature[c++]=image.pixels[j][i];
					}
				}
				
				// calculate means and standard deviation
				final float mean = mean(feature);

				float m=0;
				// subtract mean and divide by std to standardise
				for (int i = 0; i < feature.length; i++) {
					feature[i] -= mean;
					m+=feature[i]*feature[i];
				}
				m=(float) Math.sqrt(m);
				if(m!=0){
					for (int i = 0; i < feature.length; i++) {
						feature[i] = feature[i]/m;
					}
				}
				
				features.add(new FloatKeypoint(x, y, 0, 1, feature));
			}
		}
	}
	
	private static float mean(float[] array) {
		float average = 0;
		
		for (int i = 0; i < array.length; i++) {
			average += array[i];				
		}
		
		average /= array.length;
		
		return average;
	}
	
	public static float std(float[] array, float mean) {
		double variance = 0;
		
		for (int i = 0; i < array.length; i++) {
			variance += Math.pow(array[i] - mean, 2);
		}
		
		variance /= array.length;
		
		return (float) Math.sqrt(variance);
	}
	
	public LocalFeatureList<FloatKeypoint> getFloatKeypoints(){
		return features;
	}
}
