package uk.ac.soton.ecs.nb4g14.coursework3;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.analyser.ImageAnalyser;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;

/**
 * Class DensePatch - Image analyser that provide fixed size densely-sampled pixel patches algorithm
 * in the form of FloatKeypoints.
 * 
 * @author nb4g14 and kbp2g14
 */
public class DensePatch implements ImageAnalyser<FImage>{
	
	// stepX and stepY are the steps with which we will move the window when taking samples
	// binX and binY are the size of the window for the pixel patch
	private int stepX, stepY, binX, binY;
	
	//list of the currently analysed pixel patches
	private LocalFeatureList<FloatKeypoint> features;

	public DensePatch(int stepX, int stepY, int binX, int binY){
		this.stepX=stepX;
		this.stepY=stepY;
		this.binX=binX;
		this.binY=binY;
		features=new MemoryLocalFeatureList<FloatKeypoint>();
	}

	@Override
	public void analyseImage(FImage image) {
		//instead of clearing the list we create a new one in case the old one has been added to an external list
		features=new MemoryLocalFeatureList<FloatKeypoint>();
		
		//loop that traverses the image
		for(int x=0;x+binX<image.getWidth();x+=stepX){
			for(int y=0;y+binY<image.getHeight();y+=stepY){
				
				//float array to hold the particular pixel patch
				float[] feature= new float[binX*binY];
				int c=0;
				
				//loop to get the pixel values for the window
				for(int i=x;i<x+binX;i++){
					for(int j=y;j<y+binY;j++){
						feature[c++]=image.pixels[j][i];
					}
				}
				
				// calculate the mean
				final float mean = mean(feature);

				float m=0;
				
				// subtract mean and meanwhile calculate the magnitude
				for (int i = 0; i < feature.length; i++) {
					feature[i] -= mean;
					m+=feature[i]*feature[i];
				}
				
				//convert to unit-length by dividing by each term the square root of the magnitude
				m=(float) Math.sqrt(m);
				if(m!=0){
					for (int i = 0; i < feature.length; i++) {
						feature[i] = feature[i]/m;
					}
				}
				
				//add the pixel patch as a feature in the list
				features.add(new FloatKeypoint(x, y, 0, 1, feature));
			}
		}
	}
	
	//method to calculate the mean for our feature array
	private static float mean(float[] array) {
		float average = 0;
		
		for (int i = 0; i < array.length; i++) {
			average += array[i];				
		}
		
		average /= array.length;
		
		return average;
	}
	
	//method the pixel patches for the latest analysed image
	public LocalFeatureList<FloatKeypoint> getFloatKeypoints(){
		return features;
	}
}
