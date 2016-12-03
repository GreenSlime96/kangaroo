package uk.ac.soton.ecs.nb4g14.coursework;

import java.util.HashMap;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class RunOne_2 {
	
	/* for choosing K */
	/*
	public static void main(String[] args) throws FileSystemException {
		//final VFSGroupDataset<FImage> allData = 
		//		new VFSGroupDataset<FImage>("/Users/khengboonpek/Downloads/training", ImageUtilities.FIMAGE_READER);
		//
		//final VFSListDataset<FImage> query = 
		//		new VFSListDataset<FImage>("/Users/khengboonpek/Downloads/testing", ImageUtilities.FIMAGE_READER);
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>("/home/nb4g14/University/ComputerVision/kangaroo/training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>("/home/nb4g14/University/ComputerVision/kangaroo/testing", ImageUtilities.FIMAGE_READER);
		
		
		List<IntFloatPair> pairs=new ArrayList<IntFloatPair>();
		
		for(int i=0;i<50;i++){
			pairs.add(new IntFloatPair(i, 0));
		}
		
		for(int i=0;i<5;i++){
			
			final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, 75, 0, 25);
			final GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			final GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			FeatureExtractor<FloatFV, FImage> featureExtractor=new TinyImageFeatureExtractor();

			KNNAnnotator<FImage, String, FloatFV> annotator=new KNNAnnotator<FImage, String, FloatFV>(featureExtractor, FloatFVComparison.EUCLIDEAN);
			
			annotator.train(training);
			System.out.println(i);
			for(int K=0;K<50;K++){
				System.out.print((K+1)+" ");
				annotator.setK(K+1);
				
				ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
						new ClassificationEvaluator<CMResult<String>, String, FImage>(
							annotator, testing, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
				
				//perform the evaluation
				Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
				CMResult<String> result = eval.analyse(guesses);
				
				pairs.get(K).second+=result.getMatrix().getAccuracy();
				
				//print the report
				//System.out.println(result.getDetailReport());
			}
			System.out.println("");
		}
		
		pairs.sort(new Comparator<IntFloatPair>(){

			@Override
			public int compare(IntFloatPair o1, IntFloatPair o2) {
				float c=o2.second-o1.second;
				if(c>0){
					c=1;
				} else {
					c=-1;
				}
				return (int) c;
			}});
		
		for(int i=0;i<50;i++){
			System.out.println((pairs.get(i).first+1)+" "+(pairs.get(i).second/5));
		}
	}
	*/
	
	public static void main(String[] argv) throws FileSystemException{
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>("/home/nb4g14/University/ComputerVision/kangaroo/training", ImageUtilities.FIMAGE_READER);
		
		final VFSListDataset<FImage> query = 
				new VFSListDataset<FImage>("/home/nb4g14/University/ComputerVision/kangaroo/testing", ImageUtilities.FIMAGE_READER);
		
		FeatureExtractor<FloatFV, FImage> featureExtractor=new TinyImageFeatureExtractor();

		KNNAnnotator<FImage, String, FloatFV> annotator=new KNNAnnotator<FImage, String, FloatFV>(featureExtractor, FloatFVComparison.EUCLIDEAN);
		
		annotator.setK(15);
		annotator.train(allData);
		
		HashMap<String,String> map=new HashMap<String,String>();
		for(int i=0;i<query.size();i++){
			FImage image=query.get(i);
			String name=query.getFileObject(i).getName().getBaseName();
			float conf=-1;
			String annotation="";
			for(ScoredAnnotation<String> ann:annotator.annotate(image)){
				if(ann.confidence>conf){
					annotation=ann.annotation;
					conf=ann.confidence;
				}
			}
			map.put(name, annotation);
		}
		
		for(int i=0;i<query.size();i++){
			System.out.println(i+".jpg "+map.get(i+".jpg"));
		}
	}

}
