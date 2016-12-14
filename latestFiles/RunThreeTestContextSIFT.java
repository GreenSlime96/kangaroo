package uk.ac.soton.ecs.nb4g14.coursework3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.ByteFVComparison;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;

/**
 * Class RunThreeTestContextSift - The idea behind this implementation is that some of the information is lost
 * when performing K-means clustering on SIFT features from all groups and that is where the SIFT feature came from.
 * In this implementation 15 different quantisations are made - one for each group, and then each test image against all clusterings.
 * The annotation that is chosen is the one for which there are the most close matches between a cluster
 * in the group and SIFT feature from the image.
 * 
 * The trainQuantisation and logic for calculating the average accuracy had to be rewritten as they are more specific in this class
 * 
 * @author nb4g14 and kbp2g14
 */
public class RunThreeTestContextSIFT {
	public static void main(String[] args) throws FileSystemException {
		
		// ----------------------------------- Load the data -----------------------------------------
		
		final String directory = System.getProperty("user.name").equals("khengboonpek")
				? "/Users/khengboonpek/Downloads/" : "/home/nb4g14/University/ComputerVision/kangaroo/";
		
		final VFSGroupDataset<FImage> allData = 
				new VFSGroupDataset<FImage>(directory + "training", ImageUtilities.FIMAGE_READER);
		
		int training	=	75;
		int testing		=	100-training;
		
		final GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(allData, training, 0, testing);
		
		// ------------------------------------   Execution   -----------------------------------------
		
		DenseSIFT dsift = new DenseSIFT(5, 7);

		//the maximum distance a SIFT feature can be to a visual word before they are considered not matching
		float maximumDistanceToCluster=200f;
		
		//the minimum energy a SIFT feature must have to be considered
		float minimumEnergyForSIFT=0.05f;
		
		//the maximum features taken from each image for clustering
		int featuresPerImage=200;
		
		//number of visual words for each group
		int clusters=2000;
		
		//Estimate the clusters in all groups
		Map<String,ByteCentroidsResult> results=new HashMap<String,ByteCentroidsResult>();
		for(Entry<String, ListDataset<FImage>> entry:splits.getTrainingDataset().entrySet()){
			results.put(entry.getKey(), trainQuantiser(entry.getValue(), dsift, minimumEnergyForSIFT, featuresPerImage, clusters));
		}

		// will hold the accuracy for each group
		Map<String, Integer> accuracies=new HashMap<String,Integer>();
		
		//loop through all groups
		for(Entry<String, ListDataset<FImage>> entry:splits.getTestDataset().entrySet()){
			
			int correctPredictions=0;
			
			//loop through all images in the group
			for(FImage img:entry.getValue()){
				
				//analyse the image to get all SIFT features from it
				dsift.analyseImage(img);
				
				String assignedScene="";
				int maxFeatures=0;
				
				//for each class of data
				for(Entry<String, ByteCentroidsResult> entry2:results.entrySet()){
					
					//SIFT features close enough to a cluster
					int foundFeatures=0;

					// go through all keypoints are close enough to a visual word
					for(ByteDSIFTKeypoint point:dsift.getByteKeypoints(minimumEnergyForSIFT)){
						for(byte[] centroid:entry2.getValue().getCentroids()){
							
							//check if the visual word is close enough
							if(point.getFeatureVector().compare(new ByteFV(centroid), ByteFVComparison.EUCLIDEAN)<maximumDistanceToCluster){
								foundFeatures++;
								break;
							}
						}
					}
					
					//save the class label for the class with most SIFT points found
					if(foundFeatures>maxFeatures){
						maxFeatures=foundFeatures;
						assignedScene=entry2.getKey();
					}
				}
				
				//check if we had a correct prediction
				if(assignedScene.equals(entry.getKey())){
					correctPredictions++;
				}
			}
			
			//the number of correct predictions for a group are put in the map
			accuracies.put(entry.getKey(), correctPredictions);
		}
		
		//Display the overall information to the user
		
		float overallAccuracy=0;
		for(Entry<String, Integer> accuracy:accuracies.entrySet()){
			System.out.println(accuracy.getKey()+" "+accuracy.getValue()+" "+((float)accuracy.getValue()/testing));
			overallAccuracy+=accuracy.getValue();
		}
		
		System.out.println("Accuracy:"+(overallAccuracy/(allData.getGroups().size()*testing)));
		System.out.println("");
		
		
	}
	
	/*
	 * Method returning the result from performing K-means clustering on single group
	 */
	static ByteCentroidsResult trainQuantiser(Dataset<FImage> sample, DenseSIFT dsift,
			float energy, int featuresPerImage ,int clusters) {
		
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		for (FImage image:sample) {
			//analyse the image and get all SIFT points above certain energy
			dsift.analyseImage(image);
			LocalFeatureList<ByteDSIFTKeypoint> keypoints=dsift.getByteKeypoints(energy);
			
			//get a random sublist
			keypoints=(LocalFeatureList<ByteDSIFTKeypoint>) keypoints.randomSubList(Math.min(featuresPerImage,keypoints.size()));
			
			//add the sublist to all features for clustering
			allkeys.add(keypoints);
		}

		//perform K-means clustering
		
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(clusters);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		
		ByteCentroidsResult result = km.cluster(datasource);
		return result;
	}

}
