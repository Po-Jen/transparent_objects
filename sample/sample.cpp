#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using namespace transpod;

void readData(const string &pathToDemoData, PinholeCamera &camera,
              Mat &objectPointCloud_1, Mat &objectNormals_1, Mat &objectPointCloud_2, Mat &objectNormals_2,
              Mat &registrationMask, Mat &image, Mat &depth);

cv::Mat getHighlightImage(const cv::Mat &bgrImage);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << argv[0] << " <path_to_sample_data>" << std::endl;
    return -1;
  }

  string pathToDemoData = argv[1];
  const string objectName_1 = "glass";
  const string objectName_2 = "middle_cup";

  // 1. Get the data
  std::cout << "Reading data...  " << std::flush;
  PinholeCamera camera;
  Mat objectPointCloud_1, objectNormals_1, objectPointCloud_2, objectNormals_2, registrationMask, image, depth;
  readData(pathToDemoData, camera, objectPointCloud_1, objectNormals_1, objectPointCloud_2, objectNormals_2, registrationMask, image, depth);
  Mat highlightImg = getHighlightImage(image);
  std::cout << "done." << std::endl;

  /*
  std::cout<<"#rows:"<< objectPointCloud_1.rows<<", #cols:"<<objectPointCloud_1.cols<<std::endl;
  std::cout<<"#rows:"<< objectPointCloud_2.rows<<", #cols:"<<objectPointCloud_2.cols<<std::endl;
  std::cout<<"#rows:"<< objectNormals_1.rows<<", #cols:"<<objectNormals_1.cols<<std::endl;
  std::cout<<"#rows:"<< objectNormals_2.rows<<", #cols:"<<objectNormals_2.cols<<std::endl;
  */
  //imshow("pcd 1",objectPointCloud_1);
  //waitKey();
  
  // 2. Initialize the detector
  std::cout << "Training...  " << std::flush;
  //    A. set morphology parameters of glass segmentation
  DetectorParams params;
  params.glassSegmentationParams.closingIterations = 6;
  params.glassSegmentationParams.openingIterations = 10;
  //    B. add train objects into the detector
  Detector detector(camera, params);
  //detector.addTrainObject(objectName_2, objectPointCloud_2, objectNormals_2);
  //std::cout << "obj 2 done." << std::endl;
  detector.addTrainObject(objectName_1, objectPointCloud_1, objectNormals_1);
  std::cout << "done." << std::endl;

  // 3. Detect transparent objects
  std::cout << "Detecting...  " << std::flush;
  vector<PoseRT> poses;
  vector<float> errors;
  vector<string> detectedObjectsNames;
  Detector::DebugInfo debugInfo;
  detector.detect(image, highlightImg, depth, registrationMask,
                  poses, errors, detectedObjectsNames, &debugInfo);
  std::cout << "\nposes rotation and trans in sample: " << poses[0].getProjectiveMatrix() << std::endl;
  std::cout << "done." << std::endl;

  // 4. Visualize results
  /*
  imshow("input rgb image", image);
  imshow("input depth image", depth);
  imshow("highlight image", highlightImg);
  */
  showSegmentation(image, debugInfo.glassMask);
  detector.showResults(poses, detectedObjectsNames, image);
  std::cout << "object detected: "<< detectedObjectsNames.size() << std::endl;
  //std::cout << detectedObjectsNames[0] << std::endl;
  waitKey();

  return 0;
}

void readData(const string &pathToDemoData, PinholeCamera &camera,
              Mat &objectPointCloud_1, Mat &objectNormals_1, Mat &objectPointCloud_2, Mat &objectNormals_2,
              Mat &registrationMask, Mat &image, Mat &depth)
{
  //const string objectPointCloudFilename_1 = pathToDemoData + "/trainObject_1.ply";
  const string objectPointCloudFilename_2 = pathToDemoData + "/trainObject_2.ply";
  //const string objectPointCloudFilename_1 = pathToDemoData + "/beaker_model_face_0.ply";
  //const string objectPointCloudFilename_1 = pathToDemoData + "/beaker_model_my.ply";
  //const string objectPointCloudFilename_1 = pathToDemoData + "/4.ply";
  const string objectPointCloudFilename_1 = pathToDemoData + "/trainObject_3_tube.ply";
  const string cameraFilename = pathToDemoData + "/camera.yml";
  const string registrationMaskFilename = pathToDemoData + "/registrationMask.png";
  //const string imageFilename = pathToDemoData + "/image.png";
  //const string depthFilename = pathToDemoData + "/depth.xml.gz";
  //const string imageFilename = pathToDemoData + "/image_test_6.png";
  //const string depthFilename = pathToDemoData + "/kinect_depth_6.xml.gz";
  const string imageFilename = pathToDemoData + "/image_test_4.png";
  const string depthFilename = pathToDemoData + "/kinect_depth_4.xml.gz";

  TODBaseImporter dataImporter;
  dataImporter.importCamera(cameraFilename, camera);
  dataImporter.importPointCloud(objectPointCloudFilename_1, objectPointCloud_1, objectNormals_1);
  dataImporter.importPointCloud(objectPointCloudFilename_2, objectPointCloud_2, objectNormals_2);
  dataImporter.importRegistrationMask(registrationMaskFilename, registrationMask);
  dataImporter.importBGRImage(imageFilename, image);
  dataImporter.importDepth(depthFilename, depth);

}

cv::Mat getHighlightImage(const cv::Mat &bgrImage)
{
	/////////////////////////////////// Highlight ///////////////////////////////////////
	
	cv::Mat Gray_img;
    cvtColor(bgrImage, Gray_img, CV_BGR2GRAY);
	
	int x = Gray_img.rows;
	int y = Gray_img.cols;

	int T_range = 255;
	long *perimeter = new long[T_range];
	cv::Mat After_threshold(x,y,CV_8U,cv::Scalar(0));
	cv::Mat cut;

	for(int k=0; k<T_range; k++)
	{
		perimeter[k]=0;
		
		for(int i=0; i<x; i++)
		{
			for(int j=0; j<y; j++)
			{
				if(Gray_img.at<uchar>(i,j)>k)
				{
					After_threshold.at<uchar>(i,j) = 255;
				}
				else
				{
					After_threshold.at<uchar>(i,j) = 0;
				}
			}
		}

		cv::Canny(After_threshold, cut, 125, 255);

		for(int i=0; i<x; i++)
		{
			for(int j=0; j<y; j++)
			{
				if(cut.at<uchar>(i,j)==255)
				perimeter[k] += 1;

				After_threshold.at<uchar>(i,j) = 0;
			}
		}
	}

	int NP, Threshold_value=255;
	double e, m, fit_perimeter;
	int degree = 2;
	double coeff[degree];
	cv::Mat Highlight(x,y,CV_8U,cv::Scalar(0));

	double *fit_error = new double[T_range-1];
	for(int i=0; i<T_range-1; i++)
	{
		fit_error[i] = 0;
	}

	for(int i=T_range-1; i>0; i--)
	{
		NP = T_range-i+1;
		double *p = new double[NP];
		double *q = new double[NP];

		for(int k=i; k<T_range; k++)
		{
			p[k-i] = k;
			q[k-i] = perimeter[k];
		}

		GlassSegmentator g;
		g.polynomialfit(NP, degree, p, q, coeff);

		for(int j=T_range-1; j>=i; j--)
		{
			fit_perimeter = coeff[1]*j + coeff[0];
			e = abs(fit_perimeter - perimeter[j]);
			fit_error[i] += e;
		}

		delete[] p;
		delete[] q;
	}


	for(int k=T_range-2; k>0; k--)
	{
		m = abs(fit_error[k] - fit_error[k-1]);
		if(m > 500)
		{
			Threshold_value = k;
			break;
		}
	}

	//std::cout << "Threshold Value: " << Threshold_value << std::endl;

	for(int i=0; i<x; i++)
	{
		for(int j=0; j<y; j++)
		{
			if(Gray_img.at<uchar>(i,j)>Threshold_value)
			{
				Highlight.at<uchar>(i,j) = 255;
			}
			else
			{
				Highlight.at<uchar>(i,j) = 0;
			}
		}
	}

	return Highlight;
}	
