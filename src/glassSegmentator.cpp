#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "edges_pose_refiner/glassSegmentator.hpp"
#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

//#define VISUALIZE

//#define VISUALIZE_TABLE


void showGrabCutResults(const Mat &bgrImage, const Mat &mask, const string &title = "grabCut");


void createMasksForGrabCut(const cv::Mat &objectMask,
                           std::vector<cv::Rect> &allRois, std::vector<cv::Mat> &allRoiMasks,
                           const GlassSegmentatorParams &params)
{
  Mat objectMaskEroded;
  erode(objectMask, objectMaskEroded, Mat(), Point(-1, -1), params.grabCutErosionsIterations);
  Mat objectMaskDilated;
  dilate(objectMask, objectMaskDilated, Mat(), Point(-1, -1), params.grabCutDilationsIterations);

  Mat tmpObjectMask = objectMask.clone();
  vector<vector<Point> > contours;
  findContours(tmpObjectMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  allRois.clear();
  allRoiMasks.clear();
  for(size_t i = 0; i < contours.size(); ++i)
  {
    //TODO: move up
    const float minArea = 40.0f;
    if (contourArea(contours[i]) < minArea)
    {
      continue;
    }

    Rect roi = boundingRect(Mat(contours[i]));
    roi.x = std::max(0, roi.x - params.grabCutMargin);
    roi.y = std::max(0, roi.y - params.grabCutMargin);
    roi.width = std::min(objectMask.cols - roi.x, roi.width + 2*params.grabCutMargin);
    roi.height = std::min(objectMask.rows - roi.y, roi.height + 2*params.grabCutMargin);

    Mat currentMask(objectMask.size(), CV_8UC1, GC_BGD);
    currentMask(roi).setTo(GC_PR_BGD, objectMaskDilated(roi));
    currentMask(roi).setTo(GC_PR_FGD, objectMask(roi));
    currentMask(roi).setTo(GC_FGD, objectMaskEroded(roi));

    Mat roiMask = currentMask(roi);

    allRois.push_back(roi);
    allRoiMasks.push_back(roiMask);
  }
  CV_Assert(allRois.size() == allRoiMasks.size());
}

void refineSegmentationByGrabCut(const Mat &bgrImage, const Mat &rawMask, Mat &refinedMask, const GlassSegmentatorParams &params)
{
  CV_Assert(!rawMask.empty());
#ifdef VISUALIZE
  imshow("before grabcut", rawMask);
#endif

  refinedMask = Mat(rawMask.size(), CV_8UC1, Scalar(GC_BGD));

  vector<Rect> allRois;
  vector<Mat> allRoiMasks;
  createMasksForGrabCut(rawMask, allRois, allRoiMasks, params);

  for(size_t i = 0; i < allRois.size(); ++i)
  {
    Rect roi = allRois[i];
    Mat roiMask = allRoiMasks[i];
#ifdef VISUALIZE
    showGrabCutResults(bgrImage(roi), roiMask, "initial mask for GrabCut segmentation");
#endif

    Mat bgdModel, fgdModel;
    grabCut(bgrImage(roi), roiMask, Rect(), bgdModel, fgdModel, params.grabCutIterations, GC_INIT_WITH_MASK);

    Mat refinedMaskRoi = refinedMask(roi);
    roiMask.copyTo(refinedMaskRoi, (roiMask != GC_BGD) & (roiMask != GC_PR_BGD));

#ifdef VISUALIZE
    showGrabCutResults(bgrImage(roi), roiMask, "grab cut segmentation");
    waitKey();
#endif
  }

  Mat prFgd = (refinedMask == GC_PR_FGD);
  Mat fgd = (refinedMask == GC_FGD);
  refinedMask = prFgd | fgd;
#ifdef VISUALIZE
  showSegmentation(bgrImage, refinedMask, "final GrabCut segmentation");
  imshow("final GrabCut mask", refinedMask);
#endif
}

void snakeImage(const Mat &image, vector<Point> &points)
{
  float alpha = 10.0f;
  float beta = 30.0f;
//  float gamma = 10000.0f;
  float gamma = 10.0f;
  const CvSize searchWindow = cvSize(15, 15);

/*
  vector<CvPoint> cvPoints(points.size());
  for(size_t i = 0; i < points.size(); ++i)
  {
    cvPoints[i] = points[i];
  }
*/

  vector<CvPoint> cvPoints;
  for(size_t i = 0; i < points.size(); ++i)
  {
    if(i % 2 == 0)
      cvPoints.push_back(points[i]);
  }


  Mat grayImage;
  if(image.channels() == 3)
  {
    cvtColor(image, grayImage, CV_BGR2GRAY);
  }
  else
  {
    grayImage = image;
  }

  IplImage imageForSnake = grayImage;
  cvSnakeImage(&imageForSnake, cvPoints.data(), cvPoints.size(), &alpha, &beta, &gamma, CV_VALUE, searchWindow, cvTermCriteria(CV_TERMCRIT_ITER, 1, 0.0), 1);


  for (size_t i = 0; i < points.size(); ++i)
  {
    points[i] = cvPoints[i / 2];
  }
}

void refineSegmentationBySnake(const Mat &bgrImage, const Mat &rawMask, Mat &refinedMask)
{
  refinedMask = Mat(rawMask.size(), CV_8UC1, Scalar(0));

  Mat tmpRawMask = rawMask.clone();
  vector<vector<Point> > contours;
  findContours(tmpRawMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  for(size_t i = 0; i < contours.size(); ++i)
  {
    for(int j=0; j<10000; ++j)
    {
      Mat drawImage = bgrImage.clone();
      drawContours(drawImage, contours, i, Scalar(0, 255, 0), 1);
      imshow("snake", drawImage);
      waitKey();

      snakeImage(bgrImage, contours[i]);
    }
  }
}


void showGrabCutResults(const Mat &bgrImage, const Mat &mask, const string &title)
{
  Mat result(mask.size(), CV_8UC3, Scalar::all(0));
  result.setTo(Scalar(255, 0, 0), mask == GC_BGD);
  result.setTo(Scalar(128, 0, 0), mask == GC_PR_BGD);
  result.setTo(Scalar(0, 0, 255), mask == GC_FGD);
  result.setTo(Scalar(0, 0, 128), mask == GC_PR_FGD);

  imshow(title, result);

  //TODO: move up
  Mat mergedMask = 0.7 * bgrImage + 0.3 * result;
  imshow(title + " merged", mergedMask);
}

void showSegmentation(const Mat &image, const Mat &mask, const string &title)
{
  Mat drawImage = drawSegmentation(image, mask);
  imshow(title, drawImage);
}

GlassSegmentator::GlassSegmentator(const GlassSegmentatorParams &_params)
{
  params = _params;
}

void refineGlassMaskByTableHull(const std::vector<cv::Point2f> &tableHull, cv::Mat &glassMask)
{
#ifdef VISUALIZE_TABLE
  Mat visualizedGlassMask;
  cvtColor(glassMask, visualizedGlassMask, CV_GRAY2BGR);
  for (size_t i = 0; i < projectedHull.size(); ++i)
  {
    circle(visualizedGlassMask, projectedHull[i], 2, Scalar(0, 0, 255), -1);
    line(visualizedGlassMask, projectedHull[i], projectedHull[(i + 1) % projectedHull.size()], Scalar(255, 0, 0));
  }
  imshow("table hull", visualizedGlassMask);
#endif

  vector<vector<Point> > contours;
  Mat copyGlassMask = glassMask.clone();
  findContours(copyGlassMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (size_t i = 0; i < contours.size(); ++i)
  {
    Moments moms = moments(contours[i]);
    Point2f centroid(moms.m10 / moms.m00, moms.m01 / moms.m00);
    if (pointPolygonTest(tableHull, centroid, false) < 0)
    {
      drawContours(glassMask, contours, i, Scalar(0, 0, 0), -1);
    }
  }
}

void GlassSegmentator::segment(const cv::Mat &bgrImage, const cv::Mat &depthMat, const cv::Mat &registrationMask, int &numberOfComponents,
                               cv::Mat &glassMask, const std::vector<cv::Point2f> *tableHull)
{
  Mat srcMask = getInvalidDepthMask(depthMat, registrationMask);
#ifdef VISUALIZE
  imshow("mask without registration errors", srcMask);
#endif

  if (tableHull != 0)
  {
    refineGlassMaskByTableHull(*tableHull, srcMask);
  }

#ifdef VISUALIZE
  imshow("mask with table", srcMask);
#endif

  Mat mask = srcMask.clone();
  morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), params.closingIterations);
#ifdef VISUALIZE
  imshow("mask after closing", mask);
#endif

  morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), params.openingIterations);
#ifdef VISUALIZE
  imshow("mask after openning", mask);
#endif
  vector<vector<Point> > contours;
  findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  numberOfComponents = static_cast<int>(contours.size());

  Mat glassImage(mask.size(), CV_8UC1, Scalar(0));
  drawContours(glassImage, contours, -1, Scalar(255), -1);

  int elementSize = params.finalClosingIterations * 2 + 1;
  Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(elementSize, elementSize), Point(params.finalClosingIterations, params.finalClosingIterations));
  morphologyEx(srcMask, srcMask, MORPH_CLOSE, structuringElement, Point(params.finalClosingIterations, params.finalClosingIterations));

//  morphologyEx(srcMask, srcMask, MORPH_CLOSE, Mat(), Point(-1, -1), params.finalClosingIterations);
#ifdef VISUALIZE
  imshow("final closing", srcMask);
#endif
  uchar foundComponentColor = 128;

  for(int i = 0; i < glassImage.rows; ++i)
  {
    for(int j = 0; j < glassImage.cols; ++j)
    {
      if(glassImage.at<uchar>(i, j) == 255 && srcMask.at<uchar>(i, j) == 255)
      {
        floodFill(srcMask, Point(j, i), Scalar(foundComponentColor));
      }
    }
  }

  glassMask = (srcMask == foundComponentColor);

#ifdef VISUALIZE
  imshow("before convex", glassMask);
#endif

  if (params.fillConvex)
  {

    Mat tmpGlassMask = glassMask.clone();
    vector<vector<Point> > srcContours;
    findContours(tmpGlassMask, srcContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < srcContours.size(); ++i)
    {
      vector<Point> hull;
      //convexHull(Mat(srcContours[i]), hull, false, true);
      convexHull(Mat(srcContours[i]), hull);
      fillConvexPoly(glassMask, hull.data(), hull.size(), Scalar(255));
    }
  }


  if (params.useGrabCut)
  {
    Mat refinedGlassMask;
    refineSegmentationByGrabCut(bgrImage, glassMask, refinedGlassMask, params);
//    Mat prFgd = (refinedGlassMask == GC_PR_FGD);
//    Mat fgd = (refinedGlassMask == GC_FGD);
//    glassMask = prFgd | fgd;
    glassMask = refinedGlassMask;
  }

#ifdef VISUALIZE
  showSegmentation(bgrImage, glassMask, "grabCut segmentation");
  waitKey();
#endif
}

bool GlassSegmentator::polynomialfit(int obs, int degree, 
		   double *dx, double *dy, double *store) /* n, p */
{
  gsl_multifit_linear_workspace *ws;
  gsl_matrix *cov, *X;
  gsl_vector *y, *c;
  double chisq;
 
  int i, j;
 
  X = gsl_matrix_alloc(obs, degree);
  y = gsl_vector_alloc(obs);
  c = gsl_vector_alloc(degree);
  cov = gsl_matrix_alloc(degree, degree);
 
  for(i=0; i < obs; i++) {
    gsl_matrix_set(X, i, 0, 1.0);
    for(j=0; j < degree; j++) {
      gsl_matrix_set(X, i, j, pow(dx[i], j));
    }
    gsl_vector_set(y, i, dy[i]);
  }
 
  ws = gsl_multifit_linear_alloc(obs, degree);
  gsl_multifit_linear(X, y, c, cov, &chisq, ws);
 
  /* store result ... */
  for(i=0; i < degree; i++)
  {
    store[i] = gsl_vector_get(c, i);
  }
 
  gsl_multifit_linear_free(ws);
  gsl_matrix_free(X);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(c);
  return true; /* we do not "analyse" the result (cov matrix mainly)
		  to know if the fit is "good" */
}

float GlassSegmentator::standard_deviation(float data[], int n)
{
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {   
        mean+=data[i];
    }   
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);    
}

void GlassSegmentator::AroundForeground(cv::Mat &src, cv::Mat &dst, long foreground_pixel)
{
    int morph_size = 60; 
    int around_pixel,pixel_dif;
    cv::Mat I2; 

    while(1)
    {   
        around_pixel = 0;
        cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*morph_size+1, 2*morph_size+1), cv::Point(morph_size, morph_size));
        cv::dilate(src, I2, element); 
        dst = I2 - src;
        for(int i=0; i<dst.rows; i++)
        {   
            for(int j=0; j<dst.cols; j++)
            {   
                if(dst.at<uchar>(i,j) == 255)
                around_pixel++;
            }
        }
        pixel_dif = around_pixel - foreground_pixel;
        if (abs(pixel_dif)<5000)
        break;
        if (pixel_dif<0)
        morph_size+=10;
        else
        morph_size-=10;
    }
}


double GlassSegmentator::transparentScore(const cv::Mat &bgrImage, cv::Mat &glassMask)
{
	cv::Mat img = bgrImage;
    //glassMask might be color image, need to be cvtColor
	cv::Mat Gray_bmp = glassMask;
	int x = img.rows;
	int y = img.cols;
	int z = img.channels();

	cv::Mat Gray_img;
	cv::cvtColor(img, Gray_img, CV_BGR2GRAY);

	cv::GaussianBlur(Gray_img, Gray_img, cv::Size(5,5), 1.5);

	cv::Mat HSV;
	cvtColor(img, HSV, CV_BGR2HSV);

	cv::threshold(Gray_bmp, Gray_bmp, 1.0, 255.0, cv::THRESH_BINARY_INV);

	long G_count = 0;

	for(int i=0; i<x; i++)
	{
		for(int j=0; j<y; j++)
		{
			if(Gray_bmp.at<uchar>(i,j) == 255)
			G_count++;
		}
	}

////////////////////////////////// Color Similarity ///////////////////////////////
	
	long V_count = 0;
	cv::Mat V(Gray_bmp.size(), CV_8UC1, cv::Scalar(0));
	AroundForeground(Gray_bmp, V, G_count);

	int gap = 5;
	int b_index, f_index;
	int his_size = 36;
	long *background_his_hue = new long[his_size];
	long *foreground_his_hue = new long[his_size];
	for(int i=0; i<his_size; i++)
	{
		background_his_hue[i] = 0;
		foreground_his_hue[i] = 0;
	}
	

	for(int i=0; i<x; i++)
	{
		for(int j=0; j<y; j++)
		{
			if(V.at<uchar>(i,j) == 255)
			{
				V_count++;
				b_index = (int)HSV.at<cv::Vec3b>(i,j)[0];
				background_his_hue[b_index/gap] = background_his_hue[b_index/gap] + 1;
			}
			if(Gray_bmp.at<uchar>(i,j) == 255)
			{
				f_index = (int)HSV.at<cv::Vec3b>(i,j)[0];
				foreground_his_hue[f_index/gap] = foreground_his_hue[f_index/gap] + 1;
			}
		}
	}

	double *BH = new double[his_size];
	double *FH = new double[his_size];

	for(int i=0;i<his_size;i++)
	{
		//cout<<"background_his("<<i+1<<") : "<<background_his_hue[i]<<endl;
		//cout<<"foreground_his("<<i+1<<") : "<<foreground_his_hue[i]<<endl;
		BH[i] = ((double)background_his_hue[i])/((double)V_count);
		FH[i] = ((double)foreground_his_hue[i])/((double)G_count);
		//cout<<"BH("<<i+1<<") : "<<BH[i]<<endl;
		//cout<<"FH("<<i+1<<") : "<<FH[i]<<endl;
	}

	double Sum = 0;
    double distance,CS_Result;

    for(int i=0;i<his_size;i++)
    {
        Sum = Sum + pow((BH[i]-FH[i]),2.0);
        distance = sqrt(Sum);
    }
	
	CS_Result = 1.0 - distance;

///////////////////////////////////// Local Standard Deviation ///////////////////////////////

	cv::Mat BW;
	cv::Canny(Gray_bmp, BW, 125, 255);

	int mask;
	int range = 20;
	double mask_occupy = 0.001;

  	mask = round(sqrt(G_count*mask_occupy));
	
	if (mask%2==0)
	mask++;
	if (mask<3)
	mask = 3;

	float *bound_Gray = new float[mask*mask];
	int inpadd = (mask+1)/2-1;
	int n = 0;
	int index = 0;
	float std_sum = 0;
	float std,SD_Result;

	for(int i=0; i<x; i++)
	{
		for(int j=0; j<y; j++)
		{
			if ((BW.at<uchar>(i,j)==255)&&(i>mask)&&(j>mask)&&(i+mask<x)&&(j+mask<y))
			{
				n++;
				for(int k=0; k<mask; k++)
				{
					for(int l=0; l<mask; l++)
					{
						bound_Gray[index] = Gray_img.at<uchar>(i-inpadd+k,j-inpadd+l);
						index++;
					}
				}
				index = 0;
				std_sum += standard_deviation(bound_Gray, mask*mask);
			}
		}
	}
	
	std = std_sum/n;
	
	if (std > range)
	std = range;
	
	SD_Result = (range - std)/range;

/////////////////////////////////// Highlight ///////////////////////////////////////

	int T_range = 255;
	long *perimeter = new long[T_range];
	cv::Mat After_threshold(x,y,CV_8U,cv::Scalar(0));
	cv::Mat cut;

	for(int k=0; k<T_range; k++)
	{
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
			}
		}
	}

	int NP, Threshold_value;
	double e, m, fit_perimeter;
	int HL_Result = 0;
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

		polynomialfit(NP, degree, p, q, coeff);
		//cout<<"Polyfit("<<i<<") : "<<coeff[1]<<"  "<<coeff[0]<<endl;

		for(int j=T_range-1; j>=i; j--)
		{
			fit_perimeter = coeff[1]*j + coeff[0];
			e = abs(fit_perimeter - perimeter[j]);
			fit_error[i] += e;
		}
		
		//cout<<"Errorfit("<<i<<") : "<< fit_error[i]<<endl;
		delete[] p;
		delete[] q;
	}


	for(int k=T_range-2; k>0; k--)
	{
		m = abs(fit_error[k] - fit_error[k-1]);
		if(m > 200)
		{
			Threshold_value = k;
			break;
		}
	}
	//cout<<"Threshold value = "<<Threshold_value<<endl;

	for(int i=0; i<x; i++)
	{
		for(int j=0; j<y; j++)
		{
			if(Gray_img.at<uchar>(i,j)>Threshold_value)
			{
				Highlight.at<uchar>(i,j) = 255;

				if(Gray_bmp.at<uchar>(i,j)==255)
				HL_Result = 1;
			}
			else
			{
				Highlight.at<uchar>(i,j) = 0;
			}
		}
	}

	//cv::imshow("Highlight Result",Highlight);

	double Result = (CS_Result + SD_Result + HL_Result)/3;

	img.release();
	Gray_bmp.release();
	HSV.release();
	V.release();
	BW.release();
	After_threshold.release();
	cut.release();
	Highlight.release();

	return Result;
}

void GlassSegmentator::rickySegment(const cv::Mat &bgrImage, const cv::Mat &HImage, const cv::Mat &depthMat, cv::Mat &glassMask)
{
	/* Extract NaN area */
	int row = depthMat.rows;
	int col = depthMat.cols;
	
	Mat ucharMat = Mat::zeros(row, col, CV_8U);
	float f;

	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			f = depthMat.at<float>(i,j);
			if(f!=f)  //For float, f != f will be true only if f is NaN
				ucharMat.at<uchar>(i,j) = 255;
			else
				ucharMat.at<uchar>(i,j) = 0;
		}
	}
#ifdef PREVISUALIZE	
	imshow("Mat before morph", ucharMat);
#endif

	/*Opening & Closing for removing noise*/
	Mat mat = Mat::zeros(row, col, CV_8U);
	morphologyEx(ucharMat,mat,cv::MORPH_OPEN, Mat(), Point(-1, -1), 10);
	morphologyEx(ucharMat,mat,cv::MORPH_CLOSE, Mat(), Point(-1, -1), 6);


	/* Fill the outer part to black (for /depth_registered/image_rect)*/
	Mat fillMat = Mat::zeros(row, col, CV_8U);
	cv::rectangle(fillMat, Point( 30, 30), Point(590, 430), Scalar(255, 255, 255), -1, 8);
	cv::bitwise_and(fillMat, mat, mat);

#if 1	
	imshow("Mat after morph", mat);
	imshow("Mat after fill", mat);
	waitKey();
#endif
	/* Find Connected Componets */
	vector< vector<cv::Point> > contours;
	Mat conMat = Mat::zeros(row, col, CV_8U);
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			conMat.at<uchar>(i,j) = mat.at<uchar>(i,j);
		}
	}

	cv::findContours(conMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//Remove contours that are too small(considered as noise)
	int cmin=50;
	std::vector< std::vector<cv::Point> >::iterator it=contours.begin();
	while(it!=contours.end())
	{
		if(it->size()<cmin)
			it=contours.erase(it);
		else
			++it;
	}

	cv::Mat connectedMat(ucharMat.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(connectedMat, contours, -1, cv::Scalar(0), 1);

#ifdef PREVISUALIZE	
	imshow("Contour of connected component", connectedMat);
#endif

	/* AND(Highlight image, Image after MORPH) */
	Mat highlightMat = HImage;
	Mat andMat=Mat::zeros(row, col, CV_8U);
	cv::bitwise_and(mat, highlightMat, andMat);

#ifdef PREVISUALIZE	
	imshow("Highlight", highlightMat);
	imshow("And of Highlight and Image after MORPH", andMat);
#endif

	/*Extract the region containing highlights*/
	//Labeling the contour of connected component
	cv::Mat labelMat(ucharMat.size(), CV_8U, cv::Scalar(255));
	for(int i=0; i<contours.size(); i++)
		cv::drawContours(labelMat, contours, i, cv::Scalar(i), CV_FILLED);

#ifdef PREVISUALIZE	
	imshow("Labeled Mat", labelMat);
#endif

	//Check which region contains highlight
	std::vector<int>  check;
	for(int i=0; i<contours.size(); i++)
		check.push_back(0);
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			if(andMat.at<uchar>(i,j) > 0)
			   check[labelMat.at<uchar>(i,j)]++;	
		}
	}

	//Draw the region contains the highlight
	vector< vector<cv::Point> > candidateContours;	
	for(int i=0; i<contours.size(); i++)
	{
		if(check[i]>0)
			candidateContours.push_back(contours[i]);
	}
	cv::Mat candidateMat(ucharMat.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(candidateMat, candidateContours, -1, cv::Scalar(0), CV_FILLED);
#ifdef PREVISUALIZE	
	imshow("Candidate Mat", candidateMat);
	std::cout << "Candidate Contour Number: " <<candidateContours.size() << std::endl;
#endif

	/*Feed the region containing highlights to Grabcut to segment contour of transparent objects*/
	Mat rgbMat = bgrImage; 
	
	//Assign Foreground model & Background model
    cv::Mat bgModel, fgModel;

    //Initialize mask
    cv::Mat1b result(row, col);
	cv::Rect rectangle(0, 0, col, row);
	
	vector<cv::Mat> transparentObjectCandidates;
    cv::Mat foreground(rgbMat.size(), CV_8UC3, cv::Scalar(255,255,255));
	
	for(int i=0; i<candidateContours.size(); i++)
	{
    	result.setTo(cv::GC_BGD);
		foreground.setTo(cv::Scalar(255,255,255));
			
		//Set Probable Foreground Mask by finding the bounding rect
		cv::Rect prRect = cv::boundingRect(cv::Mat(candidateContours[i]));
		cv::rectangle(result, prRect, cv::Scalar(cv::GC_PR_FGD), CV_FILLED);
	
		//Set Foreground Mask
		cv::Mat fgMat(ucharMat.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(result, candidateContours, i, cv::Scalar(cv::GC_FGD), CV_FILLED);
	
		//Grabcut
		cv::grabCut(rgbMat, result, rectangle, fgModel, bgModel, 2, cv::GC_INIT_WITH_MASK);

    	//Extract Foreground Image
    	result = result & 1;
    	rgbMat.copyTo(foreground, result);
	
		cv::Mat transObject = foreground.clone();	

		transparentObjectCandidates.push_back(transObject);

#ifdef PREVISUALIZE
		cv::rectangle(fgMat, prRect, cv::Scalar(50), CV_FILLED);
		cv::drawContours(fgMat, candidateContours, i, cv::Scalar(0), CV_FILLED);
	    cv::imshow("Foreground Mask", fgMat);  
		waitKey();
#endif
	}
	
#ifdef VISUALIZE
	cv::imshow("Color Image", rgbMat);	
	for(int i=0; i<transparentObjectCandidates.size(); i++)
	{
    	cv::imshow("Transparent Object", transparentObjectCandidates[i]);
		waitKey();
	}
#endif
	
	double score[transparentObjectCandidates.size()];

	//Generate black transparent object mask
	for(int i=0; i<transparentObjectCandidates.size(); i++)
	{
		cvtColor(transparentObjectCandidates[i], transparentObjectCandidates[i], CV_RGB2GRAY);	
		threshold(transparentObjectCandidates[i], transparentObjectCandidates[i], 254, 255, 0); //white mask
		score[i] = transparentScore(bgrImage, transparentObjectCandidates[i]);	
	}
	
	cv::Mat andCandidate(ucharMat.size(), CV_8U, cv::Scalar(0));
	for(int i=0; i<transparentObjectCandidates.size(); i++)
	{
		//Turn black mask to white mask
		//cv::bitwise_not(transparentObjectCandidates[i], transparentObjectCandidates[i]);
		if(score[i]>0.6)
			cv::bitwise_or(andCandidate, transparentObjectCandidates[i], andCandidate);
	}
	
	/* glassMask is the mask with highest score */
	glassMask = andCandidate;
}


void segmentGlassManually(const cv::Mat &image, cv::Mat &glassMask)
{
  vector<vector<Point> > contours(1);
  markContourByUser(image, contours[0], "manual glass segmentation");

  glassMask = Mat(image.size(), CV_8UC1, Scalar(0));
  drawContours(glassMask, contours, -1, Scalar::all(255), -1);
}
