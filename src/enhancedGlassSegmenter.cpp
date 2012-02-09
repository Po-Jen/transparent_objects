#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

struct VertexProperties
{
  cv::Point pt;
  float orientation;
  bool isRegion;
  int regionIndex;
};

struct EdgeProperties
{
  float length;
  float maximumAngle;

  EdgeProperties();
};

EdgeProperties::EdgeProperties()
{
  length = 0.0f;
  maximumAngle = 0.0f;
}

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, VertexProperties, EdgeProperties> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor VertexDescriptor;
typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;


//TODO: remove
void estimateAffinities(const Graph &graph, size_t regionCount, cv::Size imageSize, int regionIndex, std::vector<float> &affinities);

class Region
{
  public:
    Region(const cv::Mat &image, const cv::Mat &mask);

    cv::Point2f getCenter() const;
    const cv::Mat& getMask() const;
    const cv::Mat& getColorHistogram() const;
    const cv::Mat& getIntensityClusters() const;
  private:
    void computeColorHistogram();
    void clusterIntensities();
    void computeCenter();

    cv::Mat image, mask;
    cv::Mat grayscaleImage;

    cv::Mat hist;
    cv::Mat intensityClusterCenters;

    cv::Point2f center;
};

Region::Region(const cv::Mat &_image, const cv::Mat &_mask)
{
  image = _image;
  mask = _mask;

  computeColorHistogram();
  clusterIntensities();
  computeCenter();
}

cv::Point2f Region::getCenter() const
{
  return center;
}

const cv::Mat& Region::getColorHistogram() const
{
  return hist;
}

const cv::Mat& Region::getIntensityClusters() const
{
  return intensityClusterCenters;
}

const cv::Mat& Region::getMask() const
{
  return mask;
}

void Region::computeCenter()
{
  Point2d sum(0.0, 0.0);
  int pointCount = 0;
  for (int i = 0; i < mask.rows; ++i)
  {
    for (int j = 0; j < mask.cols; ++j)
    {
      if (mask.at<uchar>(i, j) != 0)
      {
        sum += Point2d(j, i);
        ++pointCount;
      }
    }
  }
  sum *= 1.0 / pointCount;
  center = sum;
}

void Region::computeColorHistogram()
{
  //TODO: move up
  const int hbins = 20;
  const int sbins = 20;

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(mask.type() == CV_8UC1);
  Mat hsv;
  cvtColor(image, hsv, CV_BGR2HSV);

  int histSize[] = {hbins, sbins};
  float hranges[] = {0, 180};
  float sranges[] = {0, 256};
  const float* ranges[] = {hranges, sranges};
  int channels[] = {0, 1};
  calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);
}

void Region::clusterIntensities()
{
  //TODO: move up
  const int clusterCount = 10;

  if (grayscaleImage.empty())
  {
    if (image.channels() == 3)
    {
      cvtColor(image, grayscaleImage, CV_BGR2GRAY);
    }
    else
    {
      grayscaleImage = image;
    }
  }
  CV_Assert(grayscaleImage.type() == CV_8UC1);
  CV_Assert(mask.type() == CV_8UC1);
  CV_Assert(grayscaleImage.size() == mask.size());

  vector<int> intensities;
  for (int i = 0; i < mask.rows; ++i)
  {
    for (int j = 0; j < mask.cols; ++j)
    {
      if (mask.at<uchar>(i, j) == 255)
      {
        intensities.push_back(grayscaleImage.at<uchar>(i, j));
      }
    }
  }

  std::sort(intensities.begin(), intensities.end());
  vector<int> boundaries;
  int currentBoundaryIndex = 0;
  int step = intensities.size() / clusterCount;
  int residual = intensities.size() % clusterCount;
  while (currentBoundaryIndex <= intensities.size())
  {
    boundaries.push_back(currentBoundaryIndex);
    currentBoundaryIndex += step;
    if (residual != 0)
    {
      ++currentBoundaryIndex;
      --residual;
    }
  }
  CV_Assert(boundaries.size() == clusterCount + 1);

  vector<float> clusterCenters(clusterCount);
  for (int i = 0; i < clusterCount; ++i)
  {
    int intensitiesSum = std::accumulate(intensities.begin() + boundaries[i], intensities.begin() + boundaries[i + 1], 0);
    clusterCenters[i] = static_cast<float>(intensitiesSum) / (boundaries[i + 1] - boundaries[i]);
  }

  intensityClusterCenters = cv::Mat(clusterCenters).clone();
  CV_Assert(intensityClusterCenters.cols == 1);
}

void computeColorSimilarity(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getColorHistogram();
  Mat hist_2 = region_2.getColorHistogram();

  //TODO: experiment with different distances
  distance = norm(hist_1 - hist_2);
}

//TODO: what if I = I_B?
void computeOverlayConsistency(const Region &region_1, const Region &region_2, float &slope, float &intercept)
{
  //TODO: move up
  const float minAlpha =-0.001f;
  const float maxAlpha = 1.001f;

  Mat clusters_1 = region_1.getIntensityClusters();
  Mat clusters_2 = region_2.getIntensityClusters();

  Mat b = clusters_1.clone();
  const int dim = 2;
  Mat A = Mat(b.rows, dim, b.type());
  Mat col_0 = A.col(0);
  clusters_2.copyTo(col_0);
  A.col(1).setTo(1.0);

  Mat model;
  solve(A, b, model, DECOMP_SVD);
  CV_Assert(model.type() == CV_32FC1);
  CV_Assert(model.total() == dim);
  if (model.at<float>(0) < minAlpha || model.at<float>(0) > maxAlpha)
  {
    b = clusters_2.clone();
    clusters_1.copyTo(col_0);
    solve(A, b, model, DECOMP_SVD);
  }
  if (model.at<float>(0) < minAlpha || model.at<float>(0) > maxAlpha)
  {
    cout << A << endl;
    cout << b << endl;
    cout << model << endl;
    CV_Error(CV_StsError, "Cannot estimate overlay consistency");
  }

  slope = model.at<float>(0);
  intercept = model.at<float>(1);
}

void oversegmentImage(const cv::Mat &image, cv::Mat &segmentation)
{
  //TODO: move up
  const float sigma = 0.2f;
  const float k = 500.0f;
  const int min_size = 200;

  const string sourceFilename = "imageForSegmenation.ppm";
  const string outputImageFilename = "segmentedImage.ppm";
  const string outputTxtFilename = "segmentation.txt";

  //TODO: re-implement
  imwrite(sourceFilename, image);
  std::stringstream command;
  command << "./segment " << sigma << " " << k << " " << min_size << " " << sourceFilename << " " << outputImageFilename;

  std::system(command.str().c_str());
  sleep(2);

  segmentation.create(image.size(), CV_32SC1);
  std::ifstream segmentationTxt(outputTxtFilename.c_str());
  CV_Assert(segmentationTxt.is_open());
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      segmentationTxt >> segmentation.at<int>(i, j);
    }
  }
  segmentationTxt.close();
}

void showSegmentation(const cv::Mat &segmentation, const std::string &title)
{
  CV_Assert(segmentation.type() == CV_32SC1);
  Vec3b *colors = new Vec3b[segmentation.total()];
  for (size_t i = 0; i < segmentation.total(); ++i)
  {
    colors[i] = Vec3b(56 + rand() % 200, 56 + rand() % 200, 56 + rand() % 200);
  }

  Mat image(segmentation.size(), CV_8UC3);
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      image.at<Vec3b>(i, j) = colors[segmentation.at<int>(i, j)];
    }
  }

  imshow(title, image);
  waitKey();
}


void segmentation2regions(const cv::Mat &image, cv::Mat &segmentation, vector<Region> &regions)
{
  CV_Assert(segmentation.type() == CV_32SC1);
  vector<int> labels = segmentation.reshape(1, 1);
  std::sort(labels.begin(), labels.end());
  vector<int>::iterator endIt = std::unique(labels.begin(), labels.end());
  labels.resize(endIt - labels.begin());

  int firstFreeLabel = 1 + *std::max_element(labels.begin(), labels.end());

  regions.clear();
  for (size_t i = 0; i < labels.size(); ++i)
  {
    Mat mask = (segmentation == labels[i]);
    segmentation.setTo(firstFreeLabel + i, mask);
    Region currentRegion(image, mask);
    regions.push_back(currentRegion);
  }
  segmentation -= firstFreeLabel;
}

enum TrainingLabels {THE_SAME = 0, GLASS_COVERED = 1};


void regions2sample(const Region &region_1, const Region &region_2, cv::Mat &sample)
{
  float colorDistance;
  computeColorSimilarity(region_1, region_2, colorDistance);
  float slope, intercept;
  computeOverlayConsistency(region_1, region_2, slope, intercept);

  const int dim = 3;
  sample = (Mat_<float>(1, dim) << colorDistance, slope, intercept);
}

void train(CvSVM &svm, float &normalizationSlope, float &normalizationIntercept)
{
  //TODO: move up
  const string trainingFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImages.txt";
  const string groundTruthFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImagesGroundTruth.txt";
  const float maxSampleDistance = 0.1f;

  vector<string> trainingGroundTruhFiles;
  readLinesInFile(groundTruthFilesList, trainingGroundTruhFiles);

  vector<string> trainingFiles;
  readLinesInFile(trainingFilesList, trainingFiles);

  const size_t imageCount = trainingGroundTruhFiles.size();
  CV_Assert(trainingFiles.size() == imageCount);

  Mat trainingData;
  vector<int> trainingLabelsVec;
  for (size_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
  {
    Mat trainingImage = imread(trainingFiles[imageIndex]);
    CV_Assert(!trainingImage.empty());

    Mat groundTruthMask = imread(trainingGroundTruhFiles[imageIndex], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!groundTruthMask.empty());
    CV_Assert(trainingImage.size() == groundTruthMask.size());

    Mat segmentation;
    oversegmentImage(trainingImage, segmentation);

    vector<Region> regions;
    segmentation2regions(trainingImage, segmentation, regions);
    vector<bool> isGlass(regions.size());
    for (size_t i = 0; i < regions.size(); ++i)
    {
      int regionArea = countNonZero(regions[i].getMask() != 0);
      int glassArea = countNonZero(regions[i].getMask() & groundTruthMask);
      const int glassFactor = 2;
      isGlass[i] = (glassFactor * glassArea > regionArea);
    }

    for (size_t i = 0; i < regions.size(); ++i)
    {
      for (size_t j = i + 1; j < regions.size(); ++j)
      {
        if (i == j)
        {
          continue;
        }

        Mat sample;
        regions2sample(regions[i], regions[j], sample);

        trainingData.push_back(sample);
        int currentLabel;
        if (isGlass[i] ^ isGlass[j])
        {
          currentLabel = GLASS_COVERED;
        }
        else
        {
          currentLabel = THE_SAME;
        }
        trainingLabelsVec.push_back(currentLabel);

        Mat symmetricSample;
        regions2sample(regions[j], regions[i], symmetricSample);
        if (norm(sample - symmetricSample) > maxSampleDistance)
        {
          //TODO: is it a correct way to process such cases?
          trainingData.push_back(symmetricSample);
          trainingLabelsVec.push_back(currentLabel);
        }
      }
    }
  }
  Mat trainingLabels = Mat(trainingLabelsVec).reshape(1, trainingLabelsVec.size());
  CV_Assert(trainingLabels.rows == trainingData.rows);

  CvSVMParams svmParams;
  //TODO: move up
  //svmParams.svm_type = CvSVM::C_SVC;

  svmParams.svm_type = CvSVM::NU_SVC;
  svmParams.nu = 0.5;
  cout << "trainingData size: " << trainingData.rows << " x " << trainingData.cols << endl;

  svm.train(trainingData, trainingLabels, Mat(), Mat(), svmParams);

  int wrongClassificationCount = 0;
  float minSVMDistance = 0.0f;
  float maxSVMDistance = 0.0f;
  for (size_t i = 0; i < trainingData.rows; ++i)
  {
    Mat sample = trainingData.row(i);
    //TODO: use one predict
    int label = cvRound(svm.predict(sample));
    float distance = svm.predict(sample, true);
    minSVMDistance = std::min(minSVMDistance, distance);
    maxSVMDistance = std::max(maxSVMDistance, distance);

    if (label != trainingLabelsVec[i])
    {
      ++wrongClassificationCount;
    }
  }

  float spread = maxSVMDistance - minSVMDistance;
  const float eps = 1e-2;
  CV_Assert(spread > eps);
  normalizationSlope = 1.0 / spread;
  normalizationIntercept = -minSVMDistance * normalizationSlope;

  //TODO: do you need this?
  normalizationSlope = -normalizationSlope;
  normalizationIntercept = -normalizationIntercept + 1;

  cout << "training error: " << static_cast<float>(wrongClassificationCount) / trainingData.rows << endl;
}

void visualizeClassification(const vector<Region> &regions, const vector<float> &labels, cv::Mat &visualization)
{
  if (visualization.empty())
  {
    visualization.create(regions[0].getMask().size(), CV_8UC1);
    visualization.setTo(0);
  }

  for (size_t i = 0; i < regions.size(); ++i)
  {
    int currentLabel = cvRound(labels[i]);
    if (currentLabel == GLASS_COVERED)
    {
      visualization.setTo(255, regions[i].getMask());
    }
  }
}







struct InteractiveClassificationData
{
  CvSVM *svm;
  Mat segmentation;
  vector<Region> regions;

  Graph graph;
};

void onMouse(int event, int x, int y, int, void *rawData)
{
  //TODO: move up
  const float regionEdgeLength = 1e6;

  if (event != CV_EVENT_LBUTTONDOWN)
  {
    return;
  }

  InteractiveClassificationData *data = static_cast<InteractiveClassificationData*>(rawData);

  int regionIndex = data->segmentation.at<int>(y, x);

  vector<float> labels(data->regions.size());
  for (size_t i = 0; i < data->regions.size(); ++i)
  {
    Mat sample;
    regions2sample(data->regions[regionIndex], data->regions[i], sample);
    labels[i] = cvRound(data->svm->predict(sample));
  }

  Mat visualization;
  visualizeClassification(data->regions, labels, visualization);
  imshow("classification", visualization);


  vector<float> affinities;
  estimateAffinities(data->graph, data->regions.size(), data->segmentation.size(), regionIndex, affinities);
  CV_Assert(data->regions.size() == affinities.size());

  Mat regionAffinities(data->segmentation.size(), CV_32FC1, Scalar(0));
  for (size_t i = 0; i < affinities.size(); ++i)
  {
    regionAffinities.setTo(affinities[i], data->regions[i].getMask());
  }
//  regionAffinities -= 2 * regionEdgeLength;
// regionAffinities.setTo(0, regionAffinities > regionEdgeLength);

  Mat affinityImage(regionAffinities.size(), CV_8UC1, Scalar(0));

  regionAffinities.convertTo(affinityImage, CV_8UC1, 255.0);
//  cout << regionAffinities << endl;
//  affinityImage.setTo(255, regionAffinities == 0.0);
 // regionAffinities.convertTo(affinityImage, CV_8UC1, 10.0);
  imshow("affinities", affinityImage);
}


//TODO: is it possible to use the index to access a vertex directly?
VertexDescriptor getRegionVertex(const Graph &graph, int regionIndex)
{
  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (graph[*vi].isRegion && graph[*vi].regionIndex == regionIndex)
    {
      return *vi;
    }
  }
  CV_Assert(false);
}

VertexDescriptor insertPoint(const cv::Mat &segmentation, const cv::Mat &orientations, cv::Point pt, Graph &graph)
{
  //TODO: move up
  const int maxDistanceToRegion = 3;
  const float regionEdgeLength = 1e6;


  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (pt == graph[*vi].pt)
    {
      return *vi;
    }
  }

  VertexDescriptor v = boost::add_vertex(graph);
  graph[v].pt = pt;
  CV_Assert(orientations.type() == CV_32FC1);
  graph[v].orientation = orientations.at<float>(pt.y, pt.x);


  CV_Assert(segmentation.type() == CV_32SC1);
  for (int dy = -maxDistanceToRegion; dy <= maxDistanceToRegion; ++dy)
  {
    for (int dx = -maxDistanceToRegion; dx <= maxDistanceToRegion; ++dx)
    {
      Point shiftedPt = pt + Point(dx, dy);
      if (!isPointInside(segmentation, shiftedPt))
      {
        continue;
      }

      int regionIndex = segmentation.at<int>(shiftedPt.y, shiftedPt.x);
      VertexDescriptor regionVertex = getRegionVertex(graph, regionIndex);
      bool isNew;
      EdgeDescriptor addedEdge;
      tie(addedEdge, isNew) = boost::add_edge(v, regionVertex, graph);
      if (isNew)
      {
        graph[addedEdge].length = regionEdgeLength;
      }
    }
  }

  return v;
}

void edges2graph(const cv::Mat &segmentation, const vector<Region> &regions, const cv::Mat &edges, Graph &graph)
{
  //TODO: move up
  const int medianIndex = 3;

  for (size_t i = 0; i < regions.size(); ++i)
  {
    VertexDescriptor v = boost::add_vertex(graph);
    graph[v].isRegion = true;
    graph[v].regionIndex = i;
    graph[v].pt = regions[i].getCenter();
  }

  Mat edgesMap = edges.clone();
  Mat orientations;
  computeEdgeOrientations(edgesMap, orientations, medianIndex);
  CV_Assert(orientations.type() == CV_32FC1);
  //TODO: check with NaNs and remove magic numbers

  edgesMap = edges.clone();
  edgesMap.setTo(0, ~((orientations >= -10.0) & (orientations <= 10.0)));

  vector<vector<Point> > contours;
  findContours(edgesMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  for (size_t contourIndex = 0; contourIndex < contours.size(); ++contourIndex)
  {
    VertexDescriptor previousVertex = insertPoint(segmentation, orientations, contours[contourIndex][0], graph);

    for (size_t edgelIndex = 1; edgelIndex < contours[contourIndex].size(); ++edgelIndex)
    {
      VertexDescriptor currentVertex = insertPoint(segmentation, orientations, contours[contourIndex][edgelIndex], graph);
      EdgeDescriptor currentEdge;
      bool isNewEdge;
      tie(currentEdge, isNewEdge) = boost::add_edge(previousVertex, currentVertex, graph);
      if (isNewEdge)
      {
        //TODO: use better estimation
        graph[currentEdge].length = norm(graph[previousVertex].pt - graph[currentVertex].pt);
        graph[currentEdge].maximumAngle = fabs(graph[previousVertex].orientation - graph[currentVertex].orientation);
      }
      previousVertex = currentVertex;
    }
  }
}

cv::Point getNextPoint(cv::Point previous, cv::Point current)
{
  Point next = current + (current - previous);
  return next;
}

//TODO: use orientations of endpoints
//TODO: process adjacent regions when the shortest path has the single edgel
bool areRegionsOnTheSameSide(const cv::Mat &path, Point firstPathEdgel, Point lastPathEdgel, Point center_1, Point center_2)
{
  //TODO: move up
  float minDistanceToEndpoints = 1.8f;
  Mat dilatedPath;
  dilate(path, dilatedPath, Mat());
  vector<vector<Point> > contours;
  findContours(dilatedPath, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  CV_Assert(contours.size() <= 1);
  if (contours.size() == 0)
  {
    return true;
  }

  for (vector<Point>::iterator it = contours[0].begin(); it != contours[0].end();)
  {
    if (norm(*it - firstPathEdgel) < minDistanceToEndpoints || norm(*it - lastPathEdgel) < minDistanceToEndpoints)
    {
      it = contours[0].erase(it);
    }
    else
    {
      ++it;
    }
  }

  Mat boundaries(path.size(), CV_8UC1, Scalar(0));
  drawContours(boundaries, contours, -1, Scalar(255));
  contours.clear();
  findContours(boundaries, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  CV_Assert(contours.size() <= 2);
  if (contours.size() != 2)
  {
    return true;
  }

  //TODO: remove code duplication
  Mat firstBoundary(path.size(), CV_8UC1, Scalar(0));
  drawContours(firstBoundary, contours, 0, Scalar(255));
  Mat firstDT;
  distanceTransform(~firstBoundary, firstDT, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  Mat secondBoundary(path.size(), CV_8UC1, Scalar(0));
  drawContours(secondBoundary, contours, 1, Scalar(255));
  Mat secondDT;
  distanceTransform(~secondBoundary, secondDT, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  bool isFirst_1 = firstDT.at<float>(center_1) < secondDT.at<float>(center_1);
  bool isFirst_2 = firstDT.at<float>(center_2) < secondDT.at<float>(center_2);

  return (!(isFirst_1 ^ isFirst_2));
}

//TODO: block junctions
//TODO: extend to the cases when regions are not connected to each other
void estimateAffinities(const Graph &graph, size_t regionCount, cv::Size imageSize, int regionIndex, std::vector<float> &affinities)
{
  //TODO: move up
  const float regionEdgeLength = 1e6;

  vector<double> distances(boost::num_vertices(graph));
  vector<VertexDescriptor> predecessors(boost::num_vertices(graph));
  boost::dijkstra_shortest_paths(graph, getRegionVertex(graph, regionIndex),
        boost::weight_map(boost::get(&EdgeProperties::length, graph))
        .distance_map(make_iterator_property_map(distances.begin(), get(boost::vertex_index, graph)))
        .predecessor_map(&predecessors[0]));


  //TODO: use dynamic programming
  affinities.resize(regionCount);
  int affinityIndex = -1;
  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (!graph[*vi].isRegion)
    {
      continue;
    }
    ++affinityIndex;

    if (graph[*vi].regionIndex == regionIndex)
    {
      affinities[affinityIndex] = 0.0f;
      continue;
    }

    VertexDescriptor currentVertex = *vi;
    VertexDescriptor predecessorVertex = predecessors[currentVertex];

    if (distances[affinityIndex] > 3 * regionEdgeLength || currentVertex == predecessorVertex)
    {
      affinities[affinityIndex] = CV_PI;
      continue;
    }

    Mat path(imageSize, CV_8UC1, Scalar(0));
    float maxAngle = 0.0f;
    Point firstEdgel = graph[predecessorVertex].pt;
    Point lastEdgel;
    Point predecessorLocation = graph[predecessorVertex].pt;
    bool pathIsInvalid = false;
    while (predecessorVertex != currentVertex)
    {
      if (currentVertex != *vi && graph[currentVertex].isRegion)
      {
        maxAngle = CV_PI;
        pathIsInvalid = true;
        break;
      }

      Point currentLocation = graph[currentVertex].pt;
      CV_Assert(isPointInside(path, currentLocation));
      CV_Assert(isPointInside(path, predecessorLocation));
      if (!graph[currentVertex].isRegion && !graph[predecessorVertex].isRegion)
      {
        line(path, currentLocation, predecessorLocation, Scalar(255));
      }

      EdgeDescriptor edge;
      bool doesExist;
      tie(edge, doesExist) = boost::edge(currentVertex, predecessorVertex, graph);
      CV_Assert(doesExist);
      maxAngle = std::max(maxAngle, graph[edge].maximumAngle);

      lastEdgel = graph[currentVertex].pt;
      currentVertex = predecessorVertex;
      predecessorVertex = predecessors[currentVertex];
      predecessorLocation = graph[predecessorVertex].pt;
    }
    if (pathIsInvalid)
    {
      affinities[affinityIndex] = CV_PI;
      continue;
    }
    if (areRegionsOnTheSameSide(path, firstEdgel, lastEdgel, graph[*vi].pt, graph[currentVertex].pt))
    {
      affinities[affinityIndex] = maxAngle;
    }
    else
    {
      affinities[affinityIndex] = CV_PI;
    }
  }

  for (size_t i = 0; i < affinities.size(); ++i)
  {
    affinities[i] = 1.0 - affinities[i] / CV_PI;
  }
}

void computeAllAffinities(const std::vector<Region> &regions, const Graph &graph, cv::Mat &affinities)
{
  affinities.create(regions.size(), regions.size(), CV_32FC1);
  affinities.setTo(-1);
  //TODO: use the Floyd-Warshall algorithm
  for (size_t regionIndex = 0; regionIndex < regions.size(); ++regionIndex)
  {
    vector<float> currentAffinities;
    estimateAffinities(graph, regions.size(), regions[0].getMask().size(), regionIndex, currentAffinities);
    CV_Assert(currentAffinities.size());
    Mat affinitiesMat = Mat(currentAffinities);
    Mat affinitiesFloat;
    affinitiesMat.convertTo(affinitiesFloat, CV_32FC1);
    Mat row = affinities.row(regionIndex);
    affinitiesFloat.reshape(1, 1).copyTo(row);
  }
}

void computeAllDiscrepancies(const std::vector<Region> &regions, const CvSVM *svm, float normalizationSlope, float normalizationIntercept, cv::Mat &discrepancies)
{
  discrepancies.create(regions.size(), regions.size(), CV_32FC1);
  for (size_t i = 0; i < regions.size(); ++i)
  {
    vector<float> labels(regions.size());
    for (size_t j = 0; j < regions.size(); ++j)
    {
      if (j == i)
      {
        labels[j] = 0.0f;
        continue;
      }

      Mat sample;
      regions2sample(regions[i], regions[j], sample);

      //TODO: use continious labels
      labels[j] = cvRound(svm->predict(sample));

      labels[j] = normalizationSlope * svm->predict(sample, true) + normalizationIntercept;
    }
    Mat row = discrepancies.row(i);
    Mat(labels).reshape(1, 1).copyTo(row);
  }
}

void computeBoundaryStrength(const cv::Mat &segmentation, const std::vector<Region> &regions, const Graph &graph, const CvSVM *svm, float normalizationSlope, float normalizationIntercept, float affinityWeight, cv::Mat &boundaryStrength)
{
  //TODO: move up
  const int neighborDistance = 1;


  CV_Assert(affinityWeight >= 0.0f && affinityWeight <= 1.0f);
  //TODO: use lazy evaluations
  Mat affinities;
  computeAllAffinities(regions, graph, affinities);
  Mat discrepancies;
  computeAllDiscrepancies(regions, svm, normalizationSlope, normalizationIntercept, discrepancies);

  CV_Assert(discrepancies.type() == CV_32FC1);
  CV_Assert(affinities.type() == CV_32FC1);

  Mat pixelDiscrepancies(segmentation.size(), CV_32FC1);
  Mat pixelAffinities(segmentation.size(), CV_32FC1);
  for (int i = 0; i < segmentation.rows; ++i)
  {
    for (int j = 0; j < segmentation.cols; ++j)
    {
      Point srcPt = Point(j, i);
      int srcRegionIndex = segmentation.at<int>(srcPt);
      float maxDiscrepancy = 0.0f;
      float maxAffinity = 0.0f;
      for (int dy = -neighborDistance; dy <= neighborDistance; ++dy)
      {
        for (int dx = -neighborDistance; dx <= neighborDistance; ++dx)
        {
          Point diffPt(dx, dy);
          Point pt = srcPt + diffPt;
          if (!isPointInside(segmentation, pt))
          {
            continue;
          }

          int regionIndex = segmentation.at<int>(pt);
          //TODO: what about symmetry
          maxDiscrepancy = std::max(maxDiscrepancy, discrepancies.at<float>(srcRegionIndex, regionIndex));
          maxAffinity = std::max(maxAffinity, affinities.at<float>(srcRegionIndex, regionIndex));
        }
      }
      pixelDiscrepancies.at<float>(srcPt) = maxDiscrepancy;
      pixelAffinities.at<float>(srcPt) = maxAffinity;
    }
  }

  boundaryStrength = (1.0 - affinityWeight) * pixelDiscrepancies - affinityWeight * pixelAffinities;
}

int main()
{
  const string svmFilename = "svm.xml";

  CvSVM svm;
  float normalizationSlope, normalizationIntercept;
//  train(svm, normalizationSlope, normalizationIntercept);
//  cout << normalizationSlope << endl;
//  cout << normalizationIntercept << endl;
//  svm.save(svmFilename.c_str());
  svm.load(svmFilename.c_str());
//  normalizationSlope = 0.163644f;
//  normalizationIntercept = 0.53197f;
  normalizationSlope = -0.163644f;
  normalizationIntercept = -0.53197f + 1.0f;

  const string testSegmentationTitle = "test segmentation";

//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Test/plate_building.jpg");




  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Training/teaB1f.jpg");
  CV_Assert(!testImage.empty());

  Mat oversegmentation;
  oversegmentImage(testImage, oversegmentation);
  vector<Region> regions;
  segmentation2regions(testImage, oversegmentation, regions);


  Mat grayscaleImage;
  cvtColor(testImage, grayscaleImage, CV_BGR2GRAY);
  Mat edges;
  Canny(grayscaleImage, edges, 100, 50);
  imshow("edges", edges.clone());
  Graph graph;
  edges2graph(oversegmentation, regions, edges, graph);

  namedWindow(testSegmentationTitle);

  Mat boundaryStrength;
  computeBoundaryStrength(oversegmentation, regions, graph, &svm, normalizationSlope, normalizationIntercept, 0.01f, boundaryStrength);

  imshow("boundary", boundaryStrength);
  waitKey();


  InteractiveClassificationData data;
  data.svm = &svm;
  data.segmentation = oversegmentation;
  data.regions = regions;
  data.graph = graph;

  setMouseCallback(testSegmentationTitle, onMouse, &data);
  showSegmentation(oversegmentation, testSegmentationTitle);

  return 0;
}
