/*
 * poseEstimator.hpp
 *
 *  Created on: Dec 2, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef POSEESTIMATOR_HPP_
#define POSEESTIMATOR_HPP_

#include "edges_pose_refiner/edgeModel.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"
#include "edges_pose_refiner/poseRT.hpp"

#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>

struct PoseEstimatorParams
{
  //training parameters
  int silhouetteCount;
  float downFactor;
  int closingIterationsCount;

  //edge detection parameters
  double cannyThreshold1;
  double cannyThreshold2;
  int dilationsForEdgesRemovalCount;

  size_t minGlassContourLength;
  double minGlassContourArea;

  float confidentDomination;

  PoseEstimatorParams()
  {
    silhouetteCount = 10;
    downFactor = 1.0f;
    closingIterationsCount = 10;

    minGlassContourLength = 10;
    minGlassContourArea = 64.0;

    cannyThreshold1 = 25;
    cannyThreshold2 = 50;
    dilationsForEdgesRemovalCount = 10;

    confidentDomination = 1.5f;
  }

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

class PoseEstimator
{
public:
  PoseEstimator(const PinholeCamera &kinectCamera = PinholeCamera(), const PoseEstimatorParams &params = PoseEstimatorParams());
  void addObject(const EdgeModel &edgeModel);
  void estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &poseQualities, const cv::Vec4f *tablePlane = 0) const;

  void read(const std::string &filename);
  void read(const cv::FileNode& fn);
  void write(const std::string &filename) const;
  void write(cv::FileStorage& fs) const;

  void visualize(const PoseRT &pose, cv::Mat &image, cv::Scalar color = cv::Scalar(0, 0, 255)) const;
  void visualize(const PoseRT &pose, const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, cv::Scalar color = cv::Scalar(0, 0, 255), const std::string &title = "object") const;
private:
  void computeCentralEdges(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, cv::Mat &centralEdges, cv::Mat &silhouetteEdges) const;
  void getInitialPoses(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities) const;
  void refineInitialPoses(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &initPoses_cam, std::vector<float> &initPosesQualities) const;
  bool tmpComputeTableOrientation(const cv::Mat &centralBgrImage, cv::Vec4f &tablePlane, ros::Publisher *pt_pub) const;
  void findTransformationToTable(PoseRT &pose_cam, const cv::Vec4f &tablePlane, float &rotationAngle, ros::Publisher *pt_pub = 0, const cv::Mat finalJacobian = cv::Mat()) const;
  void refinePosesByTableOrientation(const cv::Vec4f &tablePlane, const cv::Mat &centralBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &initPosesQualities, ros::Publisher *pt_pub = 0) const;

  EdgeModel edgeModel;
  std::vector<Silhouette> silhouettes;

  PoseEstimatorParams params;
  PinholeCamera kinectCamera;
};

#endif /* POSEESTIMATOR_HPP_ */
