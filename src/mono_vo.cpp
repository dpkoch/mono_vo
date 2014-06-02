/**
 * \file mono_vo.cpp
 * \author Daniel Koch
 */

#include "mono_vo.h"
#include <omp.h>

namespace mono_vo
{

MonoVO::MonoVO() :
  initialized_(false),
  ignore_images_(30),
  nh_(),
  nh_private_("~")
{
  // ROS
  image_subscriber_ = nh_.subscribe("image", 1, &MonoVO::imageCallback, this);
  pose_publisher_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 5);

  // OpenCV
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");
  feature_detector_ = new cv::GridAdaptedFeatureDetector(detector, 750, 6, 8);
  descriptor_extractor_ = new cv::BriefDescriptorExtractor(64);
  forward_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);
  reverse_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);

  // image display
  cv::namedWindow(CURRENT_IMAGE_WINDOW);
  cv::namedWindow(MATCHES_WINDOW);
}

MonoVO::~MonoVO()
{
  delete feature_detector_;
  delete descriptor_extractor_;
  delete forward_matcher_;
  delete reverse_matcher_;

  cv::destroyWindow(CURRENT_IMAGE_WINDOW);
  cv::destroyWindow(MATCHES_WINDOW);
}

void MonoVO::imageCallback(const sensor_msgs::Image::ConstPtr &msg)
{
  ROS_INFO_ONCE("first image received, encoding is %s", msg->encoding.c_str());

  // try converting image to OpenCV grayscale image format
  cv_bridge::CvImagePtr image;
  try
  {
    image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // initialize if necessary
  if (!initialized_)
  {
    if (ignore_images_ == 0)
    {
      keyframe_image_ = image;
      detectFeatures(keyframe_image_, keyframe_features_);
      computeDescriptors(keyframe_image_, keyframe_features_, keyframe_descriptors_);

      initialized_ = true;
      return;
    }
    else
    {
      ignore_images_--;
      return;
    }
  }

  ROS_INFO_ONCE("initialized, %d features detected on keyframe image", keyframe_features_.size());

  // detect features
  std::vector<cv::KeyPoint> keypoints;
  detectFeatures(image, keypoints);

  // draw keypoints on image and display
  cv::Mat image_keypoints;
  cv::drawKeypoints(image->image,
                    keypoints,
                    image_keypoints,
                    cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow(CURRENT_IMAGE_WINDOW, image_keypoints);
  cv::waitKey(3);

  // compute descriptors
  cv::Mat descriptors;
  computeDescriptors(image, keypoints, descriptors);

  // compute matches
  std::vector<cv::DMatch> matches;
  ros::WallTime start = ros::WallTime::now();
  computeMatches(keyframe_descriptors_, descriptors, matches);
  ros::WallDuration compute_time = ros::WallTime::now() - start;
  ROS_INFO_THROTTLE(1, "Computing matches took %f seconds", compute_time.toSec());

  // draw matches on image and display
  cv::Mat image_matches;
  cv::drawMatches(keyframe_image_->image, keyframe_features_, image->image, keypoints, matches, image_matches);
  cv::imshow(MATCHES_WINDOW, image_matches);
  cv::waitKey(3);
}

inline void MonoVO::detectFeatures(const cv_bridge::CvImagePtr &image, std::vector<cv::KeyPoint>& keypoints)
{
   feature_detector_->detect(image->image, keypoints);
}

inline void MonoVO::computeDescriptors(const cv_bridge::CvImagePtr &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
  descriptor_extractor_->compute(image->image, keypoints, descriptors);
}

void MonoVO::computeMatches(const cv::Mat& current_descriptors, const cv::Mat& keyframe_descriptors, std::vector<cv::DMatch>& matches)
{
  // initialize
  std::vector<cv::DMatch> forward;
  std::vector<cv::DMatch> reverse;

  // forward and reverse matches (run on parallel threads)
  #pragma omp parallel sections
  {
    #pragma omp section
    forward_matcher_->match(current_descriptors, keyframe_descriptors, forward);

    #pragma omp section
    reverse_matcher_->match(keyframe_descriptors, current_descriptors, reverse);
  }

  // mutual consistency check
  for (size_t i = 0; i < forward.size(); i++)
  {
    if (reverse[forward[i].trainIdx].trainIdx == i)
    {
      matches.push_back(forward[i]);
    }
  }
}

} // namespace mono_vo
