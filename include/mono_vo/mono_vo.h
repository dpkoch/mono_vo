/**
 * \file mono_vo.h
 * \author Daniel Koch
 * \brief Monocular visual odometry implementation
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>


#include <cv_bridge/cv_bridge.h>

namespace mono_vo
{

static const std::string CURRENT_IMAGE_WINDOW = "Current Image";
static const std::string MATCHES_WINDOW = "Matches";

/**
 * \class MonoVO
 * \brief Provides a basic implementation of a monocular visual odometry algorithm.
 *
 * Methodology and implementation details were taken from the references listed below:
 *
 * [1] B. D. Scaramuzza and F. Fraundorfer, “Visual Odometry,” IEEE Robot. Autom. Mag., no. December, pp. 80–92, 2011.
 * [2] B. F. Fraundorfer and D. Scaramuzza, “Visual Odometry,” IEEE Robot. Autom. Mag., no. June, pp. 78–90, 2012.
 * [3] D. Nister, O. Naroditsky, and J. Bergen, “Visual Odometry,” in Proceedings of the International Computer Society
 *     Conference on Computer Vision and Pattern Recognition, 2004, vol. 1, pp. 652–659.
 * [4] D. Nistér, “Preemptive RANSAC for live structure and motion estimation,” Mach. Vis. Appl., vol. 16, no. 5,
 *     pp. 321–329, Nov. 2005.
 */
class MonoVO
{
private:

  //---------------------------------------------------------------------------
  // data members
  //---------------------------------------------------------------------------

  // state
  bool initialized_;
  int ignore_images_;

  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber image_subscriber_;
  ros::Publisher pose_publisher_;

  // VO
  cv_bridge::CvImagePtr keyframe_image_;
  std::vector<cv::KeyPoint> keyframe_features_;
  cv::Mat keyframe_descriptors_;

  cv::FeatureDetector* feature_detector_;
  cv::DescriptorExtractor* descriptor_extractor_;
  cv::DescriptorMatcher* forward_matcher_;
  cv::DescriptorMatcher* reverse_matcher_;

  //---------------------------------------------------------------------------
  // private methods
  //---------------------------------------------------------------------------

  // callbacks
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

  // helper functions
  void detectFeatures(const cv_bridge::CvImagePtr& image, std::vector<cv::KeyPoint>& keypoints);
  void computeDescriptors(const cv_bridge::CvImagePtr& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  void computeMatches(const cv::Mat& current_descriptors, const cv::Mat& keyframe_descriptors, std::vector<cv::DMatch>& matches);

public:

  MonoVO();
  ~MonoVO();
};

} // namespace mono_vo
