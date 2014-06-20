/**
 * \file mono_vo.h
 * \author Daniel Koch
 * \brief Monocular visual odometry implementation
 */

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <boost/bind.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <cv_bridge/cv_bridge.h>

#include <string>
#include <vector>

namespace mono_vo
{

static const std::string CURRENT_IMAGE_WINDOW = "Current Image";
static const std::string MATCHES_WINDOW = "Matches";

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> sync_policy_t;

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
 * [5] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press,
 *     2003.
 */
class MonoVO
{
private:

  //---------------------------------------------------------------------------
  // data members
  //---------------------------------------------------------------------------

  // state
  bool initialized_; //!< Flag for whether the VO algorithm has been initialized
  int ignore_images_; //!< The number of images to display on startup (to allow for exposure level to settle out)
  bool display_; //!< Flag for whether to display debugging images

  // ROS
  ros::NodeHandle nh_; //!< Public node handle
  ros::NodeHandle nh_private_; //!< Private node handle

  message_filters::Subscriber<sensor_msgs::Image>* image_subscriber_; //!< Subscriber for RGB images
  message_filters::Subscriber<sensor_msgs::CameraInfo>* camera_info_subscriber_; //!< Subscriber for camera info topic
  message_filters::Synchronizer<sync_policy_t>* synchronizer_; //!< Message synchronizer to get image and calibration info at the same time

  ros::Publisher pose_publisher_; //!< Publisher for computed pose

  std::string frame_id_; //!< TF frame in which to publish pose messages

  // VO
  double inlier_threshold_; //!< Distance threshold for a keypoint to be considered an inlier

  cv_bridge::CvImagePtr keyframe_image_; //!< The keyframe image
  std::vector<cv::KeyPoint> keyframe_keypoints_pixel_; //!< The keyframe feature keypoints in pixel coordinates
  std::vector<cv::KeyPoint> keyframe_keypoints_; //!< The keyframe feature keypoints in normalized image coordinates
  cv::Mat keyframe_descriptors_; //!< The descriptors associated with the keyframe features

  cv::RNG* rng_; //!< Random number generator for creating RANSAC samples

  cv::FeatureDetector* feature_detector_; //!< OpenCV feature detector
  cv::DescriptorExtractor* descriptor_extractor_; //!< OpenCV descriptor extractor
  cv::DescriptorMatcher* forward_matcher_; //!< OpenCV descriptor matcher for forward search
  cv::DescriptorMatcher* reverse_matcher_; //!< OpenCV descriptor matcher for reverse search

  cv::Mat K_; //!< Intrinsic camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat D_; //!< Camera distortion parameters [k1 k2 p1 p2 k3]

  //---------------------------------------------------------------------------
  // private methods
  //---------------------------------------------------------------------------

  // callbacks
  void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg, const sensor_msgs::CameraInfo::ConstPtr &info_msg);

  // helper functions

  /**
   * \brief Detects features in the given image
   * \param[in] image The image on which to detect features
   * \param[out] keypoints_pixel The detected features in pixel coordinates
   * \param[out] keypoints The detected features in normalized coordinates
   */
  void detectFeatures(const cv_bridge::CvImagePtr& image,
                      std::vector<cv::KeyPoint>* keypoints_pixel,
                      std::vector<cv::KeyPoint>* keypoints);

  /**
   * \brief Computes descriptors from an image for a given list of keypoints
   * \param[in] image The image from which to compute descriptors
   * \param[in,out] keypoints The keypoints for which to compute descriptors. Keypoints for which descriptors cannot be
   *                          computed will be removed from the list
   * \param[out] descriptors The computed descriptors
   */
  void computeDescriptors(const cv_bridge::CvImagePtr& image,
                          std::vector<cv::KeyPoint>* keypoints,
                          cv::Mat* descriptors);

  /**
   * \brief Computes correspondences between two sets of features using a brute force search and forward/reverse mutual
   * consistency check
   * \param[in] reference_descriptors List of descriptors for the reference image
   * \param[in] current_descriptors List of descriptors for the current image
   * \param[out] matches List of correspondences between the reference and current images
   */
  void computeMatches(const cv::Mat& reference_descriptors,
                      const cv::Mat& current_descriptors,
                      std::vector<cv::DMatch>* matches);

  /**
   * \brief Selects a random sample of corresponding keypoints
   *
   * \attention The input feature lists must be ordered, i.e. reference_keypoints[i] must correspond to
   * current_keypoints[i] for all values of i. The sample will similarly be returned in an ordered fashion.
   *
   * \param[in] sample_size The desired sample size
   * \param[in] reference_keypoints The ordered list of reference keypoints from which to select the sample
   * \param[in] current_keypoints The ordered list of current keypoints from which to select the sample
   * \param[out] reference_sample The ordered list of reference keypoints in the random sample
   * \param[out] current_sample The ordered list of current keypoints in the random sample
   */
  void getRandomSample(int sample_size,
                       const std::vector<cv::KeyPoint>& reference_keypoints,
                       const std::vector<cv::KeyPoint>& current_keypoints,
                       std::vector<cv::KeyPoint>* reference_sample,
                       std::vector<cv::KeyPoint>* current_sample);

  /**
   * \brief Computes the set of solutions for a given sample
   *
   * \attention The sample keypoint lists must be ordered, i.e. reference_sample[i] must correspond to current_sample[i]
   * for all values of i.
   *
   * \param[in] reference_sample The ordered sample of reference keypoints
   * \param[in] current_sample The ordered sample of current keypoints
   * \param[out] E The list of all valid solutions of the essential matrix for the sample
   * \param[out] P The list of all valid solutions of the projection matrix for the sample
   * \return True if the solution was successfully computed, false otherwise
   */
  bool sampleSolution(const std::vector<cv::KeyPoint>& reference_sample,
                      const std::vector<cv::KeyPoint>& current_sample,
                      std::vector<cv::Mat>* E,
                      std::vector<cv::Mat>* P);

  /**
   * \brief Computes the reprojection error associated with a solution
   *
   * \attention The input feature lists must be ordered, i.e. reference_keypoints[i] must correspond to
   * current_keypoints[i] for all values of i.
   *
   * \param[in] reference_keypoints The ordered list of keypoints for the reference image
   * \param[in] current_keypoints The ordered list of keypoints for the current image
   * \param[in] E The essential matrix solution
   * \param[in] P The projection matrix solution
   * \param[out] error The error associated with each keypoint correspondence
   * \param[out] inliers The list of indices in the keypoint lists of inliers for the current solution
   * \return The sum total error associated with the current solution
   */
  double solutionError(const std::vector<cv::KeyPoint>& reference_keypoints,
                       const std::vector<cv::KeyPoint>& current_keypoints,
                       const cv::Mat& E,
                       const cv::Mat& P,
                       std::vector<double>* error,
                       std::vector<int>* inliers);

  /**
   * \brief Runs the requested number of RANSAC iterations and returns the best solution
   *
   * \attention The input feature lists must be ordered, i.e. reference_keypoints[i] must correspond to
   * current_keypoints[i] for all values of i. The sample will similarly be returned as ordered lists.
   *
   * \param[in] num_iterations The number of iterations to run
   * \param[in] reference_keypoints The ordered list of reference keypoints
   * \param[in] current_keypoints The ordered list of current keypoints
   * \param[out] E The best essential matrix solution
   * \param[out] P The best projection matrix solution
   * \param[out] inliers The list of indices in the keypoint lists of inliers to the best solution
   */
  void runRANSAC(int num_iterations,
                 const std::vector<cv::KeyPoint>& reference_keypoints,
                 const std::vector<cv::KeyPoint>& current_keypoints,
                 cv::Mat* E,
                 cv::Mat* P,
                 std::vector<int>* inliers);

public:

  MonoVO();
  ~MonoVO();
};

} // namespace mono_vo
