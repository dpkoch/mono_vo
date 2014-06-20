/**
 * \file mono_vo.cpp
 * \author Daniel Koch
 */

#include "mono_vo.h"
#include <5point.h>
#include <omp.h>

namespace mono_vo
{

MonoVO::MonoVO() :
  initialized_(false),
  ignore_images_(30),
  nh_(),
  nh_private_("~"),
  inlier_threshold_(0.0005)
{
  // parameters
  nh_private_.param<bool>("display", display_, false);
  nh_private_.param<std::string>("frame_id", frame_id_, "mono_vo");

  // ROS
  image_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image", 1);
  camera_info_subscriber_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh_, "camera_info", 1);
  synchronizer_ = new message_filters::Synchronizer<sync_policy_t>(sync_policy_t(2), *image_subscriber_, *camera_info_subscriber_);
  synchronizer_->registerCallback(boost::bind(&MonoVO::imageCallback, this, _1, _2));

  pose_publisher_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 5);

  // OpenCV
  rng_ = new cv::RNG();

  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");
  feature_detector_ = new cv::GridAdaptedFeatureDetector(detector, 750, 6, 8);
  descriptor_extractor_ = new cv::BriefDescriptorExtractor(64);
  forward_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);
  reverse_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);

  // image display
  if (display_)
  {
    cv::namedWindow(CURRENT_IMAGE_WINDOW);
    cv::namedWindow(MATCHES_WINDOW);
  }
}

MonoVO::~MonoVO()
{
  delete synchronizer_;
  delete image_subscriber_;
  delete camera_info_subscriber_;

  delete rng_;
  delete feature_detector_;
  delete descriptor_extractor_;
  delete forward_matcher_;
  delete reverse_matcher_;

  if (display_)
  {
    cv::destroyWindow(CURRENT_IMAGE_WINDOW);
    cv::destroyWindow(MATCHES_WINDOW);
  }
}

void MonoVO::imageCallback(const sensor_msgs::Image::ConstPtr &image_msg, const sensor_msgs::CameraInfo::ConstPtr &info_msg)
{
  ROS_INFO_ONCE("first image received, encoding is %s", image_msg->encoding.c_str());

  // try converting image to OpenCV grayscale image format
  cv_bridge::CvImagePtr image;
  try
  {
    image = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // initialize if necessary
  if (!initialized_)
  {
    if (ignore_images_ == 0) // throw away the first few images
    {
      // camera calibration parameters
      ROS_ASSERT_MSG(info_msg->distortion_model == sensor_msgs::distortion_models::PLUMB_BOB,
                     "Expected distortion model to be %s, got %s.",
                     sensor_msgs::distortion_models::PLUMB_BOB.c_str(),
                     info_msg->distortion_model.c_str());
      K_ = (cv::Mat_<double>(3,3) << info_msg->K[0], info_msg->K[1], info_msg->K[2],
                                    info_msg->K[3], info_msg->K[4], info_msg->K[5],
                                    info_msg->K[6], info_msg->K[7], info_msg->K[8]);
      D_ = (cv::Mat_<double>(1,5) << info_msg->D[0], info_msg->D[1], info_msg->D[2], info_msg->D[3], info_msg->D[4]);

      // keyframe image
      keyframe_image_ = image;
      detectFeatures(keyframe_image_, &keyframe_keypoints_pixel_, &keyframe_keypoints_);
      computeDescriptors(keyframe_image_, &keyframe_keypoints_, &keyframe_descriptors_);

      initialized_ = true;
      return;
    }
    else
    {
      ignore_images_--;
      return;
    }
  }

  ROS_INFO_ONCE("initialized, %d features detected on keyframe image", (int)keyframe_keypoints_.size());

  // detect features
  std::vector<cv::KeyPoint> keypoints_pixel;
  std::vector<cv::KeyPoint> keypoints;
  detectFeatures(image, &keypoints_pixel, &keypoints);

  // draw keypoints on image and display
  if (display_)
  {
    cv::Mat image_keypoints;
    cv::drawKeypoints(image->image,
                      keypoints_pixel,
                      image_keypoints,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(CURRENT_IMAGE_WINDOW, image_keypoints);
    cv::waitKey(3);
  }

  // compute descriptors
  cv::Mat descriptors;
  computeDescriptors(image, &keypoints, &descriptors);

  // compute matches
  std::vector<cv::DMatch> matches;
  ros::WallTime start = ros::WallTime::now();
  computeMatches(keyframe_descriptors_, descriptors, &matches);
  ros::WallDuration compute_time = ros::WallTime::now() - start;
  ROS_INFO_THROTTLE(1, "Computing matches took %f seconds", compute_time.toSec());

  // build correspondences vectors
  std::vector<cv::KeyPoint> keyframe_correspondences;
  std::vector<cv::KeyPoint> current_correspondences;
  for (size_t i = 0; i < matches.size(); i++)
  {
    keyframe_correspondences.push_back(keyframe_keypoints_[matches[i].trainIdx]);
    current_correspondences.push_back(keypoints[matches[i].queryIdx]);
  }

  // run 5-point RANSAC
  cv::Mat E;
  cv::Mat P;
  std::vector<int> inliers;
  start = ros::WallTime::now();
  runRANSAC(100, keyframe_correspondences, current_correspondences, &E, &P, &inliers);
  compute_time = ros::WallTime::now() - start;
  ROS_INFO_THROTTLE(1, "%d iterations of RANSAC took %f seconds, returned %d inliers", 100, compute_time.toSec(), (int)inliers.size());

  // draw correspondences and inliers on image and display
  if (display_)
  {
    cv::Mat image_matches;
    cv::addWeighted(keyframe_image_->image, 0.5, image->image, 0.5, 0.0, image_matches);
    cv::cvtColor(image_matches, image_matches, CV_GRAY2BGR);

    cv::Scalar inlier_color(0,255,0,0);
    cv::Scalar correspondence_color(220,220,220,0);

    // draw correspondences
    for (size_t i = 0; i < keyframe_correspondences.size(); i++)
    {
      cv::Scalar color = (std::find(inliers.begin(), inliers.end(), i) != inliers.end()) ? inlier_color : correspondence_color;

      cv::circle(image_matches, keyframe_correspondences[i].pt, 2, color, 2);
      cv::line(image_matches, keyframe_correspondences[i].pt, current_correspondences[i].pt, color, 1);
      cv::circle(image_matches, current_correspondences[i].pt, 2, color, 1);
    }

    cv::imshow(MATCHES_WINDOW, image_matches);
    cv::waitKey(3);
  }
}

inline void MonoVO::detectFeatures(const cv_bridge::CvImagePtr &image,
                                   std::vector<cv::KeyPoint>* keypoints_pixel,
                                   std::vector<cv::KeyPoint>* keypoints)
{
   feature_detector_->detect(image->image, *keypoints);
   cv::undistortPoints(*keypoints_pixel, *keypoints, K_, D_);
}

inline void MonoVO::computeDescriptors(const cv_bridge::CvImagePtr& image,
                                       std::vector<cv::KeyPoint>* keypoints,
                                       cv::Mat* descriptors)
{
  descriptor_extractor_->compute(image->image, *keypoints, *descriptors);
}

void MonoVO::computeMatches(const cv::Mat& reference_descriptors,
                            const cv::Mat& current_descriptors,
                            std::vector<cv::DMatch>* matches)
{
  // initialize
  std::vector<cv::DMatch> forward;
  std::vector<cv::DMatch> reverse;

  // forward and reverse matches (run on parallel threads)
  #pragma omp parallel sections
  {
    #pragma omp section
    forward_matcher_->match(current_descriptors, reference_descriptors, forward);

    #pragma omp section
    reverse_matcher_->match(reference_descriptors, current_descriptors, reverse);
  }

  // mutual consistency check
  for (size_t i = 0; i < forward.size(); i++)
  {
    if (reverse[forward[i].trainIdx].trainIdx == i)
    {
      (*matches).push_back(forward[i]);
    }
  }
}

void MonoVO::getRandomSample(int sample_size,
                             const std::vector<cv::KeyPoint>& reference_keypoints,
                             const std::vector<cv::KeyPoint>& current_keypoints,
                             std::vector<cv::KeyPoint>* reference_sample,
                             std::vector<cv::KeyPoint>* current_sample)
{
  ROS_ASSERT(reference_keypoints.size() >= sample_size);
  ROS_ASSERT(current_keypoints.size() == reference_keypoints.size());

  reference_sample->clear();
  current_sample->clear();

  int population_size = reference_keypoints.size();
  std::set<int> sample_set;

  int num_samples = 0;
  while (num_samples < sample_size)
  {
    int index = rng_->uniform(0, population_size - 1);
    if (sample_set.find(index) == sample_set.end())
    {
      reference_sample->push_back(reference_keypoints[index]);
      current_sample->push_back(current_keypoints[index]);
      num_samples++;
    }
  }
}

bool MonoVO::sampleSolution(const std::vector<cv::KeyPoint>& reference_sample,
                            const std::vector<cv::KeyPoint>& current_sample,
                            std::vector<cv::Mat>* E,
                            std::vector<cv::Mat>* P)
{
  ROS_ASSERT(reference_sample.size() >= 5);
  ROS_ASSERT(current_sample.size() == reference_sample.size());

  std::vector<int> ret_inliers;
  return Solve5PointEssential(reference_sample, current_sample, *E, *P, ret_inliers);
}

double MonoVO::solutionError(const std::vector<cv::KeyPoint>& reference_keypoints,
                             const std::vector<cv::KeyPoint>& current_keypoints,
                             const cv::Mat& E,
                             const cv::Mat& P,
                             std::vector<double>* error,
                             std::vector<int>* inliers)
{
  // initialize
  double total_error = 0;
  error->clear();
  inliers->clear();

  // initialize points and vectors
  cv::Mat reference_pt(3, 1, CV_64F);
  cv::Mat current_pt(3, 1, CV_64F);
  cv::Mat reference_vec(3, 1, CV_64F);
  cv::Mat current_vec(3, 1, CV_64F);

  // compute first-order Sampson distance at each point (Multiple View Geometry, Equation (11.9))
  for (size_t i = 0; i < reference_keypoints.size(); i++)
  {
    // points
    reference_pt.at<double>(0,0) = reference_keypoints[i].pt.x;
    reference_pt.at<double>(1,0) = reference_keypoints[i].pt.y;
    reference_pt.at<double>(2,0) = 1.0;

    current_pt.at<double>(0,0) = current_keypoints[i].pt.x;
    current_pt.at<double>(1,0) = current_keypoints[i].pt.y;
    current_pt.at<double>(2,0) = 1.0;

    // vectors (part of Sampson distance equation)
    reference_vec = E * reference_pt;
    current_vec = E.t() * current_pt;

    // Sampson distance calculation
    double a = ((cv::Mat)(current_pt.t() * E * reference_pt)).at<double>(0,0);
    double b = reference_vec.at<double>(0,0);
    double c = reference_vec.at<double>(1,0);
    double d = current_vec.at<double>(0,0);
    double e = current_vec.at<double>(1,0);
    double dist = a*a / (b*b + c*c + d*d + e*e);

    error->push_back(dist);
    total_error += std::fabs(dist);

    if (std::fabs(dist) < inlier_threshold_)
    {
      inliers->push_back(i);
    }
  }
}

void MonoVO::runRANSAC(int num_iterations,
                       const std::vector<cv::KeyPoint>& reference_keypoints,
                       const std::vector<cv::KeyPoint>& current_keypoints,
                       cv::Mat* E,
                       cv::Mat* P,
                       std::vector<int>* inliers)
{
  ROS_ASSERT(reference_keypoints.size() >= 5);
  ROS_ASSERT(current_keypoints.size() == reference_keypoints.size());

  // initialize
  inliers->clear();

  // initialize ransac
  double best_error = std::numeric_limits<double>::max();
  int best_num_inliers = 0;

  cv::Mat best_E;
  cv::Mat best_P;
  std::vector<int> best_inliers;

  // RANSAC loop
  #pragma omp parallel for shared(best_error, best_num_inliers)
  for (size_t i = 0; i < num_iterations; i++)
  {
    // get random sample
    std::vector<cv::KeyPoint> reference_sample;
    std::vector<cv::KeyPoint> current_sample;
    getRandomSample(5, reference_keypoints, current_keypoints, &reference_sample, &current_sample);

    // get potential solutions for sample
    std::vector<cv::Mat> ret_E;
    std::vector<cv::Mat> ret_P;
    sampleSolution(reference_sample, current_sample, &ret_E, &ret_P);

    // find best solution for the sample
    double best_error_temp;
    int best_num_inliers_temp;
    cv::Mat best_E_temp;
    cv::Mat best_P_temp;
    std::vector<int> best_inliers_temp;
    for (size_t j = 0; j < ret_E.size(); j++)
    {
      std::vector<double> errors_temp;
      std::vector<int> inliers_temp;
      double error_temp = solutionError(reference_keypoints, current_keypoints, ret_E[j], ret_P[j], &errors_temp, &inliers_temp);

      if (inliers_temp.size() > best_num_inliers_temp || (inliers_temp.size() == best_num_inliers_temp && error_temp < best_error_temp))
      {
        best_error_temp = error_temp;
        best_num_inliers_temp = inliers_temp.size();
        best_E_temp = ret_E[j];
        best_P_temp = ret_P[j];
        best_inliers_temp = inliers_temp;
      }
    }

    // check if best sample solution is best solution so far
    #pragma omp critical
    if (best_num_inliers_temp > best_num_inliers
        || (best_num_inliers_temp == best_num_inliers && best_error_temp < best_error))
    {
      best_error = best_error_temp;
      best_num_inliers = best_num_inliers_temp;
      best_E = best_E_temp;
      best_P = best_P_temp;
      best_inliers = best_inliers_temp;
    }
  }

  // return final values
  *E = best_E;
  *P = best_P;
  *inliers = best_inliers;
}

} // namespace mono_vo
