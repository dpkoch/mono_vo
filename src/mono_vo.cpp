/**
 * \file mono_vo.cpp
 * \author Daniel Koch
 */

#include "mono_vo.h"

#include <5point.h>

#include <omp.h>
#include <unordered_set>

namespace mono_vo
{

MonoVO::MonoVO() :
  initialized_(false),
  ignore_images_(30),
  nh_(),
  nh_private_("~")
{
  // parameters
  nh_private_.param<bool>("display_current", display_current_, false);
  nh_private_.param<bool>("display_matches", display_matches_, false);
  nh_private_.param<std::string>("frame_id", frame_id_, "mono_vo");
  nh_private_.param<int>("max_ransac_iterations", max_ransac_iterations_, 500);
  nh_private_.param<double>("inlier_threshold", inlier_threshold_, 3.0);

  // ROS
  image_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image", 1);
  camera_info_subscriber_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh_, "camera_info", 1);
  synchronizer_ = new message_filters::Synchronizer<sync_policy_t>(sync_policy_t(2), *image_subscriber_, *camera_info_subscriber_);
  synchronizer_->registerCallback(boost::bind(&MonoVO::imageCallback, this, _1, _2));

  pose_publisher_ = nh_.advertise<geometry_msgs::TransformStamped>("pose", 5);

  // OpenCV
  rng_ = new cv::RNG();

  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");
  feature_detector_ = new cv::GridAdaptedFeatureDetector(detector, 750, 6, 8);
  descriptor_extractor_ = new cv::BriefDescriptorExtractor(64);
  forward_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);
  reverse_matcher_ = new cv::BFMatcher(cv::NORM_HAMMING);

  // image display
  if (display_matches_)
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

  if (display_matches_)
  {
    cv::destroyWindow(CURRENT_IMAGE_WINDOW);
    cv::destroyWindow(MATCHES_WINDOW);
  }
}

void MonoVO::imageCallback(const sensor_msgs::Image::ConstPtr &image_msg,
                           const sensor_msgs::CameraInfo::ConstPtr &info_msg)
{
  ros::WallTime start = ros::WallTime::now();

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
  if (display_current_)
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
  computeMatches(keyframe_descriptors_, descriptors, &matches);

  // build correspondences vectors
  std::vector<cv::KeyPoint> keyframe_correspondences;
  std::vector<cv::KeyPoint> current_correspondences;
  for (size_t i = 0; i < matches.size(); i++)
  {
    keyframe_correspondences.push_back(keyframe_keypoints_[matches[i].trainIdx]);
    current_correspondences.push_back(keypoints[matches[i].queryIdx]);
  }

  // run 8-point RANSAC
  cv::Mat F;
  std::vector<int> inliers;
  runRANSAC(max_ransac_iterations_, keyframe_correspondences, current_correspondences, &F, &inliers);

  // least squares solution
  std::vector<cv::KeyPoint> reference_inliers;
  std::vector<cv::KeyPoint> current_inliers;
  for (size_t i = 0; i < inliers.size(); i++)
  {
    reference_inliers.push_back(keyframe_correspondences[inliers[i]]);
    current_inliers.push_back(current_correspondences[inliers[i]]);
  }
  normalized8pt(reference_inliers, current_inliers, &F);

  std::vector<double> errors;
  solutionError(keyframe_correspondences, current_correspondences, F, &errors, &inliers);
  ROS_INFO_THROTTLE(1, "Final number of inliers: %d", (int)inliers.size());

  int old_num_inliers;
  do
  {
    old_num_inliers = (int)inliers.size();

    reference_inliers.clear();
    current_inliers.clear();
    for (size_t i = 0; i < inliers.size(); i++)
    {
      reference_inliers.push_back(keyframe_correspondences[inliers[i]]);
      current_inliers.push_back(current_correspondences[inliers[i]]);
    }

    normalized8pt(reference_inliers, current_inliers, &F);
    solutionError(keyframe_correspondences, current_correspondences, F, &errors, &inliers);

  } while (inliers.size() != old_num_inliers); //! \todo add a max loop iterations limit

  // draw correspondences and inliers on image and display
  if (display_matches_)
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

  // compute the essential matrix (Equation (9.12) in [Hartley,2003])
  cv::Mat E = K_.t() * F * K_;

  // enforce constraint that SVD of E is U*diag(1,1,0)*Vt (see Result 9.18 in [Hartley,2003])
  cv::Mat diag = cv::Mat::zeros(3, 3, CV_64F);
  diag.at<double>(0,0) = 1.0;
  diag.at<double>(1,1) = 1.0;

  cv::SVD svd(E, cv::SVD::FULL_UV);
//  E = svd.u * diag * svd.vt;
//  ROS_INFO_STREAM("Actual diag(1,1,0) = " << svd.w);

  // compute possible projection matrices P' (of the form P' = [R|t], see Result 9.19 in [Hartley,2003])
  cv::Mat W = cv::Mat::zeros(3, 3, CV_64F);
  W.at<double>(0,1) = -1.0;
  W.at<double>(1,0) = 1.0;
  W.at<double>(2,2) = 1.0;

  svd(E, cv::SVD::FULL_UV);

  // extract translation up to scale and sign
  cv::Mat t_raw = (cv::Mat_<double>(3,1) << svd.u.at<double>(0,2), svd.u.at<double>(1,2), svd.u.at<double>(2,2)) / svd.w.at<double>(0,0);
//  ROS_INFO_STREAM("t = " << t << ", norm(t) = " << cv::norm(t));

  // compute possible rotation matrices
  cv::Mat R1 = svd.u * W * svd.vt;
  cv::Mat R2 = svd.u * W.t() * svd.vt;

  // test each possible solution to find valid solution
  int i;
  cv::Mat R;
  cv::Mat t;
  cv::Mat P = cv::Mat::eye(3, 4, CV_64F);
  for (i = 0; i < 4; i++)
  {
    // build projection matrix of the form P = [R|t]
    switch (i)
    {
    case 0:
      R = R1;
      t = t_raw;
      break;
    case 1:
      R = R1;
      t = -t_raw;
      break;
    case 2:
      R = R2;
      t = t_raw;
      break;
    case 3:
      R = R2;
      t = -t_raw;
      break;
    }

    cv::Mat Pp = (cv::Mat_<double>(3,4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
                                           R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
                                           R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));

    // triangulate a point X from x and x' using the computed projection matrix
    cv::Mat X = triangulatePoint(keyframe_correspondences[inliers[0]].pt, current_correspondences[inliers[0]].pt, Pp);

    // test if the reprojection of the triangulated point X is in front of both cameras
    if (pointDepth(X, P) > 0 && pointDepth(X, Pp) > 0)
      break;
  }

  // did we get a valid solution?
  if (i < 4)
  {
    geometry_msgs::TransformStamped msg;

    msg.header.frame_id = info_msg->header.frame_id;
    msg.header.stamp = image_msg->header.stamp;
    msg.child_frame_id = frame_id_;

    msg.transform.translation.x = t.at<double>(0,0);
    msg.transform.translation.y = t.at<double>(1,0);
    msg.transform.translation.z = t.at<double>(2,0);

    tf::Matrix3x3 rotation(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
                           R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
                           R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2));

    tf::Quaternion quaternion;
    rotation.getRotation(quaternion);
    quaternion.normalize();

    tf::quaternionTFToMsg(quaternion, msg.transform.rotation);

    pose_publisher_.publish(msg);
    tf_broadcaster_.sendTransform(msg);
  }
  else
  {
    ROS_WARN("Valid solution not found");
  }

  ros::WallDuration time = (ros::WallTime::now() - start);
  ROS_INFO_THROTTLE(1, "Processing image took %f seconds, estimated rate at %.1f Hz", time.toSec(), 1.0/time.toSec());
}

inline void MonoVO::detectFeatures(const cv_bridge::CvImagePtr &image,
                                   std::vector<cv::KeyPoint>* keypoints_pixel,
                                   std::vector<cv::KeyPoint>* keypoints)
{
   feature_detector_->detect(image->image, *keypoints_pixel);
//   cv::undistortPoints(*keypoints_pixel, *keypoints, K_, D_);
   *keypoints = *keypoints_pixel;
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
  std::unordered_set<int> sample_set;

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

bool MonoVO::sampleSolution8pt(const std::vector<cv::KeyPoint>& reference_sample,
                               const std::vector<cv::KeyPoint>& current_sample,
                               std::vector<cv::Mat>* F)
{
  int num_points = 8;
  ROS_ASSERT(reference_sample.size() == num_points && current_sample.size() == reference_sample.size());

  F->clear();

  // build up A matrix (Equation (11.3) in Multiple View Geometry)
  cv::Mat A = cv::Mat_<double>(num_points, 9);
  for (int i = 0; i < num_points; i++)
  {
    double x = reference_sample[i].pt.x;
    double y = reference_sample[i].pt.y;
    double xp = current_sample[i].pt.x ;
    double yp = current_sample[i].pt.y;

    A.at<double>(i,0) = xp * x;
    A.at<double>(i,1) = xp * y;
    A.at<double>(i,2) = xp;
    A.at<double>(i,3) = yp * x;
    A.at<double>(i,4) = yp * y;
    A.at<double>(i,5) = yp;
    A.at<double>(i,6) = x;
    A.at<double>(i,7) = y;
    A.at<double>(i,8) = 1.0;
  }

  // solve (following section 11.3 in Multiple View Geometry)
  cv::Mat f = cv::Mat_<double>(9,1);
  cv::SVD::solveZ(A,f);

  F->push_back((cv::Mat_<double>(3,3) << f.at<double>(0,0), f.at<double>(1,0), f.at<double>(2,0),
                                         f.at<double>(3,0), f.at<double>(4,0), f.at<double>(5,0),
                                         f.at<double>(6,0), f.at<double>(7,0), f.at<double>(6,0)));

  return true;
}

bool MonoVO::sampleSolution7pt(const std::vector<cv::KeyPoint>& reference_sample,
                               const std::vector<cv::KeyPoint>& current_sample,
                               std::vector<cv::Mat>* F)
{
  int num_points = 7;
  ROS_ASSERT(reference_sample.size() == num_points && current_sample.size() == reference_sample.size());

  F->clear();

  // build up A matrix (equation (11.3) in Multiple View Geometry)
  cv::Mat A = cv::Mat_<double>(num_points, 9);
  for (int i = 0; i < num_points; i++)
  {
    double x = reference_sample[i].pt.x;
    double y = reference_sample[i].pt.y;
    double xp = current_sample[i].pt.x ;
    double yp = current_sample[i].pt.y;

    A.at<double>(i,0) = xp * x;
    A.at<double>(i,1) = xp * y;
    A.at<double>(i,2) = xp;
    A.at<double>(i,3) = yp * x;
    A.at<double>(i,4) = yp * y;
    A.at<double>(i,5) = yp;
    A.at<double>(i,6) = x;
    A.at<double>(i,7) = y;
    A.at<double>(i,8) = 1.0;
  }

  // compute right null space for A*f = 0 (basis vectors f1 and f2, refactored into F1 and F2 matrices)
  cv::Mat w;
  cv::Mat u;
  cv::Mat vt;
  cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A|cv::SVD::FULL_UV);

  cv::Mat F1 = (cv::Mat_<double>(3,3) << vt.at<double>(7,0), vt.at<double>(7,1), vt.at<double>(7,2),
                                         vt.at<double>(7,3), vt.at<double>(7,4), vt.at<double>(7,5),
                                         vt.at<double>(7,6), vt.at<double>(7,7), vt.at<double>(7,6));
  cv::Mat F2 = (cv::Mat_<double>(3,3) << vt.at<double>(8,0), vt.at<double>(8,1), vt.at<double>(8,2),
                                         vt.at<double>(8,3), vt.at<double>(8,4), vt.at<double>(8,5),
                                         vt.at<double>(8,6), vt.at<double>(8,7), vt.at<double>(8,6));

  // solve det(alpha*F1 + (1-alpha)*F2)=0 for alpha
  cv::Mat coeffs = cv::Mat_<double>(4,1);
  for (int i = 0; i < 6; i++)
  {
    double fa1 = F1.at<double>(0, DET_LOOKUP[i][0]);
    double fa2 = F2.at<double>(0, DET_LOOKUP[i][0]);
    double fb1 = F1.at<double>(1, DET_LOOKUP[i][1]);
    double fb2 = F2.at<double>(1, DET_LOOKUP[i][1]);
    double fc1 = F1.at<double>(2, DET_LOOKUP[i][2]);
    double fc2 = F2.at<double>(2, DET_LOOKUP[i][2]);
    int sign = DET_LOOKUP[i][3];

    coeffs.at<double>(0,0) += sign*((fa1 - fa2)*(fb1*fc1 - fb1*fc2 - fb2*fc1 + fb2*fc2));
    coeffs.at<double>(1,0) += sign*(fa1*(fb1*fc2 + fb2*fc1 - 2*fb2*fc2) + fa2*(fb1*fc1 - 2*fb1*fc2 - 2*fb2*fc1 + 3*fb2*fc2));
    coeffs.at<double>(2,0) += sign*(fa1*fb2*fc2 + fa2*(fb1*fc2 + fb2*fc1 - 3*fb2*fc2));
    coeffs.at<double>(3,0) += sign*(fa2*fb2*fc2);
  }

  cv::Mat roots;
  cv::solveCubic(coeffs, roots);

  // subsitute back into F = alpha*F1 + (1-alpha)*F2
  for (int i = 0; i < roots.rows; i++)
  {
    F->push_back(roots.at<double>(i,0)*F1 + (1 - roots.at<double>(i,0))*F2);
  }

  return true;
}

bool MonoVO::normalized8pt(const std::vector<cv::KeyPoint>& reference_keypoints,
                           const std::vector<cv::KeyPoint>& current_keypoints,
                           cv::Mat* F)
{
  ROS_ASSERT(reference_keypoints.size() >= 8 && current_keypoints.size() == reference_keypoints.size());
  int num_points = reference_keypoints.size();

  // find normalizing translations
  double sum_x = 0;
  double sum_y = 0;
  double sum_xp = 0;
  double sum_yp = 0;
  for (int i = 0; i < num_points; i++)
  {
    sum_x += reference_keypoints[i].pt.x;
    sum_y += reference_keypoints[i].pt.y;
    sum_xp += current_keypoints[i].pt.x;
    sum_yp += current_keypoints[i].pt.y;
  }
  double trans_x = -sum_x / num_points;
  double trans_y = -sum_y / num_points;
  double trans_xp = -sum_xp / num_points;
  double trans_yp = -sum_yp / num_points;

  // find normalizing scale factors (applied after translation)
  double sum_dist_sq = 0;
  double sum_dist_sq_p = 0;
  for (int i = 0; i < num_points; i++)
  {
    double x = reference_keypoints[i].pt.x + trans_x;
    double y = reference_keypoints[i].pt.y + trans_y;
    double xp = current_keypoints[i].pt.x + trans_xp;
    double yp = current_keypoints[i].pt.y + trans_yp;

    sum_dist_sq += x*x + y*y;
    sum_dist_sq_p += xp*xp + yp*yp;
  }
  double scale = std::sqrt(2.0 / sum_dist_sq);
  double scale_p = std::sqrt(2.0 / sum_dist_sq_p);

  // normalizing transformations
  cv::Mat T = (cv::Mat_<double>(3,3) << 1.0,  0,    trans_x,
                                        0,    1.0,  trans_y,
                                        0,    0,    1.0/scale);
  cv::Mat Tp = (cv::Mat_<double>(3,3) << 1.0,  0,    trans_xp,
                                         0,    1.0,  trans_yp,
                                         0,    0,    1.0/scale_p);

  // build up A matrix (Equation (11.3) in Multiple View Geometry) using normalized coordinates
  cv::Mat A = cv::Mat_<double>(num_points,9);
  for (int i = 0; i < num_points; i++)
  {
    double x = scale * (reference_keypoints[i].pt.x + trans_x);
    double y = scale * (reference_keypoints[i].pt.y + trans_y);
    double xp = scale_p * (current_keypoints[i].pt.x + trans_xp);
    double yp = scale_p * (current_keypoints[i].pt.y + trans_yp);

    A.at<double>(i,0) = xp * x;
    A.at<double>(i,1) = xp * y;
    A.at<double>(i,2) = xp;
    A.at<double>(i,3) = yp * x;
    A.at<double>(i,4) = yp * y;
    A.at<double>(i,5) = yp;
    A.at<double>(i,6) = x;
    A.at<double>(i,7) = y;
    A.at<double>(i,8) = 1.0;
  }

  // solve least squares solution
  cv::Mat f = cv::Mat_<double>(9,1);
  cv::SVD::solveZ(A,f);

  *F = (cv::Mat_<double>(3,3) << f.at<double>(0,0), f.at<double>(1,0), f.at<double>(2,0),
                                 f.at<double>(3,0), f.at<double>(4,0), f.at<double>(5,0),
                                 f.at<double>(6,0), f.at<double>(7,0), f.at<double>(8,0));

//  cv::SVD svd(A);
//  *F = (cv::Mat_<double>(3,3) << svd.vt.at<double>(8,0), svd.vt.at<double>(8,1), svd.vt.at<double>(8,2),
//                                 svd.vt.at<double>(8,3), svd.vt.at<double>(8,4), svd.vt.at<double>(8,5),
//                                 svd.vt.at<double>(8,6), svd.vt.at<double>(8,7), svd.vt.at<double>(8,8));

  // enforce singularity constraint (Section 11.1.1 in [Hartley,2003])
  cv::SVD svd(*F);
  ROS_INFO_THROTTLE(1, "Values of D: [%f, %f, %f]", svd.w.at<double>(0,0), svd.w.at<double>(1,0), svd.w.at<double>(2,0));
  *F = svd.u * cv::Mat::diag((cv::Mat_<double>(3,1) << svd.w.at<double>(0,0), svd.w.at<double>(1,0), 0.0)) * svd.vt;

  // denormalize
  *F = Tp.t() * (*F) * T;
}

double MonoVO::solutionError(const std::vector<cv::KeyPoint>& reference_keypoints,
                             const std::vector<cv::KeyPoint>& current_keypoints,
                             const cv::Mat& F,
                             std::vector<double>* error,
                             std::vector<int>* inliers)
{
  // initialize
  double total_error = 0;
  error->clear();
  inliers->clear();

  double F_array[3][3] = { { F.at<double>(0,0), F.at<double>(0,1), F.at<double>(0,2) },
                           { F.at<double>(1,0), F.at<double>(1,1), F.at<double>(1,2) },
                           { F.at<double>(2,0), F.at<double>(2,1), F.at<double>(2,2) }};

  // compute first-order Sampson distance at each point
  for (size_t i = 0; i < reference_keypoints.size(); i++)
  {
    double x[2] = { reference_keypoints[i].pt.x, reference_keypoints[i].pt.y };
    double xp[2] = { current_keypoints[i].pt.x, current_keypoints[i].pt.y };
    double dist = sampsonDistance(x, xp, F_array);

    error->push_back(dist);
    total_error += std::fabs(dist);

    if (std::fabs(dist) < inlier_threshold_)
    {
      inliers->push_back(i);
    }
  }
}

void MonoVO::runRANSAC(int max_iterations,
                       const std::vector<cv::KeyPoint>& reference_keypoints,
                       const std::vector<cv::KeyPoint>& current_keypoints,
                       cv::Mat* F,
                       std::vector<int>* inliers)
{
  ros::WallTime start = ros::WallTime::now();

  int num_points = 7;
  ROS_ASSERT(reference_keypoints.size() >= num_points);
  ROS_ASSERT(current_keypoints.size() == reference_keypoints.size());

  // initialize
  inliers->clear();

  // initialize shared variables
  int best_num_inliers = 0;
  double best_error = std::numeric_limits<double>::max();

  cv::Mat best_F;
  std::vector<int> best_inliers;

  int num_iterations = max_iterations;
  int count = -1;
  bool stop = false;

  // parallelized while loop
  #pragma omp parallel shared(num_iterations, count, stop, best_error, best_num_inliers, best_F, best_inliers)
  {
    // Each thread will continue to independently do random samples until the stopping criteria has been reached by
    // any thread. Stopping criteria is from Algorithm 4.5 in [Hartley,2003].
    while (!stop)
    {
      // get random sample
      std::vector<cv::KeyPoint> reference_sample;
      std::vector<cv::KeyPoint> current_sample;
      getRandomSample(num_points, reference_keypoints, current_keypoints, &reference_sample, &current_sample);

      // get potential solutions for sample
      std::vector<cv::Mat> F_temp;
      switch (num_points)
      {
      case 7:
        sampleSolution7pt(reference_sample, current_sample, &F_temp);
        break;
      case 8:
        sampleSolution8pt(reference_sample, current_sample, &F_temp);
        break;
      }

      // find best solution for the sample
      int best_num_inliers_temp = 0;
      double best_error_temp = std::numeric_limits<double>::max();
      cv::Mat best_F_temp;
      std::vector<int> best_inliers_temp;
      for (size_t i = 0; i < F_temp.size(); i++)
      {
        std::vector<int> inliers_temp;
        std::vector<double> errors_temp;
        double error_temp = solutionError(reference_keypoints, current_keypoints, F_temp[i], &errors_temp, &inliers_temp);
        if (inliers_temp.size() > best_num_inliers_temp
            || (inliers_temp.size() == best_num_inliers_temp && error_temp < best_error_temp))
        {
          best_num_inliers_temp = inliers_temp.size();
          best_error_temp = error_temp;
          best_F_temp = F_temp[i];
          best_inliers_temp = inliers_temp;
        }
      }

      #pragma omp critical
      {
        // update global thread count
        count++;

        // is this solution better than global best?
        if (best_num_inliers_temp > best_num_inliers
            || (best_num_inliers_temp == best_num_inliers && best_error_temp < best_error))
        {
          // update global best
          best_num_inliers = best_num_inliers_temp;
          best_error = best_error_temp;
          best_F = best_F_temp;
          best_inliers = best_inliers_temp;

          // adaptively adjust number of RANSAC samples (Algorithm 4.5 in [Hartley,2003])
          double epsilon = 1 - ((double) best_num_inliers) / reference_keypoints.size();
          int new_num_iterations = std::ceil(std::log10(0.01) / std::log10(1.0 - std::pow(1.0 - epsilon, num_points)));
          if (new_num_iterations > 0 && new_num_iterations < num_iterations)
            num_iterations = new_num_iterations;
        }

        // check if we've completed the specified number of iterations
        if (count >= num_iterations)
        {
          stop = true;
          #pragma omp flush(stop)
        }

        // ROS_INFO("Thread %d, iteration %d, stop=%s", omp_get_thread_num(), count, (stop ? "true" : "false"));
      }
    }
  }

  // return final values
  *F = best_F;
  *inliers = best_inliers;

  ros::WallDuration compute_time = ros::WallTime::now() - start;
  ROS_INFO_THROTTLE(1, "%d iterations of RANSAC took %f seconds, returned %d inliers", num_iterations, compute_time.toSec(), (int)(*inliers).size());
}

double MonoVO::sampsonDistance(const double x[2], const double xp[2], const double F[3][3])
{
  // compute relevant entries of F*x
  double Fx1 = F[0][0]*x[0] + F[0][1]*x[1] + F[0][2];
  double Fx2 = F[1][0]*x[0] + F[1][1]*x[1] + F[1][2];

  // compute relevant entries of Ft*xp
  double Ftxp1 = F[0][0]*xp[0] + F[1][0]*xp[1] + F[2][0];
  double Ftxp2 = F[0][1]*xp[0] + F[1][1]*xp[1] + F[2][1];

  // compute xpt*F*x, taking advantage of precomputed values of F*x
  double xptFx = xp[0]*Fx1 + xp[1]*Fx2 + F[2][0]*x[0] + F[2][1]*x[1] + F[2][2];

  // compute Sampson distance ([Hartley,2003], Equation (11.9))
  return xptFx*xptFx / (Fx1*Fx1 + Fx2*Fx2 + Ftxp1*Ftxp1 + Ftxp2*Ftxp2);
}

cv::Mat MonoVO::triangulatePoint(const cv::Point2d& x, const cv::Point2d& xp, const cv::Mat& Pp)
{
  cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);

  A.at<double>(0,0) = -1.0;
  A.at<double>(0,2) = x.x;

  A.at<double>(1,1) = -1.0;
  A.at<double>(1,2) = x.y;

  A.at<double>(2,0) = xp.x*Pp.at<double>(2,0) - Pp.at<double>(0,0);
  A.at<double>(2,1) = xp.x*Pp.at<double>(2,1) - Pp.at<double>(0,1);
  A.at<double>(2,2) = xp.x*Pp.at<double>(2,2) - Pp.at<double>(0,2);
  A.at<double>(2,3) = xp.x*Pp.at<double>(2,3) - Pp.at<double>(0,3);

  A.at<double>(3,0) = xp.y*Pp.at<double>(2,0) - Pp.at<double>(1,0);
  A.at<double>(3,1) = xp.y*Pp.at<double>(2,1) - Pp.at<double>(1,1);
  A.at<double>(3,2) = xp.y*Pp.at<double>(2,2) - Pp.at<double>(1,2);
  A.at<double>(3,3) = xp.y*Pp.at<double>(2,3) - Pp.at<double>(1,3);

  cv::SVD svd(A);

  return (cv::Mat_<double>(4,1) << svd.vt.at<double>(3,0),
                                   svd.vt.at<double>(3,1),
                                   svd.vt.at<double>(3,2),
                                   svd.vt.at<double>(3,3));
}

double MonoVO::pointDepth(const cv::Mat &X, const cv::Mat &P)
{
  // sign(det M)
  int sign = (cv::determinant(P(cv::Range(0,3), cv::Range(0,3))) > 0) ? 1 : -1;

  // ||m3||
  double norm_m3 = cv::norm(P(cv::Range(0,3), cv::Range(2,3)));

  // w and T
  cv::Mat x = P*X;
  double w = x.at<double>(2,0);
  double T = X.at<double>(3,0);

  // depth (Equation (6.15) in [Hartley,2003])
  return sign * w / (T * norm_m3);
}

} // namespace mono_vo
