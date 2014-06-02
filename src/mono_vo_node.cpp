/**
 * \file mono_vo_node.cpp
 * \author Daniel Koch
 * \brief Entry point for the mono_vo node
 */

#include <ros/ros.h>
#include "mono_vo.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mono_vo");

  mono_vo::MonoVO vo;
  ros::spin();

  return 0;
}
