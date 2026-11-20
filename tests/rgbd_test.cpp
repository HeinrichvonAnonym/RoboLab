#include "rgbd_fixture.h"
#include <gtest/gtest.h>
#include <limits>

using namespace robo_lab::camera_support;

TEST(CaptureConfig, LegacyAndDeviceSelection) {
  auto c=CaptureConfig::load(YAML::LoadFile(std::string(ROBOLAB_SOURCE_DIR)+"/config/plugins/realsense_plugin_config.yaml"));
  EXPECT_EQ(c.fps,15); EXPECT_EQ(c.width,640); EXPECT_TRUE(c.rgb); EXPECT_TRUE(c.depth); EXPECT_FALSE(c.publish_rgbd);
  c=CaptureConfig::load(YAML::Load("serial: test-serial\nwidth: 848\nheight: 480\nfps: 60\npublish_rgbd: true"));
  EXPECT_EQ(c.serial,"test-serial"); EXPECT_EQ(c.width,848); EXPECT_EQ(c.fps,60); EXPECT_TRUE(c.publish_rgbd);
}
TEST(CaptureConfig, InvalidConfiguration) {
  for(const char* text:{"fps: 0","fps: 29.5","fps: .nan","width: -1","enable_rgb: false\nenable_depth: false\nenable_ir: false","publish_rgbd: true\nenable_depth: false","max_pair_skew_ms: -1","topic: []"})
    EXPECT_THROW(CaptureConfig::load(YAML::Load(text)),std::exception)<<text;
}
TEST(RgbdPair, MissingDuplicateSkewAndDomains) {
  auto f=rgbd_fixture(); PairGate p;
  EXPECT_TRUE(p.accept(f.color(),f.depth(),5)); EXPECT_FALSE(p.accept(f.color(),f.depth(),5));
  f.mutable_color()->set_frame_number(2); EXPECT_FALSE(p.accept(f.color(),f.depth(),5));
  f.mutable_depth()->set_frame_number(2); f.mutable_depth()->set_device_timestamp_ms(106);
  EXPECT_FALSE(p.accept(f.color(),f.depth(),5)); f.mutable_depth()->set_device_timestamp_ms(105);
  EXPECT_TRUE(p.accept(f.color(),f.depth(),5));
  f.mutable_color()->set_frame_number(3); f.mutable_depth()->set_frame_number(3); f.mutable_color()->set_timestamp_domain(1);
  EXPECT_FALSE(p.accept(f.color(),f.depth(),5)); f.mutable_color()->set_timestamp_domain(0); f.mutable_depth()->clear_data();
  EXPECT_FALSE(p.accept(f.color(),f.depth(),5));
}
TEST(RgbdWire, PaddingUnitsAndRoundTrip) {
  auto f=rgbd_fixture(8,8,5); std::string bytes; ASSERT_TRUE(f.SerializeToString(&bytes));
  camera::RgbdFrame copy; ASSERT_TRUE(copy.ParseFromString(bytes)); EXPECT_NO_THROW(validate_frame(copy));
  const auto mesh=build_mesh(copy,PreviewConfig{});
  EXPECT_EQ(mesh.valid_depth_pixels,64); EXPECT_EQ(mesh.indices.size(),7*7*6);
  EXPECT_NEAR(mesh.vertices[0].position.z(),0.20,1e-6);
  EXPECT_NEAR(mesh.vertices[0].position.x(),-0.0875,1e-6);
}
TEST(RgbdWire, D405InverseBrownColorCalibration) {
  auto f=rgbd_fixture();
  // librealsense 2.56.5 supports this model for projection, and D405 reports it.
  f.mutable_color_intrinsics()->set_distortion_model(2);
  f.mutable_depth_intrinsics()->set_distortion_model(2);
  EXPECT_EQ(build_mesh(f,PreviewConfig{}).indices.size(),7*7*6);
}
TEST(StereoGeometry, UsesSeparateIntrinsicsAndDepthToColorExtrinsics) {
  auto f=rgbd_fixture();
  // Rotate 90 degrees around Z, then translate 1 cm in color-camera X.
  const float rotation[]={0,1,0,-1,0,0,0,0,1};
  for(int i=0;i<9;++i) f.mutable_depth_to_color()->set_rotation(i,rotation[i]);
  f.mutable_depth_to_color()->set_translation_m(0,0.01f);
  f.mutable_color_intrinsics()->set_fx(4);
  f.mutable_color_intrinsics()->set_fy(4);
  const auto mesh=build_mesh(f,PreviewConfig{});
  const auto& vertex=mesh.vertices[0];
  EXPECT_NEAR(vertex.position.x(),0.0975f,1e-6);
  EXPECT_NEAR(vertex.position.y(),-0.0875f,1e-6);
  EXPECT_NEAR(vertex.position.z(),0.2f,1e-6);
  EXPECT_NEAR(vertex.uv.x(),(4*0.0975f/0.2f+4)/8,1e-6);
  EXPECT_NEAR(vertex.uv.y(),(4*-0.0875f/0.2f+4)/8,1e-6);
}
TEST(RgbdWire, RejectMalformedCalibrationAndPayload) {
  auto f=rgbd_fixture(); f.mutable_depth()->set_stride(1); EXPECT_THROW(validate_frame(f),std::invalid_argument);
  f=rgbd_fixture(); f.set_depth_units_m(0); EXPECT_THROW(validate_frame(f),std::invalid_argument);
  f=rgbd_fixture(); f.mutable_depth_intrinsics()->set_fx(0); EXPECT_THROW(validate_frame(f),std::invalid_argument);
  f=rgbd_fixture(); f.mutable_depth_to_color()->set_rotation(0,3); EXPECT_THROW(validate_frame(f),std::invalid_argument);
  f=rgbd_fixture(); f.mutable_color_intrinsics()->set_coefficients(0,std::numeric_limits<float>::quiet_NaN()); EXPECT_THROW(validate_frame(f),std::invalid_argument);
}
TEST(StereoGeometry, ZeroEyeAndConvergenceAndDisparitySign) {
  auto f=rgbd_fixture(); PreviewConfig c;
  const auto l=eye_matrix(f.color_intrinsics(),c,-0.009f),r=eye_matrix(f.color_intrinsics(),c,0.009f);
  auto project=[](const Eigen::Matrix4f& m,float z){Eigen::Vector4f v=m*Eigen::Vector4f(0,0,z,1);return v.x()/v.w();};
  EXPECT_GT(project(l,0.2f),project(r,0.2f));
  EXPECT_NEAR(project(l,c.convergence_m),project(r,c.convergence_m),1e-6);
  EXPECT_LT(project(l,0.4f),project(r,0.4f));
  c.eye_separation_m=0;
  EXPECT_TRUE(eye_matrix(f.color_intrinsics(),c,-c.eye_separation_m/2).isApprox(eye_matrix(f.color_intrinsics(),c,c.eye_separation_m/2)));
}
TEST(StereoGeometry, ClipHolesAndDepthEdges) {
  auto f=rgbd_fixture(); PreviewConfig c;
  c.max_depth_m=0.15f; EXPECT_TRUE(build_mesh(f,c).indices.empty());
  c.max_depth_m=0.5f; c.min_depth_m=0.25f; EXPECT_TRUE(build_mesh(f,c).indices.empty()); c.min_depth_m=0.07f;
  auto* d=f.mutable_depth()->mutable_data(); (*d)[2*(3*8+3)]=0;(*d)[2*(3*8+3)+1]=0;
  auto mesh=build_mesh(f,c); EXPECT_EQ(mesh.valid_depth_pixels,63);
  for(auto i:mesh.indices) EXPECT_NE(i,27);
  f=rgbd_fixture(); d=f.mutable_depth()->mutable_data();
  for(int y=0;y<8;++y)for(int x=4;x<8;++x){(*d)[2*(y*8+x)]=4000&255;(*d)[2*(y*8+x)+1]=4000>>8;}
  mesh=build_mesh(f,c);
  for(size_t t=0;t<mesh.indices.size();t+=3) {
    const float a=mesh.vertices[mesh.indices[t]].position.z();
    for(int i=1;i<3;++i) EXPECT_NEAR(a,mesh.vertices[mesh.indices[t+i]].position.z(),1e-6);
  }
}
